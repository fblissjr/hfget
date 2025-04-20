# hf_organizer_core.py
import os
import shutil
import json
import time
import datetime
import logging
import re
import fnmatch
from pathlib import Path
from typing import Dict, Optional, Union, List, Any, Tuple

import structlog
import humanize

from huggingface_hub import (
    HfApi,
    snapshot_download,
    hf_hub_download,
    list_repo_files,
    scan_cache_dir,
    HFCacheInfo,
    CommitInfo,
)
# Import RepoFile from the correct submodule
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError, hf_raise_for_status

# --- Configuration ---
DEFAULT_CONFIG_PATH = "~/.config/hf_organizer/config.json"
DEFAULT_STRUCTURED_ROOT = "~/huggingface_organized"
# DEFAULT_HF_HOME removed - will determine dynamically or use fallback

METADATA_FILENAME = "downloads.json"
METADATA_DIR_NAME = ".metadata"

# Define the ultimate fallback path here if needed elsewhere, or inline it.
FALLBACK_HF_HOME = "~/.cache/huggingface"

class HfHubOrganizer:
    """
    Core class to manage HuggingFace Hub downloads, cache, and organization.
    Uses huggingface_hub library functions where possible.
    Designed to be used by different interfaces (CLI, Web).
    """

    # Default environment variable values - use fallback path here
    ENV_VARS = {
        "HF_HOME": FALLBACK_HF_HOME, # Ultimate fallback if nothing else sets it
        "HF_HUB_CACHE": "${HF_HOME}/hub",
        "HF_ASSETS_CACHE": "${HF_HOME}/assets",
        "HF_TOKEN": None,
        "HF_HUB_VERBOSITY": "warning",
        "HF_HUB_ETAG_TIMEOUT": "10",
        "HF_HUB_DOWNLOAD_TIMEOUT": "10"
    }

    BOOLEAN_ENV_VARS = {
        "HF_DEBUG": False,
        "HF_HUB_OFFLINE": False,
        "HF_HUB_DISABLE_PROGRESS_BARS": False,
        "HF_HUB_DISABLE_TELEMETRY": True,
        "HF_HUB_ENABLE_HF_TRANSFER": True
    }

    def __init__(
        self,
        profile: Optional[str] = None,
        base_path: Optional[str] = None,
        structured_root: Optional[str] = None,
        token: Optional[str] = None,
        enable_hf_transfer: Optional[bool] = None,
        verbose: bool = False,
        config_path: Optional[str] = None,
        log_format: str = "console" # Keep log format option for core
    ):
        """Initialize with custom paths and settings."""
        self.config_path = os.path.expanduser(config_path or DEFAULT_CONFIG_PATH)
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        # Setup logger first
        self.logger = self._setup_logger(verbose, log_format)

        self.config = self._load_config()
        self.selected_profile = profile
        profile_settings = {}

        if profile:
            if profile not in self.config.get("profiles", {}):
                self.logger.error("profile_not_found", profile=profile,
                                  available=list(self.config.get("profiles", {}).keys()))
                raise ValueError(f"Profile '{profile}' not found.")
            profile_settings = self.config["profiles"][profile]
            self.logger.debug("using_profile_settings", profile=profile)

        # Determine effective settings (CLI/Constructor > Profile > Environment > Fallback Default)
        # 1. Constructor argument `base_path`
        # 2. Profile setting `base_path`
        # 3. Environment variable `HF_HOME`
        # 4. Fallback default path
        effective_base_path = base_path or profile_settings.get("base_path") or os.environ.get("HF_HOME") or FALLBACK_HF_HOME # Added fallback here

        effective_structured_root = structured_root or profile_settings.get("structured_root") or DEFAULT_STRUCTURED_ROOT
        effective_token = token or profile_settings.get("token") or os.environ.get("HF_TOKEN")

        if enable_hf_transfer is None:
             effective_enable_hf_transfer = profile_settings.get("enable_hf_transfer", self.BOOLEAN_ENV_VARS["HF_HUB_ENABLE_HF_TRANSFER"])
        else:
             effective_enable_hf_transfer = enable_hf_transfer

        # Set base HF path if needed (use the determined effective path)
        # os.environ["HF_HOME"] will be set here if not already set by the environment
        # The _initialize_env_vars call later will respect this if already set.
        if "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = os.path.expanduser(effective_base_path)
        # If HF_HOME *was* already set in the environment, we respect it and don't overwrite here.
        # effective_base_path will have picked it up anyway.

        self.structured_root = os.path.expanduser(effective_structured_root)
        os.makedirs(self.structured_root, exist_ok=True)

        # Set token if needed
        if effective_token and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = effective_token

        # Initialize all other environment variables
        # This will set HF_HUB_CACHE based on the final HF_HOME if not already set.
        # It will also set HF_HOME to the fallback default *only if* it wasn't set by env/profile/arg before.
        self._initialize_env_vars(force_hf_transfer_setting=effective_enable_hf_transfer)

        # Keep track of effective paths (re-read from env after initialization)
        hf_home_final = os.environ["HF_HOME"] # Read the final value from env
        hf_hub_cache_default = os.path.join(hf_home_final, "hub")
        hf_hub_cache_final = os.environ.get("HF_HUB_CACHE", hf_hub_cache_default)
        if "HF_HUB_CACHE" not in os.environ or not os.environ["HF_HUB_CACHE"]:
             os.environ["HF_HUB_CACHE"] = hf_hub_cache_final

        self.effective_paths = {
            "HF_HOME": hf_home_final,
            "HF_HUB_CACHE": hf_hub_cache_final,
            "structured_root": self.structured_root
        }

        # Bind core context to logger
        self.logger = self.logger.bind(
             profile=profile or "Default",
             hf_home=self.effective_paths["HF_HOME"],
             cache=self.effective_paths["HF_HUB_CACHE"],
             org_root=self.structured_root,
             hf_transfer=os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"
        )
        self.logger.info("organizer_initialized")

        # Initialize HF API
        self.api = HfApi(token=os.environ.get("HF_TOKEN"))

    def _setup_logger(self, verbose: bool, format_type: str) -> structlog.BoundLogger:
        """Set up structured logging."""
        log_level = logging.DEBUG if verbose else logging.INFO

        # Common processors
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        ]

        # Format-specific processors
        if format_type == "json":
            processors.extend([
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ])
            formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(sort_keys=True),
                foreign_pre_chain=processors,
            )
        else: # console or structured (use console for both for now)
            processors.extend([
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ])
            formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=(format_type == "console"), exception_formatter=structlog.dev.plain_traceback),
                foreign_pre_chain=processors,
            )

        # Configure standard logging handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        # Avoid adding handler multiple times if logger is re-initialized
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
             root_logger.addHandler(handler)
        root_logger.setLevel(log_level)

        # Configure structlog
        structlog.configure(
            processors=[structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        # Return a logger instance specific to this class
        return structlog.get_logger(self.__class__.__name__)


    def _load_config(self) -> Dict[str, Any]:
        """Load config from disk or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.debug("config_loaded", path=self.config_path)
                if not isinstance(config.get("profiles"), dict):
                    self.logger.warning("invalid_profiles_section_resetting", path=self.config_path)
                    config["profiles"] = {}
                return config
            except json.JSONDecodeError:
                self.logger.warning("invalid_config_file_format", path=self.config_path, action="creating_default")
            except Exception as e:
                self.logger.error("config_load_failed", path=self.config_path, error=str(e), action="creating_default")
        else:
             self.logger.debug("config_file_not_found_creating_default", path=self.config_path)
        return {"profiles": {}}

    def _save_config(self):
        """Save current config to disk."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2, sort_keys=True)
            self.logger.debug("config_saved", path=self.config_path)
        except Exception as e:
            self.logger.error("config_save_failed", path=self.config_path, error=str(e))

    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        """List all available profiles with their details."""
        return self.config.get("profiles", {})

    def add_profile(
        self,
        name: str,
        base_path: Optional[str] = None,
        structured_root: Optional[str] = None,
        token: Optional[str] = None,
        enable_hf_transfer: Optional[bool] = None,
        description: Optional[str] = None
    ):
        """Add or update a profile."""
        if "profiles" not in self.config:
            self.config["profiles"] = {}

        profile_data = {
            "base_path": os.path.expanduser(base_path) if base_path else None,
            "structured_root": os.path.expanduser(structured_root) if structured_root else None,
            "token": token,
            "enable_hf_transfer": enable_hf_transfer,
            "description": description or f"Profile '{name}'"
        }
        profile_data = {k: v for k, v in profile_data.items() if v is not None}

        self.config["profiles"][name] = profile_data
        self._save_config()
        self.logger.info("profile_added_or_updated", name=name, details=profile_data)

    def remove_profile(self, name: str):
        """Remove a profile."""
        if "profiles" in self.config and name in self.config["profiles"]:
            del self.config["profiles"][name]
            self._save_config()
            self.logger.info("profile_removed", name=name)
            return True
        else:
            self.logger.warning("profile_not_found_for_removal", name=name)
            return False

    def _initialize_env_vars(self, force_hf_transfer_setting: Optional[bool] = None):
        """Initialize environment variables with defaults if not already set."""
        # Ensure HF_HOME is handled first if not already set in the environment
        if "HF_HOME" not in os.environ:
             # Use the fallback default defined in ENV_VARS
             os.environ["HF_HOME"] = os.path.expanduser(self.ENV_VARS["HF_HOME"])
             self.logger.debug("env_var_set_default_string", key="HF_HOME", value=os.environ["HF_HOME"])

        # Process other string environment variables
        for key, default_value in self.ENV_VARS.items():
            if key == "HF_HOME": continue # Already handled

            if key not in os.environ and default_value is not None:
                # Handle variable expansion like ${HF_HOME}
                # Use current os.environ for expansion lookup (HF_HOME is now guaranteed to be set)
                env_dict = os.environ
                expanded_value = os.path.expandvars(default_value.replace("${HF_HOME}", env_dict.get("HF_HOME", "")))

                # Expand user path ~
                final_value = os.path.expanduser(expanded_value)
                os.environ[key] = final_value
                # Avoid logging token if it's the default None
                if key != "HF_TOKEN" or final_value is not None:
                     self.logger.debug("env_var_set_default_string", key=key, value=final_value)

        # Process boolean environment variables
        for key, default_value in self.BOOLEAN_ENV_VARS.items():
             # Special handling for HF_HUB_ENABLE_HF_TRANSFER based on forced setting
             if key == "HF_HUB_ENABLE_HF_TRANSFER" and force_hf_transfer_setting is not None:
                  current_value = "1" if force_hf_transfer_setting else "0"
                  if os.environ.get(key) != current_value:
                       os.environ[key] = current_value
                       self.logger.debug("env_var_forced_bool", key=key, value=current_value)
             elif key not in os.environ:
                  os.environ[key] = "1" if default_value else "0"
                  self.logger.debug("env_var_set_default_bool", key=key, value=os.environ[key])

    def _determine_category_and_paths(self, repo_id: str, category: Optional[str] = None, subfolder: Optional[str] = None) -> Tuple[str, str, str, str]:
        """Determine the category and create the organized path structure."""
        repo_type_guess = "model"  # Default to model
        namespace = "library"
        repo_name = repo_id

        if "/" in repo_id:
            namespace, repo_name = repo_id.split("/", 1)
        else:
            pass  # Will attempt repo_info lookup

        if category is None:
            try:
                repo_info = self.api.repo_info(repo_id=repo_id)
                # FIX 1: Use getattr for robustness against missing attribute
                repo_type_from_api = getattr(repo_info, "repo_type", None)

                if "/" not in repo_id and "/" in repo_info.id:
                     namespace, repo_name = repo_info.id.split("/", 1)

                if repo_type_from_api == "dataset":
                    repo_type_guess = "dataset"
                elif repo_type_from_api == "space":
                    repo_type_guess = "space"
                elif repo_type_from_api == "model":  # Explicitly check 'model'
                    repo_type_guess = "model"
                elif repo_type_from_api is not None:
                    self.logger.warning(
                        "unrecognized_repo_type_api",
                        repo_id=repo_info.id,
                        type=repo_type_from_api,
                        fallback=repo_type_guess,
                    )

                self.logger.debug("category_detected_via_api", repo_id=repo_info.id, category=repo_type_guess)

            except RepositoryNotFoundError:
                self.logger.error(
                    "repo_not_found_api", repo_id=repo_id, action="assuming_model"
                )
                repo_type_guess = "models"
            except HfHubHTTPError as http_err:
                if http_err.response.status_code == 401:
                    self.logger.error(
                        "authentication_error_api", repo_id=repo_id, error=str(http_err)
                    )
                    raise ValueError(
                        f"Authentication failed for {repo_id}. Check your HF_TOKEN."
                    ) from http_err
                else:
                    self.logger.warning(
                        "category_detection_http_error",
                        repo_id=repo_id,
                        status=http_err.response.status_code,
                        error=str(http_err),
                        fallback=repo_type_guess,
                    )
            except Exception as e:
                # Catch AttributeError here specifically if getattr wasn't used, or other errors
                self.logger.warning(
                    "category_detection_failed_api",
                    repo_id=repo_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    fallback=repo_type_guess,
                )
        else:
            repo_type_guess = category # User override

        # Create organized base path using potentially updated namespace/repo_name
        org_repo_path = os.path.join(
            self.structured_root,
            repo_type_guess,
            namespace,
            repo_name
        )
        org_download_path = os.path.join(org_repo_path, subfolder) if subfolder else org_repo_path
        os.makedirs(org_download_path, exist_ok=True)
        self.logger.debug("organizing_files_target", path=org_download_path)

        return org_repo_path, org_download_path, repo_type_guess, namespace


    def _link_or_copy(self, cache_path: str, org_path: str, symlink_to_cache: bool):
        """Helper to symlink or copy a file/directory."""
        org_dir = os.path.dirname(org_path)
        if not os.path.exists(org_dir):
             os.makedirs(org_dir, exist_ok=True)
             self.logger.debug("created_parent_dir", path=org_dir)

        if os.path.lexists(org_path):
            is_link = os.path.islink(org_path)
            remove_existing = True
            if is_link:
                 try:
                      link_target = os.readlink(org_path)
                      if link_target == os.path.abspath(cache_path):
                           self.logger.debug("target_already_correct_symlink", path=org_path)
                           remove_existing = False
                 except OSError: pass # Broken link, remove

            if remove_existing:
                try:
                    if os.path.isfile(org_path) or is_link:
                        os.remove(org_path)
                        self.logger.debug("removed_existing_target_link_or_file", path=org_path)
                    elif os.path.isdir(org_path):
                        shutil.rmtree(org_path)
                        self.logger.debug("removed_existing_target_dir", path=org_path)
                except OSError as e:
                    self.logger.error("failed_removing_existing_target", path=org_path, error=str(e))
                    raise

        if symlink_to_cache:
            try:
                abs_cache_path = os.path.abspath(cache_path)
                os.symlink(abs_cache_path, org_path)
                self.logger.debug("symlink_created", source=abs_cache_path, target=org_path)
            except OSError as e:
                 self.logger.error("symlink_failed", source=cache_path, target=org_path, error=str(e))
                 raise
        else:
            try:
                if os.path.isdir(cache_path):
                    shutil.copytree(cache_path, org_path, symlinks=True)
                    self.logger.debug("directory_copied", source=cache_path, target=org_path)
                elif os.path.isfile(cache_path):
                    shutil.copy2(cache_path, org_path)
                    self.logger.debug("file_copied", source=cache_path, target=org_path)
                else:
                    self.logger.warning("copy_source_not_found_or_not_file_or_dir", source=cache_path)
            except Exception as e:
                self.logger.error("copy_failed", source=cache_path, target=org_path, error=str(e))
                raise

    def download(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        category: Optional[str] = None,
        symlink_to_cache: bool = True,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Download repo/file and organize it."""
        start_time = time.time()
        log_ctx = {"repo_id": repo_id, "filename": filename or "entire_repo", "subfolder": subfolder, "revision": revision or "main"}
        self.logger.info("download_started", **log_ctx)

        try:
            org_repo_path, org_download_path, detected_category, _ = self._determine_category_and_paths(
                repo_id, category, subfolder
            )
            log_ctx["category"] = detected_category

            self._save_download_metadata(org_repo_path, repo_id, detected_category, filename or "entire_repo", subfolder, revision)

            downloaded_path_in_cache: str
            final_organized_path: str

            if filename:
                self.logger.debug("downloading_single_file", **log_ctx)
                downloaded_path_in_cache = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder,
                    revision=revision,
                    repo_type=detected_category
                    if detected_category in ["model", "dataset", "space"]
                    else None,
                    token=self.api.token,
                    **kwargs,
                )
                final_organized_path = os.path.join(org_download_path, os.path.basename(filename))
                self._link_or_copy(downloaded_path_in_cache, final_organized_path, symlink_to_cache)
            else:
                # FIX 2: Remove unsupported 'subfolder' argument from snapshot_download
                self.logger.debug("downloading_snapshot", allow_patterns=allow_patterns, ignore_patterns=ignore_patterns, **log_ctx)
                # Note: snapshot_download downloads the whole repo (respecting patterns), not just a subfolder.
                downloaded_path_in_cache = snapshot_download(
                    repo_id=repo_id,
                    # subfolder=subfolder, # <--- Removed this line
                    revision=revision,
                    repo_type=detected_category
                    if detected_category in ["model", "dataset", "space"]
                    else None,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    token=self.api.token,
                    **kwargs,
                )
                # If a subfolder was requested by the user, the final organized path still includes it,
                # but the download itself fetched the whole repo into the cache snapshot.
                # We will link/copy the entire snapshot contents into the target (sub)folder.
                final_organized_path = org_download_path

                if os.path.lexists(final_organized_path):
                    if os.path.islink(final_organized_path): os.remove(final_organized_path)
                    elif os.path.isdir(final_organized_path):
                         for item in os.listdir(final_organized_path):
                              item_path = os.path.join(final_organized_path, item)
                              if os.path.islink(item_path) or os.path.isfile(item_path): os.remove(item_path)
                              elif os.path.isdir(item_path): shutil.rmtree(item_path)
                         self.logger.debug("cleared_existing_target_dir_contents", path=final_organized_path)
                    else: os.remove(final_organized_path)
                else: os.makedirs(final_organized_path, exist_ok=True)

                item_count = 0
                items_in_cache = os.listdir(downloaded_path_in_cache)
                for item_name in items_in_cache:
                    cache_item_path = os.path.join(downloaded_path_in_cache, item_name)
                    # Link/copy items directly into the target path (which might be a subfolder)
                    org_item_path = os.path.join(final_organized_path, item_name)
                    self._link_or_copy(cache_item_path, org_item_path, symlink_to_cache)
                    item_count += 1
                self.logger.debug("snapshot_contents_processed", count=item_count, source=downloaded_path_in_cache, target=final_organized_path, items=len(items_in_cache))

            elapsed = time.time() - start_time
            self.logger.info("download_completed", organized_path=final_organized_path, elapsed_seconds=round(elapsed, 2), symlinked=symlink_to_cache, **log_ctx)
            return final_organized_path

        except RepositoryNotFoundError:
             self.logger.error("download_failed_repo_not_found", **log_ctx)
             raise
        except HfHubHTTPError as http_err:
             self.logger.error("download_failed_http_error", status=http_err.response.status_code, error=str(http_err), **log_ctx)
             raise
        except Exception as e:
            self.logger.exception("download_failed_unexpected", error=str(e), **log_ctx)
            raise

    def download_recent(
        self,
        repo_id: str,
        days_ago: int,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        category: Optional[str] = None,
        symlink_to_cache: bool = True,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Download recently modified files."""
        start_time = time.time()
        log_ctx = {"repo_id": repo_id, "days_ago": days_ago, "subfolder": subfolder, "revision": revision or "main"}
        self.logger.info("download_recent_started", **log_ctx)

        try:
            org_repo_path, org_download_path, detected_category, _ = self._determine_category_and_paths(
                repo_id, category, subfolder
            )
            log_ctx["category"] = detected_category

            cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_ago)
            self.logger.debug("filtering_commits_since", cutoff_date=cutoff_date.isoformat(), **log_ctx)

            all_files_info: List[RepoFile] = list_repo_files(
                repo_id=repo_id, revision=revision, token=self.api.token
            )
            self.logger.debug("retrieved_repo_file_info", count=len(all_files_info), **log_ctx)

            recent_files_to_download: List[RepoFile] = []
            for file_info in all_files_info:
                is_recent = file_info.last_commit and file_info.last_commit.date and file_info.last_commit.date > cutoff_date
                if not is_recent: continue

                in_subfolder = True
                if subfolder:
                    norm_subfolder = subfolder.strip('/')
                    in_subfolder = file_info.path == norm_subfolder or file_info.path.startswith(norm_subfolder + '/')
                if not in_subfolder: continue

                path_matches = True
                if allow_patterns: path_matches = any(fnmatch.fnmatch(file_info.path, pattern) for pattern in allow_patterns)
                if path_matches and ignore_patterns:
                     if any(fnmatch.fnmatch(file_info.path, pattern) for pattern in ignore_patterns): path_matches = False
                if not path_matches:
                     self.logger.debug("file_skipped_by_pattern", file=file_info.path, **log_ctx)
                     continue

                recent_files_to_download.append(file_info)
                self.logger.debug("file_marked_for_recent_download", file=file_info.path, commit_date=file_info.last_commit.date, **log_ctx)

            if not recent_files_to_download:
                self.logger.info("no_recent_files_found_matching_criteria", **log_ctx)
                self._save_download_metadata(org_repo_path, repo_id, detected_category, f"recent_{days_ago}d", subfolder, revision)
                return org_download_path

            self.logger.info("downloading_recent_files", count=len(recent_files_to_download), **log_ctx)
            downloaded_count = 0
            failed_count = 0
            for file_info in recent_files_to_download:
                try:
                    file_repo_subfolder = os.path.dirname(file_info.path)
                    file_basename = os.path.basename(file_info.path)
                    self.logger.debug("downloading_individual_recent_file", file=file_info.path, **log_ctx)
                    downloaded_path_in_cache = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_basename,
                        subfolder=file_repo_subfolder or None,
                        revision=revision,
                        repo_type=detected_category
                        if detected_category in ["model", "dataset", "space"]
                        else None,
                        token=self.api.token,
                        **kwargs,
                    )

                    relative_file_path = file_info.path
                    if subfolder:
                         norm_subfolder = subfolder.strip('/')
                         if file_info.path.startswith(norm_subfolder + '/'): relative_file_path = os.path.relpath(file_info.path, norm_subfolder)
                         elif file_info.path == norm_subfolder: relative_file_path = os.path.basename(file_info.path)

                    final_organized_path = os.path.join(org_download_path, relative_file_path)
                    self._link_or_copy(downloaded_path_in_cache, final_organized_path, symlink_to_cache)
                    downloaded_count += 1
                except Exception as e_file:
                    failed_count += 1
                    self.logger.error("failed_downloading_recent_file", file=file_info.path, error=str(e_file), **log_ctx)

            elapsed = time.time() - start_time
            self.logger.info("download_recent_completed", files_downloaded=downloaded_count, files_failed=failed_count, target_path=org_download_path, elapsed_seconds=round(elapsed, 2), symlinked=symlink_to_cache, **log_ctx)
            self._save_download_metadata(org_repo_path, repo_id, detected_category, f"recent_{days_ago}d", subfolder, revision)
            return org_download_path

        except RepositoryNotFoundError:
             self.logger.error("download_failed_repo_not_found", **log_ctx)
             raise
        except HfHubHTTPError as http_err:
             self.logger.error("download_recent_failed_http_error", status=http_err.response.status_code, error=str(http_err), **log_ctx)
             raise
        except Exception as e:
            self.logger.exception("download_recent_failed_unexpected", error=str(e), **log_ctx)
            raise

    def _save_download_metadata(self, path: str, repo_id: str, category: str, download_type: str, subfolder: Optional[str] = None, revision: Optional[str] = None):
        """Save metadata about download."""
        metadata_dir = os.path.join(self.structured_root, METADATA_DIR_NAME)
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_file = os.path.join(metadata_dir, METADATA_FILENAME)
        metadata = {"downloads": []}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, dict) and isinstance(loaded_data.get("downloads"), list): metadata = loaded_data
                    else: self.logger.warning("invalid_metadata_format_resetting", path=metadata_file)
            except Exception as e: self.logger.error("failed_loading_metadata", path=metadata_file, error=str(e))

        entry = {k: v for k, v in {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repo_id": repo_id, "category": category, "type": download_type,
            "relative_path": os.path.relpath(path, self.structured_root),
            "profile": self.selected_profile, "revision": revision or "main",
            "subfolder": subfolder,
        }.items() if v is not None}
        metadata["downloads"].insert(0, entry)
        max_history = 500
        if len(metadata["downloads"]) > max_history: metadata["downloads"] = metadata["downloads"][:max_history]
        try:
            with open(metadata_file, 'w') as f: json.dump(metadata, f, indent=2, sort_keys=True)
        except Exception as e: self.logger.error("failed_saving_metadata", path=metadata_file, error=str(e))

    def list_downloads(self, limit: Optional[int] = None, category: Optional[str] = None, profile_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List download history."""
        metadata_file = os.path.join(self.structured_root, METADATA_DIR_NAME, METADATA_FILENAME)
        if not os.path.exists(metadata_file): return []
        try:
            with open(metadata_file, 'r') as f: metadata = json.load(f)
            downloads = metadata.get("downloads", [])
            if not isinstance(downloads, list): raise ValueError("Invalid format")
        except Exception as e:
             self.logger.error("failed_loading_metadata_for_list", path=metadata_file, error=str(e))
             return []

        filtered = [d for d in downloads if isinstance(d, dict) and
                    (not category or d.get("category") == category) and
                    (not profile_filter or d.get("profile") == profile_filter)]
        return filtered[:limit] if limit and limit > 0 else filtered

    def scan_cache(self) -> List[Dict[str, Any]]:
        """Scan cache using huggingface_hub.scan_cache_dir."""
        cache_dir = self.effective_paths["HF_HUB_CACHE"]
        self.logger.info("scanning_cache_with_hf_hub", path=cache_dir)
        try:
            scan = scan_cache_dir(cache_dir)
            self.logger.info("cache_scan_complete", repos=scan.repos_count, size=scan.size_on_disk_str)
            results = []
            for repo_info in scan.repos:
                 for revision_info in repo_info.revisions:
                      largest_file = {"name": "", "size": 0}
                      snapshot_path = Path(revision_info.snapshot_path)
                      try:
                           files = [p for p in snapshot_path.rglob('*') if p.is_file()]
                           for file_path in files:
                                try:
                                     size = file_path.stat().st_size
                                     if size > largest_file["size"]: largest_file = {"name": str(file_path.relative_to(snapshot_path)), "size": size}
                                except OSError: self.logger.warning("getsize_failed_cache_scan", path=str(file_path))
                      except OSError as e: self.logger.warning("failed_listing_snapshot_files", path=str(snapshot_path), error=str(e))
                      results.append({
                          "repo_id": repo_info.repo_id, "repo_type": repo_info.repo_type, "revision": revision_info.commit_hash,
                          "size_bytes": revision_info.size_on_disk, "size_human": revision_info.size_on_disk_str,
                          "file_count": len(revision_info.files), "largest_file": largest_file,
                          "last_modified": revision_info.last_modified.isoformat() if revision_info.last_modified else None,
                          "cache_path": str(revision_info.snapshot_path)
                      })
            results.sort(key=lambda x: x["size_bytes"], reverse=True)
            return results
        except Exception as e:
            self.logger.exception("cache_scan_failed_hf_hub", path=cache_dir, error=str(e))
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage."""
        cache_items = self.scan_cache()
        if not cache_items: return {"total_size": 0, "total_size_human": "0 B", "repo_count": 0, "snapshot_count": 0, "file_count": 0, "largest_snapshots": [], "organizations": []}
        total_size = sum(item["size_bytes"] for item in cache_items)
        total_files = sum(item["file_count"] for item in cache_items)
        unique_repos = {item["repo_id"] for item in cache_items}
        largest_snapshots = sorted(cache_items, key=lambda x: x["size_bytes"], reverse=True)[:5]
        orgs = {}
        for item in cache_items:
            org = item["repo_id"].split("/")[0] if "/" in item["repo_id"] else "library"
            if org not in orgs: orgs[org] = {"size_bytes": 0, "snapshot_count": 0}
            orgs[org]["size_bytes"] += item["size_bytes"]
            orgs[org]["snapshot_count"] += 1
        formatted_orgs = [{"name": org, "size_bytes": stats["size_bytes"], "size_human": humanize.naturalsize(stats["size_bytes"]), "snapshot_count": stats["snapshot_count"], "percentage": (stats["size_bytes"] / total_size * 100 if total_size > 0 else 0)} for org, stats in orgs.items()]
        top_orgs = sorted(formatted_orgs, key=lambda x: x["size_bytes"], reverse=True)
        return {"total_size": total_size, "total_size_human": humanize.naturalsize(total_size), "repo_count": len(unique_repos), "snapshot_count": len(cache_items), "file_count": total_files, "largest_snapshots": largest_snapshots, "organizations": top_orgs}

    def clean_cache(self, older_than_days: Optional[int] = None, min_size_mb: Optional[int] = None, dry_run: bool = False) -> Tuple[int, int, List[Dict]]:
        """Clean up cached snapshots."""
        cache_items = self.scan_cache()
        action = "dry_run" if dry_run else "clean_cache"
        log_ctx = {"older_than_days": older_than_days, "min_size_mb": min_size_mb, "dry_run": dry_run}
        self.logger.info(f"{action}_started", **log_ctx)
        if not cache_items: return (0, 0, [])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        freed_bytes, removed_count, removed_items_details = 0, 0, []
        items_to_remove = []
        for item in cache_items:
            meets_age, meets_size = False, False
            if older_than_days is not None and item["last_modified"]:
                try:
                    last_mod = datetime.datetime.fromisoformat(item["last_modified"])
                    if last_mod.tzinfo is None: last_mod = last_mod.replace(tzinfo=datetime.timezone.utc)
                    if (now_utc - last_mod).days >= older_than_days: meets_age = True
                except ValueError: self.logger.warning("invalid_date_format_for_clean", repo_id=item['repo_id'], revision=item['revision'], date_str=item["last_modified"])
            if min_size_mb is not None and (item["size_bytes"] / (1024*1024)) >= min_size_mb: meets_size = True
            should_remove = (older_than_days is not None and meets_age) or (min_size_mb is not None and meets_size)
            if older_than_days is not None and min_size_mb is not None: should_remove = meets_age and meets_size # Both must be true if both specified
            if should_remove: items_to_remove.append(item)
            else: self.logger.debug("keeping_item_criteria_not_met", repo_id=item['repo_id'], revision=item['revision'])

        if not items_to_remove:
             self.logger.info(f"{action}_no_items_match_criteria", **log_ctx); return (0, 0, [])
        self.logger.info(f"{action}_items_to_process", count=len(items_to_remove), **log_ctx)

        # Get repo info map for efficient deletion
        repo_scan_map = {repo.repo_id: repo for repo in scan_cache_dir(self.effective_paths["HF_HUB_CACHE"]).repos}

        for item in items_to_remove:
            repo_id, revision, size_human, cache_path = item["repo_id"], item["revision"], item["size_human"], item["cache_path"]
            if dry_run:
                self.logger.info("dry_run_would_remove", repo_id=repo_id, revision=revision, size=size_human, path=cache_path)
                removed_items_details.append(item); freed_bytes += item["size_bytes"]; removed_count += 1
            else:
                if os.path.exists(cache_path):
                    try:
                        target_repo_info = repo_scan_map.get(repo_id)
                        if target_repo_info:
                             self.logger.info("removing_snapshot", repo_id=repo_id, revision=revision, size=size_human, path=cache_path)
                             delete_strategy = target_repo_info.delete_revisions(revision)
                             delete_strategy.execute()
                             if not os.path.exists(cache_path):
                                  freed_bytes += item["size_bytes"]; removed_count += 1; removed_items_details.append(item)
                                  self.logger.debug("removal_successful", path=cache_path)
                             else: self.logger.warning("removal_failed_dir_still_exists_after_delete_revisions", path=cache_path)
                        else: self.logger.warning("could_not_find_repo_info_for_deletion", repo_id=repo_id)
                    except Exception as e: self.logger.exception("failed_to_remove_snapshot_hf_hub", repo_id=repo_id, revision=revision, path=cache_path, error=str(e))
                else: self.logger.warning("snapshot_path_not_found_for_removal", path=cache_path)
        self.logger.info(f"{action}_completed", items_processed=removed_count, space_affected=humanize.naturalsize(freed_bytes), **log_ctx)
        return (removed_count, freed_bytes, removed_items_details)

    def get_organization_overview(self) -> Dict[str, Any]:
        """Get overview of the structured root directory."""
        structured_root = self.effective_paths["structured_root"]
        if not os.path.exists(structured_root): return {"total_size": 0, "total_size_human": "0 B", "categories": {}}
        self.logger.info("generating_organization_overview", path=structured_root)
        overview = {"total_size": 0, "categories": {}}
        try:
            for cat_name in os.listdir(structured_root):
                cat_path = os.path.join(structured_root, cat_name)
                if cat_name.startswith('.') or not os.path.isdir(cat_path): continue
                cat_data = {"size_bytes": 0, "org_count": 0, "organizations": {}}
                for org_name in os.listdir(cat_path):
                    org_path = os.path.join(cat_path, org_name)
                    if not os.path.isdir(org_path): continue
                    org_data = {"size_bytes": 0, "repo_count": 0, "repos": []}
                    for repo_name in os.listdir(org_path):
                        repo_path = os.path.join(org_path, repo_name)
                        if not os.path.isdir(repo_path): continue
                        repo_size, sym_count, file_count = 0, 0, 0
                        try:
                            for dp, _, fns in os.walk(repo_path, followlinks=False):
                                for fn in fns:
                                    fp = os.path.join(dp, fn)
                                    if os.path.islink(fp): sym_count += 1
                                    elif os.path.isfile(fp):
                                        try: repo_size += os.path.getsize(fp); file_count += 1
                                        except OSError: self.logger.warning("could_not_get_size_overview", file=fp)
                            org_data["repos"].append({"name": repo_name, "size_bytes": repo_size, "size_human": humanize.naturalsize(repo_size), "symlink_count": sym_count, "file_count": file_count, "path": os.path.relpath(repo_path, structured_root)})
                            org_data["size_bytes"] += repo_size
                        except OSError as e: self.logger.warning("error_walking_repo_dir_overview", path=repo_path, error=str(e))
                    if org_data["repos"]:
                        org_data["repo_count"] = len(org_data["repos"]); org_data["size_human"] = humanize.naturalsize(org_data["size_bytes"])
                        org_data["repos"].sort(key=lambda x: x["size_bytes"], reverse=True)
                        cat_data["organizations"][org_name] = org_data; cat_data["size_bytes"] += org_data["size_bytes"]
                if cat_data["organizations"]:
                    cat_data["org_count"] = len(cat_data["organizations"]); cat_data["size_human"] = humanize.naturalsize(cat_data["size_bytes"])
                    overview["categories"][cat_name] = cat_data; overview["total_size"] += cat_data["size_bytes"]
            overview["total_size_human"] = humanize.naturalsize(overview["total_size"])
            self.logger.info("organization_overview_complete", total_size=overview["total_size_human"])
            return overview
        except Exception as e:
            self.logger.exception("organization_overview_failed", path=structured_root, error=str(e))
            return {"total_size": 0, "total_size_human": "0 B", "categories": {}}

