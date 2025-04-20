#!/usr/bin/env python
import os
import shutil
import json
import time
import datetime
import argparse
import logging  # Standard library logging for level constants
import re
import fnmatch # For pattern matching
from pathlib import Path
from typing import Dict, Optional, Union, List, Any, Tuple

import structlog
from tabulate import tabulate
import humanize # For user-friendly sizes

from huggingface_hub import (
    HfApi,
    snapshot_download,
    hf_hub_download,
    list_repo_files_info, # <--- Changed from list_repo_files
    scan_cache_dir,
    CacheInfo,
    CommitInfo,
    RepoFile
)
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# --- Configuration ---
DEFAULT_CONFIG_PATH = "~/.config/hf_organizer/config.json"
DEFAULT_STRUCTURED_ROOT = "~/huggingface_organized"
DEFAULT_HF_HOME = "~/.cache/huggingface"

class HfHubOrganizer:
    """
    Manages HuggingFace Hub downloads, cache, and organization with profiles.
    Uses huggingface_hub library functions where possible.
    """

    # Default environment variable values
    ENV_VARS = {
        "HF_HOME": DEFAULT_HF_HOME,
        "HF_HUB_CACHE": "${HF_HOME}/hub",
        "HF_ASSETS_CACHE": "${HF_HOME}/assets",
        "HF_TOKEN": None,
        "HF_HUB_VERBOSITY": "warning",
        "HF_HUB_ETAG_TIMEOUT": "10",
        "HF_HUB_DOWNLOAD_TIMEOUT": "10"
        # Note: HF_XET_CACHE is not standard in huggingface_hub, removed for clarity
    }

    BOOLEAN_ENV_VARS = {
        "HF_DEBUG": False,
        "HF_HUB_OFFLINE": False,
        "HF_HUB_DISABLE_PROGRESS_BARS": False,
        "HF_HUB_DISABLE_TELEMETRY": True,  # Default to privacy-friendly
        "HF_HUB_ENABLE_HF_TRANSFER": True   # <--- Default to True for speed
    }

    def __init__(
        self,
        profile: Optional[str] = None,
        base_path: Optional[str] = None,
        structured_root: Optional[str] = None,
        token: Optional[str] = None,
        enable_hf_transfer: Optional[bool] = None, # Allow override via constructor/CLI
        verbose: bool = False,
        config_path: Optional[str] = None,
        log_format: str = "console"
    ):
        """Initialize with custom paths and settings."""
        self.config_path = os.path.expanduser(config_path or DEFAULT_CONFIG_PATH)
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

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
            self.logger.info("using_profile", profile=profile)

        # Determine effective settings (CLI/Constructor > Profile > Environment > Default)
        effective_base_path = base_path or profile_settings.get("base_path") or os.environ.get("HF_HOME")
        effective_structured_root = structured_root or profile_settings.get("structured_root") or DEFAULT_STRUCTURED_ROOT
        effective_token = token or profile_settings.get("token") or os.environ.get("HF_TOKEN")
        # Handle hf_transfer enable flag priority
        if enable_hf_transfer is None: # Not set via CLI/constructor
             # Use profile setting if available, otherwise default (True)
             effective_enable_hf_transfer = profile_settings.get("enable_hf_transfer", self.BOOLEAN_ENV_VARS["HF_HUB_ENABLE_HF_TRANSFER"])
        else: # Explicitly set via CLI/constructor
             effective_enable_hf_transfer = enable_hf_transfer


        # Set base HF path if needed
        if effective_base_path:
            os.environ["HF_HOME"] = os.path.expanduser(effective_base_path)
        elif "HF_HOME" not in os.environ: # Ensure default is set if nothing overrides
             os.environ["HF_HOME"] = os.path.expanduser(self.ENV_VARS["HF_HOME"])


        self.structured_root = os.path.expanduser(effective_structured_root)
        os.makedirs(self.structured_root, exist_ok=True)

        # Set token if needed (and not already in env)
        if effective_token and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = effective_token

        # Initialize all other environment variables
        self._initialize_env_vars(force_hf_transfer_setting=effective_enable_hf_transfer)

        # Keep track of effective paths
        # Recalculate HF_HUB_CACHE based on final HF_HOME
        hf_home_final = os.environ["HF_HOME"]
        hf_hub_cache_default = os.path.join(hf_home_final, "hub")
        hf_hub_cache_final = os.environ.get("HF_HUB_CACHE", hf_hub_cache_default)
        # Ensure HF_HUB_CACHE is explicitly set if it was derived
        if "HF_HUB_CACHE" not in os.environ or not os.environ["HF_HUB_CACHE"]:
             os.environ["HF_HUB_CACHE"] = hf_hub_cache_final

        self.effective_paths = {
            "HF_HOME": hf_home_final,
            "HF_HUB_CACHE": hf_hub_cache_final,
            "structured_root": self.structured_root
        }

        self.logger = self.logger.bind(
             profile=profile or "Default",
             hf_home=self.effective_paths["HF_HOME"],
             cache=self.effective_paths["HF_HUB_CACHE"],
             org_root=self.structured_root,
             hf_transfer=os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"
        )
        self.logger.info("initialized")

        # Initialize HF API (will use HF_TOKEN from env if set)
        self.api = HfApi(token=os.environ.get("HF_TOKEN")) # Explicitly pass None if not set

    def _setup_logger(self, verbose: bool, format_type: str) -> structlog.BoundLogger:
        """Set up structured logging."""
        log_level = logging.DEBUG if verbose else logging.INFO

        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False), # Use local time
        ]

        if format_type == "json":
            processors.extend([
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(sort_keys=True) # Sort keys for consistency
            ])
        elif format_type == "structured":
            processors.extend([
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(colors=False) # Simple structured, no colors
            ])
        else:  # console (default)
            processors.extend([
                structlog.dev.set_exc_info,
                structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback) # Pretty console
            ])

        # Filter logs based on level
        # Use standard logging handler to respect level
        logging.basicConfig(
            format="%(message)s",
            level=log_level,
        )

        structlog.configure(
            processors=processors,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger(organizer=self.__class__.__name__)

    def _load_config(self) -> Dict[str, Any]:
        """Load config from disk or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.logger.debug("config_loaded", path=self.config_path)
                    # Basic validation
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

        # Return default if load failed or file didn't exist
        return {"profiles": {}}


    def _save_config(self):
        """Save current config to disk."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2, sort_keys=True)
            self.logger.debug("config_saved", path=self.config_path)
        except Exception as e:
            self.logger.error("config_save_failed", path=self.config_path, error=str(e))

    def list_profiles(self) -> List[str]:
        """List all available profiles."""
        return list(self.config.get("profiles", {}).keys())

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

        # Ensure paths are stored expanded but handle potential None
        profile_data = {
            "base_path": os.path.expanduser(base_path) if base_path else None,
            "structured_root": os.path.expanduser(structured_root) if structured_root else None,
            "token": token, # Keep token as is (might be None)
            "enable_hf_transfer": enable_hf_transfer, # Store the boolean or None
            "description": description or f"Profile created for {name}"
        }

        # Remove None values so they don't override defaults unnecessarily when loading
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
        else:
            self.logger.warning("profile_not_found_for_removal", name=name)

    def _initialize_env_vars(self, force_hf_transfer_setting: Optional[bool] = None):
        """Initialize environment variables with defaults if not already set."""
        # Process string environment variables
        for key, default_value in self.ENV_VARS.items():
            if key not in os.environ and default_value is not None:
                # Handle variable expansion like ${HF_HOME}
                # Use current os.environ for expansion lookup
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
        repo_type_guess = "model" # Default guess
        namespace = "library" # Default namespace
        repo_name = repo_id

        if "/" in repo_id:
            namespace, repo_name = repo_id.split("/", 1)

        if category is None:
            try:
                # Use api.repo_info which returns repo_type correctly
                repo_info = self.api.repo_info(repo_id=repo_id)
                repo_type_from_api = repo_info.repo_type
                if repo_type_from_api == "dataset":
                    repo_type_guess = "datasets"
                elif repo_type_from_api == "space":
                    repo_type_guess = "spaces"
                else:
                    repo_type_guess = "models" # Explicitly models
                self.logger.debug("category_detected_via_api", repo_id=repo_id, category=repo_type_guess)
            except RepositoryNotFoundError:
                 self.logger.error("repo_not_found_api", repo_id=repo_id)
                 raise # Re-raise the specific error
            except HfHubHTTPError as http_err:
                 # Handle potential auth errors more specifically
                 if http_err.response.status_code == 401:
                      self.logger.error("authentication_error_api", repo_id=repo_id, error=str(http_err))
                      raise ValueError(f"Authentication failed for {repo_id}. Check your HF_TOKEN.") from http_err
                 else:
                      self.logger.warning("category_detection_http_error", repo_id=repo_id, status=http_err.response.status_code, error=str(http_err), fallback=repo_type_guess)
            except Exception as e:
                self.logger.warning("category_detection_failed_api", repo_id=repo_id, error=str(e), error_type=type(e).__name__, fallback=repo_type_guess)
        else:
            # User provided category overrides detection
            repo_type_guess = category

        # Create organized base path for the repo
        org_repo_path = os.path.join(
            self.structured_root,
            repo_type_guess,
            namespace,
            repo_name
        )

        # Full path including potential subfolder for downloads
        org_download_path = os.path.join(org_repo_path, subfolder) if subfolder else org_repo_path

        os.makedirs(org_download_path, exist_ok=True)
        self.logger.debug("organizing_files_target", path=org_download_path)

        return org_repo_path, org_download_path, repo_type_guess, namespace

    def _link_or_copy(self, cache_path: str, org_path: str, symlink_to_cache: bool):
        """Helper to symlink or copy a file/directory."""
        # Ensure parent directory exists
        org_dir = os.path.dirname(org_path)
        if not os.path.exists(org_dir):
             os.makedirs(org_dir, exist_ok=True)
             self.logger.debug("created_parent_dir", path=org_dir)


        # Handle existing target path
        if os.path.lexists(org_path): # Use lexists to check for broken symlinks too
            is_link = os.path.islink(org_path)
            is_dir = os.path.isdir(org_path) and not is_link # Real directory
            is_file = os.path.isfile(org_path) and not is_link # Real file

            # Decide if removal is needed
            remove_existing = True
            if is_link:
                 try:
                      link_target = os.readlink(org_path)
                      # Don't remove if it's already linked to the correct cache path
                      if link_target == os.path.abspath(cache_path):
                           self.logger.debug("target_already_correct_symlink", path=org_path)
                           remove_existing = False
                 except OSError: # Broken link
                      pass # Remove broken link
            elif is_dir and not symlink_to_cache:
                 # If copying a directory, remove existing dir first
                 pass
            elif is_file and not symlink_to_cache:
                 # If copying a file, remove existing file first
                 pass
            elif is_dir and symlink_to_cache:
                 # Trying to symlink where a directory exists - remove dir
                 pass
            elif is_file and symlink_to_cache:
                 # Trying to symlink where a file exists - remove file
                 pass
            else: # Other cases?
                 self.logger.warning("unhandled_existing_target_state", path=org_path, is_link=is_link, is_dir=is_dir, is_file=is_file)
                 remove_existing = False # Be safe

            if remove_existing:
                try:
                    if is_link or is_file:
                        os.remove(org_path)
                        self.logger.debug("removed_existing_target_link_or_file", path=org_path)
                    elif is_dir:
                        shutil.rmtree(org_path)
                        self.logger.debug("removed_existing_target_dir", path=org_path)
                except OSError as e:
                    self.logger.error("failed_removing_existing_target", path=org_path, error=str(e))
                    raise # Stop if we can't prepare the target location


        # Create link or copy
        if symlink_to_cache:
            try:
                # Ensure cache_path is absolute for reliable symlinking
                abs_cache_path = os.path.abspath(cache_path)
                os.symlink(abs_cache_path, org_path)
                self.logger.debug("symlink_created", source=abs_cache_path, target=org_path)
            except OSError as e:
                 self.logger.error("symlink_failed", source=cache_path, target=org_path, error=str(e))
                 raise # Let the error propagate
        else:
            try:
                if os.path.isdir(cache_path):
                    shutil.copytree(cache_path, org_path, symlinks=True) # Preserve symlinks within copied structure if any
                    self.logger.debug("directory_copied", source=cache_path, target=org_path)
                elif os.path.isfile(cache_path):
                    shutil.copy2(cache_path, org_path) # copy2 preserves metadata
                    self.logger.debug("file_copied", source=cache_path, target=org_path)
                else:
                    self.logger.warning("copy_source_not_found_or_not_file_or_dir", source=cache_path)

            except Exception as e:
                self.logger.error("copy_failed", source=cache_path, target=org_path, error=str(e))
                raise # Let the error propagate

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
        **kwargs # Pass extra args like force_download
    ) -> str:
        """
        Download a full repository or a specific file and organize it.
        Uses snapshot_download or hf_hub_download from huggingface_hub.
        """
        start_time = time.time()
        log_ctx = {"repo_id": repo_id, "filename": filename or "entire_repo", "subfolder": subfolder, "revision": revision or "main"}
        self.logger.info("download_started", **log_ctx)

        try:
            org_repo_path, org_download_path, detected_category, _ = self._determine_category_and_paths(
                repo_id, category, subfolder
            )
            log_ctx["category"] = detected_category

            # Save metadata about this download attempt (even if it fails later)
            self._save_download_metadata(org_repo_path, repo_id, detected_category, filename or "entire_repo", subfolder, revision)

            downloaded_path_in_cache: str
            final_organized_path: str

            if filename:
                # --- Download a specific file ---
                self.logger.debug("downloading_single_file", **log_ctx)
                downloaded_path_in_cache = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder,
                    revision=revision,
                    repo_type=detected_category if detected_category in ["models", "datasets", "spaces"] else None,
                    token=self.api.token, # Pass token
                    **kwargs
                )
                # Target path in the organized structure
                final_organized_path = os.path.join(org_download_path, os.path.basename(filename))
                self._link_or_copy(downloaded_path_in_cache, final_organized_path, symlink_to_cache)

            else:
                # --- Download entire repo (or subfolder) ---
                self.logger.debug("downloading_snapshot", allow_patterns=allow_patterns, ignore_patterns=ignore_patterns, **log_ctx)
                downloaded_path_in_cache = snapshot_download(
                    repo_id=repo_id,
                    subfolder=subfolder,
                    revision=revision,
                    repo_type=detected_category if detected_category in ["models", "datasets", "spaces"] else None,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    token=self.api.token, # Pass token
                    **kwargs
                )
                # Target path is the directory itself
                final_organized_path = org_download_path

                # Ensure target directory exists and is empty/correctly linked
                if os.path.lexists(final_organized_path):
                    if os.path.islink(final_organized_path):
                        # If it's a link, remove it to copy/link contents
                        os.remove(final_organized_path)
                    elif os.path.isdir(final_organized_path):
                         # If it's a directory, remove its contents before populating
                         for item in os.listdir(final_organized_path):
                              item_path = os.path.join(final_organized_path, item)
                              if os.path.islink(item_path) or os.path.isfile(item_path):
                                   os.remove(item_path)
                              elif os.path.isdir(item_path):
                                   shutil.rmtree(item_path)
                         self.logger.debug("cleared_existing_target_dir_contents", path=final_organized_path)
                    else: # It's a file? Remove it.
                         os.remove(final_organized_path)
                else:
                     # If it doesn't exist, create it
                     os.makedirs(final_organized_path, exist_ok=True)


                # Link or copy contents item by item
                item_count = 0
                items_in_cache = os.listdir(downloaded_path_in_cache)
                for item_name in items_in_cache:
                    cache_item_path = os.path.join(downloaded_path_in_cache, item_name)
                    org_item_path = os.path.join(final_organized_path, item_name)
                    self._link_or_copy(cache_item_path, org_item_path, symlink_to_cache)
                    item_count += 1
                self.logger.debug("snapshot_contents_processed", count=item_count, source=downloaded_path_in_cache, target=final_organized_path, items=len(items_in_cache))


            elapsed = time.time() - start_time
            self.logger.info("download_completed",
                            organized_path=final_organized_path,
                            elapsed_seconds=round(elapsed, 2),
                            symlinked=symlink_to_cache,
                            **log_ctx)

            return final_organized_path

        except RepositoryNotFoundError:
             self.logger.error("download_failed_repo_not_found", **log_ctx)
             raise # Re-raise for CLI handling
        except HfHubHTTPError as http_err:
             self.logger.error("download_failed_http_error", status=http_err.response.status_code, error=str(http_err), **log_ctx)
             raise
        except Exception as e:
            self.logger.exception("download_failed_unexpected", error=str(e), **log_ctx) # Use exception for traceback
            raise # Re-raise for CLI handling

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
        **kwargs # Pass extra args like force_download
    ) -> str:
        """
        Download only files modified within the last N days for a repository.
        Uses list_repo_files_info and hf_hub_download.
        """
        start_time = time.time()
        log_ctx = {"repo_id": repo_id, "days_ago": days_ago, "subfolder": subfolder, "revision": revision or "main"}
        self.logger.info("download_recent_started", **log_ctx)

        try:
            # Determine target paths and category
            org_repo_path, org_download_path, detected_category, _ = self._determine_category_and_paths(
                repo_id, category, subfolder
            )
            log_ctx["category"] = detected_category

            # Calculate the cutoff date (make it timezone-aware UTC)
            cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_ago)
            self.logger.debug("filtering_commits_since", cutoff_date=cutoff_date.isoformat(), **log_ctx)

            # Get file info including commit details
            # Use token explicitly if needed, otherwise API client uses env var
            all_files_info: List[RepoFile] = list_repo_files_info( # <--- FIXED: Use list_repo_files_info
                repo_id=repo_id,
                revision=revision,
                token=self.api.token # Pass token from initialized API client
            )
            self.logger.debug("retrieved_repo_file_info", count=len(all_files_info), **log_ctx)


            # Filter files based on last commit date AND patterns
            recent_files_to_download: List[RepoFile] = []
            for file_info in all_files_info:
                # 1. Check commit date
                is_recent = False
                if file_info.last_commit and file_info.last_commit.date and file_info.last_commit.date > cutoff_date:
                    is_recent = True

                if not is_recent:
                    continue # Skip if not recent

                # 2. Check subfolder
                in_subfolder = True # Assume yes if no subfolder specified
                if subfolder:
                    norm_subfolder = subfolder.strip('/')
                    # Check if path starts with subfolder/ or is exactly subfolder
                    if not (file_info.path == norm_subfolder or file_info.path.startswith(norm_subfolder + '/')):
                         in_subfolder = False

                if not in_subfolder:
                     continue # Skip if not in requested subfolder


                # 3. Check allow/ignore patterns
                path_matches = True # Assume match unless excluded
                if allow_patterns:
                     path_matches = any(fnmatch.fnmatch(file_info.path, pattern) for pattern in allow_patterns)
                if path_matches and ignore_patterns: # Only check ignore if it wasn't excluded by allow
                     if any(fnmatch.fnmatch(file_info.path, pattern) for pattern in ignore_patterns):
                          path_matches = False

                if not path_matches:
                     self.logger.debug("file_skipped_by_pattern", file=file_info.path, **log_ctx)
                     continue # Skip if excluded by patterns

                # If all checks pass, add to list
                recent_files_to_download.append(file_info)
                self.logger.debug("file_marked_for_recent_download", file=file_info.path, commit_date=file_info.last_commit.date, **log_ctx)


            if not recent_files_to_download:
                self.logger.info("no_recent_files_found_matching_criteria", **log_ctx)
                # Still save metadata indicating an attempt was made
                self._save_download_metadata(org_repo_path, repo_id, detected_category, f"recent_{days_ago}d", subfolder, revision)
                return org_download_path # Return the base path even if nothing downloaded

            self.logger.info("downloading_recent_files", count=len(recent_files_to_download), **log_ctx)

            # Download each recent file individually
            downloaded_count = 0
            failed_count = 0
            for file_info in recent_files_to_download:
                try:
                    # Determine the correct subfolder relative to the repo root for hf_hub_download
                    file_repo_subfolder = os.path.dirname(file_info.path)
                    file_basename = os.path.basename(file_info.path)

                    self.logger.debug("downloading_individual_recent_file", file=file_info.path, **log_ctx)
                    downloaded_path_in_cache = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_basename,
                        subfolder=file_repo_subfolder if file_repo_subfolder else None,
                        revision=revision, # Use specific revision if needed
                        repo_type=detected_category if detected_category in ["models", "datasets", "spaces"] else None,
                        token=self.api.token, # Pass token
                        **kwargs
                    )

                    # Determine the final organized path for this specific file
                    relative_file_path = file_info.path # Path relative to repo root
                    if subfolder:
                         # Adjust relative path if user requested a subfolder download
                         norm_subfolder = subfolder.strip('/')
                         if file_info.path.startswith(norm_subfolder + '/'):
                             relative_file_path = os.path.relpath(file_info.path, norm_subfolder)
                         elif file_info.path == norm_subfolder:
                              relative_file_path = os.path.basename(file_info.path)

                    final_organized_path = os.path.join(org_download_path, relative_file_path)

                    # Link or copy the downloaded file
                    self._link_or_copy(downloaded_path_in_cache, final_organized_path, symlink_to_cache)
                    downloaded_count += 1

                except Exception as e_file:
                    failed_count += 1
                    self.logger.error("failed_downloading_recent_file", file=file_info.path, error=str(e_file), **log_ctx)
                    # Continue with other files

            elapsed = time.time() - start_time
            self.logger.info("download_recent_completed",
                            files_downloaded=downloaded_count,
                            files_failed=failed_count,
                            target_path=org_download_path,
                            elapsed_seconds=round(elapsed, 2),
                            symlinked=symlink_to_cache,
                            **log_ctx)

            # Save metadata indicating a successful recent download
            self._save_download_metadata(org_repo_path, repo_id, detected_category, f"recent_{days_ago}d", subfolder, revision)

            return org_download_path # Return the base path where files were placed

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
        """Save metadata about downloaded repo/file for future reference."""
        metadata_dir = os.path.join(self.structured_root, ".metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_file = os.path.join(metadata_dir, "downloads.json")

        metadata = {"downloads": []}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, dict) and isinstance(loaded_data.get("downloads"), list):
                         metadata = loaded_data
                    else:
                         self.logger.warning("invalid_metadata_format_resetting", path=metadata_file)
            except json.JSONDecodeError:
                 self.logger.warning("invalid_metadata_file_resetting", path=metadata_file)
            except Exception as e:
                 self.logger.error("failed_loading_metadata", path=metadata_file, error=str(e))

        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "repo_id": repo_id,
            "category": category,
            "type": download_type, # e.g., "entire_repo", "filename.txt", "recent_7d"
            "relative_path": os.path.relpath(path, self.structured_root),
            "profile": self.selected_profile,
            "revision": revision or "main",
            "subfolder": subfolder,
        }
        # Remove None values for cleaner output
        entry = {k: v for k, v in entry.items() if v is not None}

        metadata["downloads"].insert(0, entry)

        # Optional: Limit history size
        max_history = 500
        if len(metadata["downloads"]) > max_history:
             metadata["downloads"] = metadata["downloads"][:max_history]

        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
        except Exception as e:
             self.logger.error("failed_saving_metadata", path=metadata_file, error=str(e))


    def list_downloads(self, limit: Optional[int] = None, category: Optional[str] = None, profile_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List download history recorded by this tool."""
        metadata_file = os.path.join(self.structured_root, ".metadata", "downloads.json")

        if not os.path.exists(metadata_file):
            self.logger.warning("no_download_history_found", path=metadata_file)
            return []

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
             self.logger.error("failed_loading_metadata_for_list", path=metadata_file, error=str(e))
             return []

        downloads = metadata.get("downloads", [])
        if not isinstance(downloads, list):
             self.logger.error("invalid_metadata_structure_downloads_not_list", path=metadata_file)
             return []


        # Apply filters
        filtered_downloads = downloads
        if category:
            filtered_downloads = [d for d in filtered_downloads if isinstance(d, dict) and d.get("category") == category]
        if profile_filter:
             filtered_downloads = [d for d in filtered_downloads if isinstance(d, dict) and d.get("profile") == profile_filter]

        # Apply limit
        if limit and limit > 0:
            filtered_downloads = filtered_downloads[:limit]

        return filtered_downloads

    def scan_cache(self) -> List[Dict[str, Any]]:
        """
        Scan the HF cache directory using huggingface_hub.scan_cache_dir.
        """
        cache_dir = self.effective_paths["HF_HUB_CACHE"]
        self.logger.info("scanning_cache_with_hf_hub", path=cache_dir)

        try:
            scan = scan_cache_dir(cache_dir)
            self.logger.info("cache_scan_complete", repos=scan.repos_count, size=scan.size_on_disk_str)

            results = []
            for repo_info in scan.repos:
                 for revision_info in repo_info.revisions:
                      # Find largest file in this specific snapshot
                      largest_file = {"name": "", "size": 0}
                      snapshot_path = Path(revision_info.snapshot_path)
                      try:
                           files_in_snapshot = [p for p in snapshot_path.rglob('*') if p.is_file()]
                           for file_path in files_in_snapshot:
                                try:
                                     file_size = file_path.stat().st_size
                                     if file_size > largest_file["size"]:
                                          largest_file = {"name": str(file_path.relative_to(snapshot_path)), "size": file_size}
                                except OSError:
                                     self.logger.warning("getsize_failed_cache_scan", path=str(file_path))
                      except OSError as e:
                           self.logger.warning("failed_listing_snapshot_files", path=str(snapshot_path), error=str(e))


                      results.append({
                          "repo_id": repo_info.repo_id,
                          "repo_type": repo_info.repo_type,
                          "revision": revision_info.commit_hash,
                          "size_bytes": revision_info.size_on_disk,
                          "size_human": revision_info.size_on_disk_str,
                          "file_count": len(revision_info.files), # Count files directly associated with revision
                          "largest_file": largest_file,
                          "last_modified": revision_info.last_modified.isoformat() if revision_info.last_modified else None,
                          "cache_path": str(revision_info.snapshot_path) # Store path for cleaning
                      })

            # Sort by size (largest first)
            results.sort(key=lambda x: x["size_bytes"], reverse=True)
            return results

        except Exception as e:
            self.logger.exception("cache_scan_failed_hf_hub", path=cache_dir, error=str(e))
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache usage based on scan_cache results."""
        cache_items = self.scan_cache() # Uses the new scan_cache

        if not cache_items:
            return {
                "total_size": 0,
                "total_size_human": "0 B",
                "repo_count": 0,
                "snapshot_count": 0,
                "file_count": 0,
                "largest_snapshots": [],
                "organizations": []
            }

        total_size = sum(item["size_bytes"] for item in cache_items)
        total_files = sum(item["file_count"] for item in cache_items) # Uses file_count from scan
        unique_repos = {item["repo_id"] for item in cache_items}

        # Get top 5 largest snapshots (revisions)
        largest_snapshots = sorted(cache_items, key=lambda x: x["size_bytes"], reverse=True)[:5]

        # Group by organization
        orgs = {}
        for item in cache_items:
            repo_id = item["repo_id"]
            org = repo_id.split("/")[0] if "/" in repo_id else "library"

            if org not in orgs:
                orgs[org] = {"size_bytes": 0, "snapshot_count": 0}

            orgs[org]["size_bytes"] += item["size_bytes"]
            orgs[org]["snapshot_count"] += 1

        formatted_orgs = []
        for org, stats in orgs.items():
            percentage = (stats["size_bytes"] / total_size) * 100 if total_size > 0 else 0
            formatted_orgs.append({
                "name": org,
                "size_bytes": stats["size_bytes"],
                "size_human": humanize.naturalsize(stats["size_bytes"]),
                "snapshot_count": stats["snapshot_count"],
                "percentage": percentage
            })

        top_orgs = sorted(formatted_orgs, key=lambda x: x["size_bytes"], reverse=True)

        return {
            "total_size": total_size,
            "total_size_human": humanize.naturalsize(total_size),
            "repo_count": len(unique_repos),
            "snapshot_count": len(cache_items),
            "file_count": total_files,
            "largest_snapshots": largest_snapshots,
            "organizations": top_orgs
        }

    def clean_cache(self, older_than_days: Optional[int] = None, min_size_mb: Optional[int] = None, dry_run: bool = False) -> Tuple[int, int, List[Dict]]:
        """
        Clean up cached snapshots (revisions) based on age or size criteria.
        Uses scan_cache() which relies on huggingface_hub.scan_cache_dir.
        """
        cache_items = self.scan_cache() # Get detailed list including paths from scan_cache
        action = "dry_run" if dry_run else "clean_cache"
        log_ctx = {"older_than_days": older_than_days, "min_size_mb": min_size_mb, "dry_run": dry_run}
        self.logger.info(f"{action}_started", **log_ctx)


        if not cache_items:
            self.logger.info("cache_empty", **log_ctx)
            return (0, 0, [])

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        freed_bytes = 0
        removed_count = 0
        removed_items_details = []
        items_to_remove = []

        # --- Filtering Logic ---
        for item in cache_items:
            meets_age_criteria = False
            if older_than_days is not None:
                if item["last_modified"]:
                    try:
                        last_mod_dt = datetime.datetime.fromisoformat(item["last_modified"])
                        # Ensure it's timezone-aware (scan_cache_dir returns aware)
                        if last_mod_dt.tzinfo is None:
                             # Should not happen with scan_cache_dir, but handle defensively
                             last_mod_dt = last_mod_dt.replace(tzinfo=datetime.timezone.utc)
                        if (now_utc - last_mod_dt).days >= older_than_days:
                            meets_age_criteria = True
                    except ValueError:
                        self.logger.warning("invalid_date_format_for_clean", repo_id=item['repo_id'], revision=item['revision'], date_str=item["last_modified"])

            meets_size_criteria = False
            if min_size_mb is not None:
                size_mb = item["size_bytes"] / (1024 * 1024)
                if size_mb >= min_size_mb:
                    meets_size_criteria = True

            should_remove = False
            if older_than_days is not None and min_size_mb is not None:
                should_remove = meets_age_criteria and meets_size_criteria
            elif older_than_days is not None:
                should_remove = meets_age_criteria
            elif min_size_mb is not None:
                should_remove = meets_size_criteria

            if should_remove:
                items_to_remove.append(item)
            else:
                 self.logger.debug("keeping_item_criteria_not_met", repo_id=item['repo_id'], revision=item['revision'])


        # --- Execution Phase ---
        if not items_to_remove:
             self.logger.info(f"{action}_no_items_match_criteria", **log_ctx)
             return (0, 0, [])

        self.logger.info(f"{action}_items_to_process", count=len(items_to_remove), **log_ctx)

        for item in items_to_remove:
            repo_id = item["repo_id"]
            revision = item["revision"]
            size_human = item["size_human"]
            cache_path = item["cache_path"] # Path to the snapshot directory

            if dry_run:
                self.logger.info("dry_run_would_remove", repo_id=repo_id, revision=revision, size=size_human, path=cache_path)
                removed_items_details.append(item)
                freed_bytes += item["size_bytes"]
                removed_count += 1
            else:
                if os.path.exists(cache_path):
                    try:
                        self.logger.info("removing_snapshot", repo_id=repo_id, revision=revision, size=size_human, path=cache_path)
                        # scan_cache_dir provides CacheInfo which has a delete_revisions method
                        # However, we scanned *all* repos. We need to delete specific revisions.
                        # Re-scan just the target repo to get its CacheRepoInfo object
                        repo_scan = scan_cache_dir(self.effective_paths["HF_HUB_CACHE"]).repos
                        target_repo_info = next((r for r in repo_scan if r.repo_id == repo_id), None)

                        if target_repo_info:
                             delete_strategy = target_repo_info.delete_revisions(revision)
                             delete_strategy.execute()
                             if not os.path.exists(cache_path): # Verify deletion
                                  freed_bytes += item["size_bytes"]
                                  removed_count += 1
                                  removed_items_details.append(item)
                                  self.logger.debug("removal_successful", path=cache_path)
                             else:
                                  # This might happen if other processes hold files
                                  self.logger.warning("removal_failed_dir_still_exists_after_delete_revisions", path=cache_path)
                        else:
                             self.logger.warning("could_not_find_repo_info_for_deletion", repo_id=repo_id)
                             # Fallback to shutil.rmtree if repo info not found? Or just log?
                             # Let's just log for now, as using the official delete is safer.

                    except Exception as e:
                        self.logger.exception("failed_to_remove_snapshot_hf_hub", repo_id=repo_id, revision=revision, path=cache_path, error=str(e))
                else:
                    self.logger.warning("snapshot_path_not_found_for_removal", path=cache_path)

        self.logger.info(f"{action}_completed",
                        items_processed=removed_count,
                        space_affected=humanize.naturalsize(freed_bytes),
                        **log_ctx)

        return (removed_count, freed_bytes, removed_items_details)


    def get_organization_overview(self) -> Dict[str, Any]:
        """Get an overview of how files are organized in the structured root."""
        structured_root = self.effective_paths["structured_root"]
        if not os.path.exists(structured_root):
            self.logger.warning("structured_root_not_found", path=structured_root)
            return {"total_size": 0, "total_size_human": "0 B", "categories": {}}

        self.logger.info("generating_organization_overview", path=structured_root)
        overview_data = {"total_size": 0, "categories": {}}

        try:
            for category_name in os.listdir(structured_root):
                category_path = os.path.join(structured_root, category_name)
                if category_name.startswith('.') or not os.path.isdir(category_path): continue

                category_data = {"size_bytes": 0, "org_count": 0, "organizations": {}}
                for org_name in os.listdir(category_path):
                    org_path = os.path.join(category_path, org_name)
                    if not os.path.isdir(org_path): continue

                    org_data = {"size_bytes": 0, "repo_count": 0, "repos": []}
                    for repo_name in os.listdir(org_path):
                        repo_path = os.path.join(org_path, repo_name)
                        if not os.path.isdir(repo_path): continue

                        repo_size = 0
                        symlink_count = 0
                        file_count = 0
                        try:
                            for dirpath, _, filenames in os.walk(repo_path, followlinks=False): # Don't follow links for size calc
                                for filename in filenames:
                                    filepath = os.path.join(dirpath, filename)
                                    if os.path.islink(filepath):
                                        symlink_count += 1
                                    elif os.path.isfile(filepath):
                                        try:
                                             repo_size += os.path.getsize(filepath)
                                             file_count += 1
                                        except OSError:
                                             self.logger.warning("could_not_get_size_overview", file=filepath)
                            org_data["repos"].append({
                                "name": repo_name,
                                "size_bytes": repo_size,
                                "size_human": humanize.naturalsize(repo_size),
                                "symlink_count": symlink_count,
                                "file_count": file_count,
                                "path": os.path.relpath(repo_path, structured_root)
                            })
                            org_data["size_bytes"] += repo_size
                        except OSError as walk_err:
                            self.logger.warning("error_walking_repo_dir_overview", path=repo_path, error=str(walk_err))

                    if org_data["repos"]:
                        org_data["repo_count"] = len(org_data["repos"])
                        org_data["size_human"] = humanize.naturalsize(org_data["size_bytes"])
                        org_data["repos"].sort(key=lambda x: x["size_bytes"], reverse=True)
                        category_data["organizations"][org_name] = org_data
                        category_data["size_bytes"] += org_data["size_bytes"]

                if category_data["organizations"]:
                    category_data["org_count"] = len(category_data["organizations"])
                    category_data["size_human"] = humanize.naturalsize(category_data["size_bytes"])
                    overview_data["categories"][category_name] = category_data
                    overview_data["total_size"] += category_data["size_bytes"]

            overview_data["total_size_human"] = humanize.naturalsize(overview_data["total_size"])
            self.logger.info("organization_overview_complete", total_size=overview_data["total_size_human"])
            return overview_data

        except Exception as e:
            self.logger.exception("organization_overview_failed", path=structured_root, error=str(e))
            return {"total_size": 0, "total_size_human": "0 B", "categories": {}}


# =============================================================================
# Command Line Interface Section
# =============================================================================

def _create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="HfHubOrganizer: Manage HuggingFace Hub downloads, cache, and organization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--profile", help="Profile name to use (defined in config). Overrides default behavior.")
    parser.add_argument("--log-format", choices=["console", "json", "structured"],
                      default="console", help="Logging output format.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (DEBUG level) logging.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to the configuration file.")
    parser.add_argument("--no-hf-transfer", action="store_true",
                        help="Disable hf_transfer for downloads (if enabled by default/profile).")


    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # --- Download Command ---
    download_parser = subparsers.add_parser("download", help="Download a full repository or specific file.")
    download_parser.add_argument("repo_id", help="Repository ID (e.g., 'google/flan-t5-base', 'gpt2')")
    download_parser.add_argument("--filename", "-f", help="Specific file to download within the repo.")
    download_parser.add_argument("--subfolder", "-s", help="Subfolder within the repository to download from/into.")
    download_parser.add_argument("--revision", "-r", help="Git revision (branch, tag, commit hash) to download.")
    download_parser.add_argument("--category", choices=["models", "datasets", "spaces"],
                        help="Manually specify category for organization (overrides auto-detection).")
    download_parser.add_argument("--base-path", help="Override HF_HOME (cache location) for this command.")
    download_parser.add_argument("--out-dir", help="Override structured organization root directory for this command.")
    download_parser.add_argument("--copy", action="store_true", help="Copy files from cache instead of symlinking.")
    download_parser.add_argument("--allow-patterns", nargs='*', help="Glob patterns to include (snapshot only).")
    download_parser.add_argument("--ignore-patterns", nargs='*', help="Glob patterns to exclude (snapshot only).")
    download_parser.add_argument("--force-download", action="store_true", help="Force re-download even if file exists in cache.")


    # --- Download Recent Command ---
    download_recent_parser = subparsers.add_parser("download-recent", help="Download only files modified recently, optionally matching patterns.")
    download_recent_parser.add_argument("repo_id", help="Repository ID (e.g., 'google/flan-t5-base')")
    download_recent_parser.add_argument("--days", "-d", type=int, required=True, help="Download files modified within the last N days.")
    download_recent_parser.add_argument("--subfolder", "-s", help="Only consider files within this subfolder.")
    download_recent_parser.add_argument("--revision", "-r", help="Git revision (branch, tag, commit hash) to check.")
    download_recent_parser.add_argument("--category", choices=["models", "datasets", "spaces"],
                        help="Manually specify category for organization.")
    download_recent_parser.add_argument("--base-path", help="Override HF_HOME (cache location).")
    download_recent_parser.add_argument("--out-dir", help="Override structured organization root directory.")
    download_recent_parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    download_recent_parser.add_argument("--allow-patterns", nargs='*', help="Glob patterns to include files.")
    download_recent_parser.add_argument("--ignore-patterns", nargs='*', help="Glob patterns to exclude files.")
    download_recent_parser.add_argument("--exclude-repo-pattern", help="Skip download if this case-insensitive text is found in the repo_id.")
    download_recent_parser.add_argument("--force-download", action="store_true", help="Force re-download even if file exists in cache.")


    # --- Profile Management Command ---
    profile_parser = subparsers.add_parser("profile", help="Manage configuration profiles.")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command", required=True, help="Profile action")
    profile_subparsers.add_parser("list", help="List available profiles.")
    add_parser = profile_subparsers.add_parser("add", help="Add or update a profile.")
    add_parser.add_argument("name", help="Profile name.")
    add_parser.add_argument("--base-path", help="Base path for HF cache (HF_HOME). Use '~' for home dir.")
    add_parser.add_argument("--out-dir", help="Directory for organized files (structured_root). Use '~' for home dir.")
    add_parser.add_argument("--token", help="HuggingFace API token (optional, stored encrypted/securely if possible - currently plain text).")
    add_parser.add_argument("--enable-hf-transfer", choices=['true', 'false'], help="Enable/disable hf_transfer for this profile (overrides default).")
    add_parser.add_argument("--description", help="Short description for the profile.")
    remove_parser = profile_subparsers.add_parser("remove", help="Remove a profile.")
    remove_parser.add_argument("name", help="Profile name to remove.")


    # --- Cache Management Command ---
    cache_parser = subparsers.add_parser("cache", help="Manage the HuggingFace Hub cache.")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", required=True, help="Cache action")
    scan_parser = cache_subparsers.add_parser("scan", help="Scan and analyze cache usage.")
    scan_parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    clean_parser = cache_subparsers.add_parser("clean", help="Clean up cached snapshots.")
    clean_parser.add_argument("--older-than", type=int, metavar='DAYS', help="Remove snapshots older than N days.")
    clean_parser.add_argument("--min-size", type=int, metavar='MB', help="Only consider snapshots >= N megabytes for removal based on other criteria.")
    clean_parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting.")
    clean_parser.add_argument("--json", action="store_true", help="Output removed items list as JSON (useful for dry-run).")


    # --- List Downloads Command ---
    list_downloads_parser = subparsers.add_parser("list", help="List download history recorded by this tool.")
    list_downloads_parser.add_argument("--limit", type=int, default=20, help="Limit number of results.")
    list_downloads_parser.add_argument("--category", choices=["models", "datasets", "spaces"], help="Filter by category.")
    list_downloads_parser.add_argument("--filter-profile", help="Show history only for a specific profile name.")
    list_downloads_parser.add_argument("--json", action="store_true", help="Output as JSON.")


    # --- Overview Command ---
    overview_parser = subparsers.add_parser("overview", help="Show overview of the organized files directory.")
    overview_parser.add_argument("--json", action="store_true", help="Output as JSON.")

    return parser

def main():
    """Main entry point for the CLI."""
    parser = _create_parser()
    args = parser.parse_args()

    # Determine if hf_transfer should be enabled/disabled based on flag
    enable_hf_transfer_flag = not args.no_hf_transfer if hasattr(args, 'no_hf_transfer') else None

    # --- Initialize Organizer ---
    try:
         base_path_override = getattr(args, 'base_path', None)
         out_dir_override = getattr(args, 'out_dir', None)

         organizer = HfHubOrganizer(
             profile=args.profile,
             base_path=base_path_override,
             structured_root=out_dir_override,
             enable_hf_transfer=enable_hf_transfer_flag, # Pass the flag value
             verbose=args.verbose,
             config_path=args.config,
             log_format=args.log_format
         )
    except ValueError as e:
         print(f"Error initializing organizer: {e}")
         structlog.get_logger("hfget_cli").error("init_failed", error=str(e))
         exit(1)
    except Exception as e:
         print(f"Unexpected error during initialization: {e}")
         structlog.get_logger("hfget_cli").exception("unexpected_init_failed", error=str(e))
         exit(1)

    # --- Execute Command ---
    try:
        if args.command == "profile":
            if args.profile_command == "list":
                profiles = organizer.list_profiles()
                if profiles:
                    print("Available profiles:")
                    headers = ["Name", "Description", "Cache Path (HF_HOME)", "Organized Root", "HF Transfer"]
                    table = []
                    for name in profiles:
                        p_config = organizer.config["profiles"].get(name, {})
                        desc = p_config.get("description", "N/A")
                        bp = p_config.get("base_path", f"Default ({DEFAULT_HF_HOME})")
                        sr = p_config.get("structured_root", f"Default ({DEFAULT_STRUCTURED_ROOT})")
                        hft = p_config.get("enable_hf_transfer") # Could be True, False, or None
                        hft_str = "Enabled" if hft is True else ("Disabled" if hft is False else f"Default ({'Enabled' if HfHubOrganizer.BOOLEAN_ENV_VARS['HF_HUB_ENABLE_HF_TRANSFER'] else 'Disabled'})")
                        table.append([name, desc, bp, sr, hft_str])
                    print(tabulate(table, headers=headers))
                else:
                    print("No profiles configured. Use 'profile add' to create one.")
                print(f"\nConfig file location: {organizer.config_path}")

            elif args.profile_command == "add":
                # Convert enable_hf_transfer string to boolean or None
                enable_transfer = None
                if args.enable_hf_transfer == 'true':
                    enable_transfer = True
                elif args.enable_hf_transfer == 'false':
                    enable_transfer = False

                organizer.add_profile(
                    name=args.name,
                    base_path=args.base_path,
                    structured_root=args.out_dir,
                    token=args.token,
                    enable_hf_transfer=enable_transfer,
                    description=args.description
                )
                print(f"Profile '{args.name}' added/updated successfully.")
                print(f"Config file: {organizer.config_path}")

            elif args.profile_command == "remove":
                organizer.remove_profile(args.name)
                print(f"Profile '{args.name}' removed (if it existed).")


        elif args.command == "download":
             dl_kwargs = {}
             if args.force_download:
                 dl_kwargs['force_download'] = True
                 dl_kwargs['resume_download'] = False # Often used with force

             path = organizer.download(
                 repo_id=args.repo_id,
                 filename=args.filename,
                 subfolder=args.subfolder,
                 revision=args.revision,
                 category=args.category,
                 symlink_to_cache=not args.copy,
                 allow_patterns=args.allow_patterns,
                 ignore_patterns=args.ignore_patterns,
                 **dl_kwargs
             )
             print(f"\nDownload complete. Organized at: {path}")

        elif args.command == "download-recent":
             if args.exclude_repo_pattern and re.search(args.exclude_repo_pattern, args.repo_id, re.IGNORECASE):
                 print(f"Skipping repository '{args.repo_id}' because it matches exclusion pattern '{args.exclude_repo_pattern}'.")
                 organizer.logger.info("repo_skipped_exclusion", repo_id=args.repo_id, pattern=args.exclude_repo_pattern)
                 return

             dl_kwargs = {}
             if args.force_download:
                 dl_kwargs['force_download'] = True
                 dl_kwargs['resume_download'] = False

             path = organizer.download_recent(
                 repo_id=args.repo_id,
                 days_ago=args.days,
                 subfolder=args.subfolder,
                 revision=args.revision,
                 category=args.category,
                 symlink_to_cache=not args.copy,
                 allow_patterns=args.allow_patterns,
                 ignore_patterns=args.ignore_patterns,
                 **dl_kwargs
             )
             print(f"\nRecent file download process complete. Target directory: {path}")


        elif args.command == "cache":
            if args.cache_command == "scan":
                cache_stats = organizer.get_cache_stats()
                if args.json:
                    print(json.dumps(cache_stats, indent=2, default=str)) # Use default=str for datetime
                else:
                    print(f"HF Cache Statistics (Profile: {args.profile or 'Default'}, Path: {organizer.effective_paths['HF_HUB_CACHE']}):")
                    print(f"-----------------------------------------------------")
                    print(f"Total Size:       {cache_stats['total_size_human']}")
                    print(f"Unique Repos:     {cache_stats['repo_count']}")
                    print(f"Total Snapshots:  {cache_stats['snapshot_count']}")
                    print(f"Total Files:      {cache_stats['file_count']}")
                    print()

                    if cache_stats['largest_snapshots']:
                         print("Largest snapshots (by revision):")
                         headers = ["Repo ID", "Revision", "Size", "Last Modified"]
                         table = [[s['repo_id'], s['revision'][:12], s['size_human'], s['last_modified']] for s in cache_stats['largest_snapshots']]
                         print(tabulate(table, headers=headers))
                         print()
                    else:
                         print("No snapshots found in cache scan.")


                    if cache_stats['organizations']:
                         print("Storage by Organization:")
                         headers = ["Organization", "Size", "Snapshots", "% of Total"]
                         table = []
                         # Show top 10 orgs + "Other" maybe? For now, show all.
                         for org in cache_stats['organizations']:
                             table.append([
                                 org['name'],
                                 org['size_human'],
                                 org['snapshot_count'],
                                 f"{org['percentage']:.1f}%"
                             ])
                         print(tabulate(table, headers=headers))

            elif args.cache_command == "clean":
                if args.older_than is None and args.min_size is None:
                     print("Error: At least one criteria (--older-than DAYS or --min-size MB) must be provided for cleaning.")
                     exit(1)

                removed_count, freed_bytes, removed_details = organizer.clean_cache(
                    older_than_days=args.older_than,
                    min_size_mb=args.min_size,
                    dry_run=args.dry_run
                )
                action_verb = "Would remove" if args.dry_run else "Removed"
                print(f"{action_verb} {removed_count} snapshots, freeing {humanize.naturalsize(freed_bytes)}.")

                if args.json:
                     print(json.dumps(removed_details, indent=2, default=str)) # Use default=str for datetime
                elif removed_details:
                     print(f"\n{action_verb.capitalize()} snapshots:")
                     headers = ["Repo ID", "Revision", "Size", "Last Modified"]
                     table = [[i['repo_id'], i['revision'][:12], i['size_human'], i['last_modified']] for i in removed_details]
                     print(tabulate(table, headers=headers))


        elif args.command == "list":
            downloads = organizer.list_downloads(
                limit=args.limit,
                category=args.category,
                profile_filter=args.filter_profile
            )

            if args.json:
                print(json.dumps(downloads, indent=2, default=str)) # Use default=str for datetime
            else:
                profile_name = args.filter_profile or args.profile or 'Default'
                if not downloads:
                    print(f"No download history found for profile '{profile_name}'.")
                    # Show path based on active profile if no filter used
                    active_root = organizer.structured_root if not args.filter_profile else f"(root for profile '{args.filter_profile}')"
                    print(f"(Metadata file expected in: {os.path.join(active_root, '.metadata', 'downloads.json')})")

                else:
                    print(f"Download History (Profile: {profile_name}, Max: {args.limit}):")
                    headers = ["Timestamp", "Repo ID", "Type", "Category", "Profile", "Revision", "Subfolder", "Rel Path"]
                    table = []
                    for item in downloads:
                        ts = "N/A"
                        try:
                             dt_obj = datetime.datetime.fromisoformat(item.get("timestamp", "")).astimezone(None)
                             ts = dt_obj.strftime("%Y-%m-%d %H:%M")
                        except (ValueError, TypeError):
                             ts = item.get("timestamp", "Invalid Date")[:16]

                        table.append([
                            ts,
                            item.get("repo_id", "N/A"),
                            item.get("type", "N/A"),
                            item.get("category", "N/A"),
                            item.get("profile", "N/A"),
                            item.get("revision", "N/A")[:12], # Shorten revision
                            item.get("subfolder", "-"),
                            item.get("relative_path", "N/A")
                        ])
                    print(tabulate(table, headers=headers, maxcolwidths=[None, None, 15, None, None, 12, 10, 25]))

        elif args.command == "overview":
            overview = organizer.get_organization_overview()
            if args.json:
                print(json.dumps(overview, indent=2, default=str)) # Use default=str for datetime
            else:
                print(f"Organization Overview (Profile: {args.profile or 'Default'}, Root: {organizer.effective_paths['structured_root']})")
                print(f"-----------------------------------------------------")
                print(f"Total Organized Size: {overview.get('total_size_human', '0 B')} (excluding symlinks)")
                print()

                categories = overview.get('categories', {})
                if not categories:
                     print("No categories found in the organized directory.")
                else:
                     sorted_categories = sorted(categories.items())
                     for category, cat_data in sorted_categories:
                         print(f"--- Category: {category} ({cat_data.get('size_human', '0 B')}) ---")
                         orgs_data = cat_data.get('organizations', {})
                         if not orgs_data:
                              print("  No organizations/namespaces found.")
                         else:
                              sorted_orgs = sorted(orgs_data.items(), key=lambda item: item[1].get('size_bytes', 0), reverse=True)
                              headers = ["Organization/Namespace", "Size", "Repositories"]
                              table = [[name, org_info.get('size_human', '0 B'), org_info.get('repo_count', 0)] for name, org_info in sorted_orgs]
                              print(tabulate(table, headers=headers, tablefmt="plain"))
                         print()


    except RepositoryNotFoundError as e:
         organizer.logger.error("command_failed_repo_not_found", repo_id=getattr(args, 'repo_id', 'N/A'), error=str(e))
         print(f"Error: Repository not found: {getattr(args, 'repo_id', 'N/A')}")
         exit(1)
    except HfHubHTTPError as http_err:
         organizer.logger.error("command_failed_http_error", command=args.command, status=http_err.response.status_code, error=str(http_err))
         print(f"\nAn HTTP error occurred: Status {http_err.response.status_code} - {http_err}")
         if http_err.response.status_code == 401:
              print("Hint: Check if your HF_TOKEN is valid and has the necessary permissions.")
         exit(1)
    except Exception as e:
         organizer.logger.exception("command_failed_unexpected", command=args.command, error=str(e))
         print(f"\nAn unexpected error occurred: {e}")
         print("Check logs or run with --verbose for more details.")
         exit(1)

if __name__ == "__main__":
    main()
