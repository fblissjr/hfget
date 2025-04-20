import os
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Optional, Union, List, Any, Tuple
import datetime
import humanize
from tabulate import tabulate
import argparse
import logging  # Standard library logging for level constants
import re  # Import regex for case-insensitive check

try:
    import structlog

    # --- FIX 3: Add list_repo_files_info to imports ---
    from huggingface_hub import (
        HfApi,
        snapshot_download,
        hf_hub_download,
        scan_cache_dir,
        list_repo_files_info,  # <-- Added import
    )
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    from huggingface_hub.hf_api import RepoFile
except ImportError:
    missing = []
    # Check main libraries first
    for lib in ["structlog", "huggingface_hub", "tabulate", "humanize"]:
        try:
            __import__(lib)
        except ImportError:
            if lib not in missing:
                missing.append(lib)

    # If huggingface_hub seems present, check for specific components needed
    if "huggingface_hub" not in missing:
        try:
            from huggingface_hub.utils import RepositoryNotFoundError
        except ImportError:
            missing.append("huggingface_hub.utils (RepositoryNotFoundError)")
        try:
            # --- FIX 3: Also check list_repo_files_info here ---
            from huggingface_hub import list_repo_files_info
        except ImportError:
            missing.append("huggingface_hub (list_repo_files_info)")
        try:
            from huggingface_hub.hf_api import RepoFile
        except ImportError:
            missing.append("huggingface_hub.hf_api (RepoFile)")

    if missing:
        # Make suggestion more robust
        install_libs = set()
        for m in missing:
            lib_name = m.split(" ")[0].split(".")[0]  # Get base library name
            install_libs.add(lib_name)
        raise ImportError(
            f"Required libraries/components not found: {', '.join(missing)}. Install/update with: pip install --upgrade {' '.join(install_libs)}"
        )


class HfHubOrganizer:
    """
    A wrapper that makes huggingface_hub behave in a sane, organized way with support
    for different storage profiles and advanced cache management.
    """

    ENV_VARS = {
        "HF_HOME": "~/.cache/huggingface",
        "HF_HUB_CACHE": "${HF_HOME}/hub",
        "HF_XET_CACHE": "${HF_HOME}/xet",
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
        "HF_HUB_DISABLE_TELEMETRY": True,  # Default to privacy-friendly
        "HF_HUB_ENABLE_HF_TRANSFER": False
    }

    def __init__(
        self,
        profile: Optional[str] = None,
        base_path: Optional[str] = None,
        structured_root: Optional[str] = None,
        token: Optional[str] = None,
        verbose: bool = False,
        config_path: Optional[str] = None,
        log_format: str = "console",
    ):
        """Initialize with custom paths and settings."""
        # Set config path
        self.config_path = os.path.expanduser(config_path or "~/.config/hf_organizer/config.json")
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        # Setup structured logging
        self.logger = self._setup_logger(verbose, log_format)

        # Load or create config
        self.config = self._load_config()
        self.selected_profile = profile

        # Load profile settings if specified
        if profile:
            if profile not in self.config.get("profiles", {}):
                self.logger.error(
                    "profile_not_found",
                    profile=profile,
                    available=list(self.config.get("profiles", {}).keys()),
                )
                raise ValueError(f"Profile '{profile}' not found.")

            profile_settings = self.config["profiles"][profile]
            base_path = profile_settings.get("base_path", base_path)
            structured_root = profile_settings.get("structured_root", structured_root)
            token = profile_settings.get("token", token)

            self.logger.info("using_profile", profile=profile)

        # Set base HF path if provided
        if base_path:
            os.environ["HF_HOME"] = os.path.expanduser(base_path)

        # Set organized structure root
        self.structured_root = os.path.expanduser(
            structured_root or "~/huggingface_organized"
        )
        os.makedirs(self.structured_root, exist_ok=True)

        # Set token if provided
        if token:
            os.environ["HF_TOKEN"] = token
        elif "HF_TOKEN" not in os.environ:
            # Use token from profile if not overridden and not in env
            profile_token = (
                self.config.get("profiles", {}).get(profile, {}).get("token")
            )
            if profile_token:
                os.environ["HF_TOKEN"] = profile_token

        # Initialize all environment variables
        self._initialize_env_vars()

        # Keep track of effective paths
        self.effective_paths = {
            "HF_HOME": os.path.expanduser(
                os.environ.get("HF_HOME", "~/.cache/huggingface")
            ),
            "HF_HUB_CACHE": os.path.expanduser(
                os.environ.get(
                    "HF_HUB_CACHE",
                    os.path.join(
                        os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"
                    ),
                )
            ),  # Ensure default hub path is derived correctly
            "structured_root": self.structured_root,
        }
        # Ensure HF_HUB_CACHE is explicitly set if derived, as other parts rely on it
        if "HF_HUB_CACHE" not in os.environ or not os.environ["HF_HUB_CACHE"]:
            os.environ["HF_HUB_CACHE"] = self.effective_paths["HF_HUB_CACHE"]

        self.logger.info(
            "initialized",
            profile=profile,
            structured_root=self.structured_root,
            hf_home=self.effective_paths["HF_HOME"],
        )

        # Initialize HF API
        # Pass token explicitly if available, otherwise it reads from env var HF_TOKEN
        self.api = HfApi(token=os.environ.get("HF_TOKEN"))

    def _setup_logger(self, verbose: bool, format_type: str) -> structlog.BoundLogger:
        """Set up structured logging."""
        log_level = logging.DEBUG if verbose else logging.INFO

        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        ]

        if format_type == "json":
            processors.extend(
                [
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ]
            )
        elif format_type == "structured":
            processors.extend(
                [
                    structlog.processors.format_exc_info,
                    structlog.dev.ConsoleRenderer(),  # Simple structured, less pretty
                ]
            )
        else:  # console (default)
            processors.extend(
                [
                    structlog.dev.set_exc_info,
                    structlog.dev.ConsoleRenderer(colors=True),  # Pretty console
                ]
            )

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(organizer=self.__class__.__name__)

    def _load_config(self) -> Dict[str, Any]:
        """Load config from disk or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    self.logger.debug("config_loaded", path=self.config_path)
                    return config
            except json.JSONDecodeError:
                self.logger.warning(
                    "invalid_config_file",
                    path=self.config_path,
                    action="creating_default",
                )
                return {"profiles": {}}
            except Exception as e:
                self.logger.error(
                    "config_load_failed", path=self.config_path, error=str(e)
                )
                return {"profiles": {}}
        self.logger.debug("default_config_created", path=self.config_path)
        return {"profiles": {}}

    def _save_config(self):
        """Save current config to disk."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
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
        description: Optional[str] = None,
    ):
        """Add or update a profile."""
        if "profiles" not in self.config:
            self.config["profiles"] = {}

        # Ensure paths are stored expanded but handle potential None
        profile_data = {
            "base_path": os.path.expanduser(base_path) if base_path else None,
            "structured_root": os.path.expanduser(structured_root)
            if structured_root
            else None,
            "token": token,  # Keep token as is (might be None)
            "description": description or f"Profile created for {name}",
        }

        # Remove None values so they don't override defaults unnecessarily
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

    def _initialize_env_vars(self):
        """Initialize environment variables with defaults if not already set."""
        # Process string environment variables
        for key, default_value in self.ENV_VARS.items():
            if key not in os.environ and default_value is not None:
                # Handle variable expansion in default values like ${HF_HOME}
                expanded_value = default_value
                # Simple expansion loop (might need more robust solution for complex cases)
                for _ in range(3):  # Limit expansion depth
                    updated = False
                    # Use current os.environ for expansion lookup
                    env_dict = os.environ
                    # Temporarily add HF_HOME if it's being defined based on ~
                    if key == "HF_HOME" and "~" in default_value:
                        env_dict = {
                            **os.environ,
                            "HF_HOME": os.path.expanduser(default_value),
                        }

                    for var_name, var_value in env_dict.items():
                        placeholder = "${" + var_name + "}"
                        if placeholder in expanded_value:
                            expanded_value = expanded_value.replace(
                                placeholder, var_value
                            )
                            updated = True
                    if not updated:
                        break  # No more placeholders found

                # Expand user path ~
                final_value = os.path.expanduser(expanded_value)
                os.environ[key] = final_value
                self.logger.debug("env_var_set_default", key=key, value=final_value)

        # Process boolean environment variables
        for key, default_value in self.BOOLEAN_ENV_VARS.items():
            if key not in os.environ:
                os.environ[key] = "1" if default_value else "0"
                self.logger.debug(
                    "env_var_set_default_bool", key=key, value=os.environ[key]
                )

    def _determine_category_and_paths(
        self,
        repo_id: str,
        category: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> Tuple[str, str, str, str]:
        """Determine the category and create the organized path structure."""
        repo_type_guess = "models"  # Default guess
        if category is None:
            if "/" in repo_id:
                namespace, repo_name = repo_id.split("/", 1)
                try:
                    # Use api.repo_type() which is the correct method
                    repo_type = self.api.repo_type(repo_id=repo_id)
                    if repo_type == "dataset":
                        repo_type_guess = "datasets"
                    elif repo_type == "space":
                        repo_type_guess = "spaces"
                    else:
                        repo_type_guess = "models"  # Explicitly models
                    self.logger.debug(
                        "category_detected_via_api",
                        repo_id=repo_id,
                        category=repo_type_guess,
                    )
                except RepositoryNotFoundError:
                    self.logger.error("repo_not_found_api", repo_id=repo_id)
                    raise  # Re-raise the specific error
                except Exception as e:
                    # Log the actual error here for better debugging
                    self.logger.warning(
                        "category_detection_failed_api",
                        repo_id=repo_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        fallback=repo_type_guess,
                    )
            else:
                # Assume it's a model under the implicit 'library' or 'huggingface' namespace
                namespace = "library"  # Or choose another default namespace
                repo_name = repo_id
                repo_type_guess = "models"
        else:
            # User provided category overrides detection
            repo_type_guess = category
            if "/" in repo_id:
                namespace, repo_name = repo_id.split("/", 1)
            else:
                namespace = "library"
                repo_name = repo_id

        # Create organized base path for the repo
        org_repo_path = os.path.join(
            self.structured_root, repo_type_guess, namespace, repo_name
        )

        # Full path including potential subfolder for downloads
        org_download_path = (
            os.path.join(org_repo_path, subfolder) if subfolder else org_repo_path
        )

        os.makedirs(org_download_path, exist_ok=True)
        self.logger.debug("organizing_files_target", path=org_download_path)

        return (
            org_repo_path,
            org_download_path,
            repo_type_guess,
            namespace,
        )  # Return namespace too

    def _link_or_copy(self, cache_path: str, org_path: str, symlink_to_cache: bool):
        """Helper to symlink or copy a file/directory."""
        if os.path.lexists(org_path):  # Use lexists to check for broken symlinks too
            if os.path.islink(org_path) or os.path.isfile(org_path):
                os.remove(org_path)
                self.logger.debug("removed_existing_target", path=org_path)
            elif os.path.isdir(org_path):
                # Ensure we are not removing the cache path itself if symlinking fails
                # Use realpath to resolve symlinks before comparison
                try:
                    if not os.path.samefile(
                        os.path.realpath(org_path), os.path.realpath(cache_path)
                    ):
                        shutil.rmtree(org_path)
                        self.logger.debug("removed_existing_target_dir", path=org_path)
                    else:
                        self.logger.warning(
                            "skipping_removal_target_is_cache", path=org_path
                        )
                except FileNotFoundError:
                    # If realpath fails (e.g., broken link target), safe to remove the link/dir itself
                    shutil.rmtree(org_path)
                    self.logger.debug(
                        "removed_existing_broken_link_or_dir", path=org_path
                    )
                except OSError as e:
                    self.logger.error(
                        "error_comparing_paths_before_removal",
                        org_path=org_path,
                        cache_path=cache_path,
                        error=str(e),
                    )
                    # Decide whether to proceed with removal or raise error - safer to skip removal
                    self.logger.warning(
                        "skipping_removal_due_to_path_comparison_error", path=org_path
                    )

        os.makedirs(os.path.dirname(org_path), exist_ok=True)  # Ensure parent exists

        if symlink_to_cache:
            try:
                # Ensure cache_path is absolute for reliable symlinking
                abs_cache_path = os.path.abspath(cache_path)
                os.symlink(abs_cache_path, org_path)
                self.logger.debug(
                    "symlink_created", source=abs_cache_path, target=org_path
                )
            except OSError as e:
                self.logger.error(
                    "symlink_failed", source=cache_path, target=org_path, error=str(e)
                )
                # Fallback or raise? For now, log error.
                raise  # Let the error propagate
        else:
            try:
                if os.path.isdir(cache_path):
                    shutil.copytree(
                        cache_path, org_path, symlinks=True
                    )  # Preserve symlinks within copied structure if any
                    self.logger.debug(
                        "directory_copied", source=cache_path, target=org_path
                    )
                else:
                    shutil.copy2(cache_path, org_path)  # copy2 preserves metadata
                    self.logger.debug("file_copied", source=cache_path, target=org_path)
            except Exception as e:
                self.logger.error(
                    "copy_failed", source=cache_path, target=org_path, error=str(e)
                )
                raise  # Let the error propagate

    def download(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        category: Optional[str] = None,
        symlink_to_cache: bool = True,
        **kwargs,  # Pass extra args like allow_patterns to underlying functions
    ) -> str:
        """
        Download a full repository or a specific file and organize it.
        """
        start_time = time.time()
        self.logger.info(
            "download_started",
            repo_id=repo_id,
            filename=filename or "entire_repo",
            subfolder=subfolder,
        )

        try:
            org_repo_path, org_download_path, detected_category, _ = (
                self._determine_category_and_paths(repo_id, category, subfolder)
            )

            # Save metadata about this download attempt (even if it fails later)
            self._save_download_metadata(
                org_repo_path,
                repo_id,
                detected_category,
                filename or "entire_repo",
                subfolder,
            )

            downloaded_path_in_cache: str
            final_organized_path: str

            if filename:
                # --- Download a specific file ---
                downloaded_path_in_cache = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder,
                    revision=revision,
                    repo_type=detected_category
                    if detected_category in ["models", "datasets", "spaces"]
                    else None,
                    token=self.api.token,  # Pass token
                    **kwargs,
                )
                # Target path in the organized structure
                final_organized_path = os.path.join(
                    org_download_path, os.path.basename(filename)
                )
                self._link_or_copy(
                    downloaded_path_in_cache, final_organized_path, symlink_to_cache
                )

            else:
                # --- Download entire repo (or subfolder) ---
                downloaded_path_in_cache = snapshot_download(
                    repo_id=repo_id,
                    subfolder=subfolder,
                    revision=revision,
                    repo_type=detected_category
                    if detected_category in ["models", "datasets", "spaces"]
                    else None,
                    token=self.api.token,  # Pass token
                    **kwargs,
                )
                # Target path is the directory itself
                final_organized_path = org_download_path
                # snapshot_download returns the path to the *root* of the snapshot,
                # even if a subfolder was requested. We need to handle linking/copying
                # the contents correctly into the target org_download_path.

                # Remove existing target dir before linking/copying contents
                if os.path.lexists(final_organized_path):  # Use lexists for symlinks
                    if os.path.islink(final_organized_path):
                        os.remove(final_organized_path)
                    elif os.path.isdir(final_organized_path):
                        # Check if it points to the cache before removing
                        try:
                            if not os.path.samefile(
                                os.path.realpath(final_organized_path),
                                os.path.realpath(downloaded_path_in_cache),
                            ):
                                shutil.rmtree(final_organized_path)
                                self.logger.debug(
                                    "removed_existing_target_dir_for_snapshot",
                                    path=final_organized_path,
                                )
                            else:
                                self.logger.warning(
                                    "skipping_removal_snapshot_target_is_cache",
                                    path=final_organized_path,
                                )
                        except FileNotFoundError:
                            shutil.rmtree(
                                final_organized_path
                            )  # Remove broken link/dir
                            self.logger.debug(
                                "removed_existing_broken_link_or_dir_snapshot",
                                path=final_organized_path,
                            )
                        except OSError as e:
                            self.logger.error(
                                "error_comparing_paths_snapshot",
                                org_path=final_organized_path,
                                cache_path=downloaded_path_in_cache,
                                error=str(e),
                            )
                            self.logger.warning(
                                "skipping_removal_due_to_path_comparison_error_snapshot",
                                path=final_organized_path,
                            )

                    else:  # It's a file? Remove it.
                        os.remove(final_organized_path)

                os.makedirs(final_organized_path, exist_ok=True)

                # Link or copy contents item by item
                item_count = 0
                for item_name in os.listdir(downloaded_path_in_cache):
                    cache_item_path = os.path.join(downloaded_path_in_cache, item_name)
                    org_item_path = os.path.join(final_organized_path, item_name)
                    self._link_or_copy(cache_item_path, org_item_path, symlink_to_cache)
                    item_count += 1
                self.logger.debug(
                    "snapshot_contents_processed",
                    count=item_count,
                    source=downloaded_path_in_cache,
                    target=final_organized_path,
                )

            elapsed = time.time() - start_time
            self.logger.info(
                "download_completed",
                repo_id=repo_id,
                organized_path=final_organized_path,
                elapsed_seconds=round(elapsed, 2),
                profile=self.selected_profile,
            )

            return final_organized_path

        except RepositoryNotFoundError:
            self.logger.error("download_failed_repo_not_found", repo_id=repo_id)
            raise  # Re-raise for CLI handling
        except Exception as e:
            self.logger.error(
                "download_failed",
                repo_id=repo_id,
                filename=filename,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise  # Re-raise for CLI handling

    def download_recent(
        self,
        repo_id: str,
        days_ago: int,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        category: Optional[str] = None,
        symlink_to_cache: bool = True,
        **kwargs,  # Pass extra args like allow_patterns to underlying functions
    ) -> str:
        """
        Download only files modified within the last N days for a repository.
        Assumes repo_id has already been validated against exclusion patterns by the caller (main).
        """
        start_time = time.time()
        self.logger.info(
            "download_recent_started",
            repo_id=repo_id,
            days_ago=days_ago,
            subfolder=subfolder,
        )

        try:
            # Determine target paths and category
            org_repo_path, org_download_path, detected_category, _ = (
                self._determine_category_and_paths(
                    repo_id,
                    category,
                    subfolder,  # Files will land relative to this path
                )
            )

            # Calculate the cutoff date (make it timezone-aware UTC)
            cutoff_date = datetime.datetime.now(
                datetime.timezone.utc
            ) - datetime.timedelta(days=days_ago)
            self.logger.debug(
                "filtering_commits_since", cutoff_date=cutoff_date.isoformat()
            )

            # Get file info including commit details
            # Use token explicitly if needed, otherwise API client uses env var
            # --- This call should now work because list_repo_files_info is imported ---
            all_files_info: List[RepoFile] = list_repo_files_info(
                repo_id=repo_id,
                revision=revision,
                # repo_type=detected_category if detected_category in ["models", "datasets", "spaces"] else None, # repo_type not needed here
                token=self.api.token,  # Pass token from initialized API client
            )

            # Filter files based on last commit date
            recent_files_to_download: List[RepoFile] = []
            for file_info in all_files_info:
                # Ensure commit info and date exist, and compare with cutoff
                if (
                    file_info.last_commit
                    and file_info.last_commit.date
                    and file_info.last_commit.date > cutoff_date
                ):
                    # Check if the file is within the requested subfolder (if specified)
                    if subfolder:
                        # file_info.path is relative to repo root
                        # Ensure consistent trailing slash handling for comparison
                        norm_subfolder = subfolder.strip("/")
                        # Check if path starts with subfolder/ or is exactly subfolder (if it's a file)
                        if (
                            file_info.path == norm_subfolder
                            or file_info.path.startswith(norm_subfolder + "/")
                        ):
                            recent_files_to_download.append(file_info)
                            self.logger.debug(
                                "found_recent_file_in_subfolder",
                                file=file_info.path,
                                commit_date=file_info.last_commit.date,
                            )
                    else:
                        # No subfolder specified, include all recent files
                        recent_files_to_download.append(file_info)
                        self.logger.debug(
                            "found_recent_file",
                            file=file_info.path,
                            commit_date=file_info.last_commit.date,
                        )

            if not recent_files_to_download:
                self.logger.info(
                    "no_recent_files_found",
                    repo_id=repo_id,
                    days_ago=days_ago,
                    subfolder=subfolder,
                )
                # Still save metadata indicating an attempt was made
                self._save_download_metadata(
                    org_repo_path,
                    repo_id,
                    detected_category,
                    f"recent_{days_ago}d",
                    subfolder,
                )
                return (
                    org_download_path  # Return the base path even if nothing downloaded
                )

            self.logger.info(
                "downloading_recent_files", count=len(recent_files_to_download)
            )

            # Download each recent file individually
            downloaded_count = 0
            for file_info in recent_files_to_download:
                try:
                    # Determine the correct subfolder relative to the repo root for hf_hub_download
                    file_repo_subfolder = os.path.dirname(file_info.path)
                    file_basename = os.path.basename(file_info.path)

                    self.logger.debug(
                        "downloading_individual_recent_file", file=file_info.path
                    )
                    downloaded_path_in_cache = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_basename,  # Just the filename
                        subfolder=file_repo_subfolder
                        if file_repo_subfolder
                        else None,  # Subfolder relative to repo root
                        revision=revision,  # Use specific revision if needed
                        repo_type=detected_category
                        if detected_category in ["models", "datasets", "spaces"]
                        else None,
                        token=self.api.token,  # Pass token
                        **kwargs,
                    )

                    # Determine the final organized path for this specific file
                    # It should mirror the structure within the repo, relative to org_download_path
                    relative_file_path = file_info.path  # Path relative to repo root
                    if subfolder:
                        # Adjust relative path if user requested a subfolder download
                        norm_subfolder = subfolder.strip("/")
                        if file_info.path.startswith(norm_subfolder + "/"):
                            relative_file_path = os.path.relpath(
                                file_info.path, norm_subfolder
                            )
                        # Handle case where the file itself is the subfolder target (unlikely but possible)
                        elif file_info.path == norm_subfolder:
                            relative_file_path = os.path.basename(file_info.path)

                    final_organized_path = os.path.join(
                        org_download_path, relative_file_path
                    )

                    # Link or copy the downloaded file
                    self._link_or_copy(
                        downloaded_path_in_cache, final_organized_path, symlink_to_cache
                    )
                    downloaded_count += 1

                except Exception as e_file:
                    self.logger.error(
                        "failed_downloading_recent_file",
                        file=file_info.path,
                        error=str(e_file),
                    )
                    # Continue with other files? Or stop? For now, continue.

            elapsed = time.time() - start_time
            self.logger.info(
                "download_recent_completed",
                repo_id=repo_id,
                files_downloaded=downloaded_count,
                target_path=org_download_path,
                elapsed_seconds=round(elapsed, 2),
                profile=self.selected_profile,
            )

            # Save metadata indicating a successful recent download
            self._save_download_metadata(
                org_repo_path,
                repo_id,
                detected_category,
                f"recent_{days_ago}d",
                subfolder,
            )

            return org_download_path  # Return the base path where files were placed

        except RepositoryNotFoundError:
            # This exception should now be defined
            self.logger.error("download_failed_repo_not_found", repo_id=repo_id)
            raise
        except Exception as e:
            self.logger.error(
                "download_recent_failed",
                repo_id=repo_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _save_download_metadata(
        self,
        path: str,
        repo_id: str,
        category: str,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None,
    ):
        """Save metadata about downloaded repo/file for future reference."""
        # Use metadata dir within the *selected profile's* structured root
        metadata_dir = os.path.join(self.structured_root, ".metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_file = os.path.join(metadata_dir, "downloads.json")

        # Load existing metadata
        metadata = {"downloads": []}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    if not isinstance(metadata.get("downloads"), list):
                        self.logger.warning(
                            "invalid_metadata_format_resetting", path=metadata_file
                        )
                        metadata = {"downloads": []}
            except json.JSONDecodeError:
                self.logger.warning(
                    "invalid_metadata_file_resetting", path=metadata_file
                )
                metadata = {"downloads": []}
            except Exception as e:
                self.logger.error(
                    "failed_loading_metadata", path=metadata_file, error=str(e)
                )
                # Proceed with empty metadata to avoid losing new entry
                metadata = {"downloads": []}

        # Add new entry
        entry = {
            "repo_id": repo_id,
            "category": category,
            # Store path relative to the structured root for portability
            "relative_path": os.path.relpath(path, self.structured_root),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "profile": self.selected_profile,  # Record which profile was used
        }
        if filename:  # Can be "entire_repo", specific file, or "recent_Xd"
            entry["type"] = filename
        if subfolder:
            entry["subfolder"] = subfolder

        # Prepend new entry (most recent first)
        metadata["downloads"].insert(0, entry)

        # Optional: Limit history size
        max_history = 500
        if len(metadata["downloads"]) > max_history:
            metadata["downloads"] = metadata["downloads"][:max_history]

        # Save updated metadata
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(
                "failed_saving_metadata", path=metadata_file, error=str(e)
            )

    def list_downloads(
        self,
        limit: Optional[int] = None,
        category: Optional[str] = None,
        profile_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List download history with filtering options."""
        # Use metadata dir within the *selected profile's* structured root
        # Or should it list across all profiles? Let's assume current profile for now.
        metadata_file = os.path.join(
            self.structured_root, ".metadata", "downloads.json"
        )

        if not os.path.exists(metadata_file):
            self.logger.warning("no_download_history_found", path=metadata_file)
            return []

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            self.logger.error(
                "failed_loading_metadata_for_list", path=metadata_file, error=str(e)
            )
            return []

        downloads = metadata.get("downloads", [])

        # Apply filters
        filtered_downloads = downloads
        if category:
            filtered_downloads = [
                d for d in filtered_downloads if d.get("category") == category
            ]
        if profile_filter:
            # Filter by the profile recorded in the metadata entry
            filtered_downloads = [
                d for d in filtered_downloads if d.get("profile") == profile_filter
            ]

        # Apply limit (already sorted newest first due to insertion order)
        if limit and limit > 0:
            filtered_downloads = filtered_downloads[:limit]

        return filtered_downloads

    def scan_cache(self) -> List[Dict[str, Any]]:
        """
        Scan the HF cache directory associated with the current profile/HF_HOME.
        Provides a summary per repo/revision found in the cache.
        Note: This is a filesystem scan and might not perfectly match HF internal state.
        """
        # Use the effective cache path based on initialized env vars
        cache_dir = self.effective_paths["HF_HUB_CACHE"]

        if not os.path.exists(cache_dir):
            self.logger.warning("cache_dir_not_found", path=cache_dir)
            return []

        self.logger.info("scanning_cache", path=cache_dir)
        scan_results = []
        processed_dirs = set()  # Avoid double counting if symlinks exist within cache

        try:
            # Walk the cache directory
            for root, dirs, files in os.walk(cache_dir, topdown=True):
                # Skip directories we know are internal or irrelevant
                dirs[:] = [
                    d for d in dirs if d not in [".locks", ".temp"]
                ]  # Modify dirs in place

                # Check if current root looks like a repo snapshot directory
                # Example: .../hub/models--google--flan-t5-base/snapshots/abcdef12345...
                # Example: .../hub/datasets--squad/snapshots/abcdef12345...
                path_parts = Path(root).parts
                if len(path_parts) > 2 and path_parts[-2] == "snapshots":
                    snapshot_hash = path_parts[-1]
                    repo_dir_name = path_parts[-3]  # e.g., models--google--flan-t5-base

                    # Prevent processing the same snapshot dir multiple times if traversed via symlink
                    try:
                        real_path = os.path.realpath(root)
                        if real_path in processed_dirs:
                            continue
                        processed_dirs.add(real_path)
                    except (
                        OSError
                    ):  # Handle cases where realpath might fail (e.g., broken links)
                        self.logger.warning("realpath_failed_skipping_dir", path=root)
                        continue

                    # Decode repo_id and type
                    repo_type = "unknown"
                    repo_id = repo_dir_name  # Default if no type prefix
                    if repo_dir_name.startswith("models--"):
                        repo_type = "model"
                        repo_id = repo_dir_name[len("models--") :].replace("--", "/", 1)
                    elif repo_dir_name.startswith("datasets--"):
                        repo_type = "dataset"
                        repo_id = repo_dir_name[len("datasets--") :].replace(
                            "--", "/", 1
                        )
                    elif repo_dir_name.startswith("spaces--"):
                        repo_type = "space"
                        repo_id = repo_dir_name[len("spaces--") :].replace("--", "/", 1)

                    # Calculate size and file count for this specific snapshot directory
                    dir_size = 0
                    file_count = 0
                    largest_file = {"name": "", "size": 0}
                    try:
                        for item in os.listdir(root):
                            item_path = os.path.join(root, item)
                            # Important: Check if it's a file and *not* a symlink pointing outside?
                            # For size calculation, follow symlinks *within* the snapshot? Let's not follow for simplicity.
                            if os.path.isfile(item_path) and not os.path.islink(
                                item_path
                            ):
                                try:
                                    file_size = os.path.getsize(item_path)
                                    dir_size += file_size
                                    file_count += 1
                                    if file_size > largest_file["size"]:
                                        largest_file = {"name": item, "size": file_size}
                                except OSError:
                                    self.logger.warning(
                                        "getsize_failed_file", path=item_path
                                    )
                            elif os.path.isdir(item_path) and not os.path.islink(
                                item_path
                            ):
                                # Recursively calculate size of subdirectories within snapshot
                                for sub_root, _, sub_files in os.walk(item_path):
                                    for sub_file in sub_files:
                                        sub_file_path = os.path.join(sub_root, sub_file)
                                        if os.path.isfile(
                                            sub_file_path
                                        ) and not os.path.islink(sub_file_path):
                                            try:
                                                file_size = os.path.getsize(
                                                    sub_file_path
                                                )
                                                dir_size += file_size
                                                file_count += 1
                                                if file_size > largest_file["size"]:
                                                    largest_file = {
                                                        "name": os.path.relpath(
                                                            sub_file_path, root
                                                        ),
                                                        "size": file_size,
                                                    }
                                            except OSError:
                                                self.logger.warning(
                                                    "getsize_failed_subfile",
                                                    path=sub_file_path,
                                                )

                    except OSError as e:
                        self.logger.warning(
                            "cache_scan_error_accessing_dir",
                            directory=root,
                            error=str(e),
                        )
                        continue  # Skip this directory if not accessible

                    # Get last modified time (use dir mtime as approximation)
                    try:
                        last_modified_ts = os.path.getmtime(root)
                        last_modified_dt = datetime.datetime.fromtimestamp(
                            last_modified_ts, tz=datetime.timezone.utc
                        )
                    except OSError:
                        last_modified_dt = None  # Handle cases where mtime fails

                    scan_results.append(
                        {
                            "repo_id": repo_id,
                            "repo_type": repo_type,
                            "revision": snapshot_hash,  # This is the commit hash (snapshot hash)
                            "size_bytes": dir_size,
                            "size_human": humanize.naturalsize(dir_size),
                            "file_count": file_count,
                            "largest_file": largest_file,
                            "last_modified": last_modified_dt.isoformat()
                            if last_modified_dt
                            else None,
                            "cache_path": root,  # Store the path for potential deletion
                        }
                    )

            # Sort by size (largest first)
            scan_results.sort(key=lambda x: x["size_bytes"], reverse=True)
            self.logger.info("cache_scan_complete", entries_found=len(scan_results))
            return scan_results

        except Exception as e:
            self.logger.error(
                "cache_scan_failed",
                path=cache_dir,
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache usage based on scan results."""
        cache_items = self.scan_cache()  # Use the detailed scan

        if not cache_items:
            return {
                "total_size": 0,
                "total_size_human": "0 B",
                "repo_count": 0,  # Count unique repo_ids
                "snapshot_count": 0,  # Count individual snapshots/revisions
                "file_count": 0,
                "largest_snapshots": [],
                "organizations": [],
            }

        total_size = sum(item["size_bytes"] for item in cache_items)
        total_files = sum(item["file_count"] for item in cache_items)
        unique_repos = {item["repo_id"] for item in cache_items}

        # Get top 5 largest snapshots (revisions)
        largest_snapshots = sorted(
            cache_items, key=lambda x: x["size_bytes"], reverse=True
        )[:5]

        # Group by organization
        orgs = {}
        for item in cache_items:
            repo_id = item["repo_id"]
            # Determine organization/namespace
            if "/" in repo_id:
                org = repo_id.split("/")[0]
            else:
                org = "library"  # Use 'library' for top-level models like 'gpt2'

            if org not in orgs:
                orgs[org] = {"size_bytes": 0, "snapshot_count": 0}

            orgs[org]["size_bytes"] += item["size_bytes"]
            orgs[org]["snapshot_count"] += 1  # Count each snapshot under the org

        # Calculate percentages and format sizes for organizations
        formatted_orgs = []
        for org, stats in orgs.items():
            percentage = (
                (stats["size_bytes"] / total_size) * 100 if total_size > 0 else 0
            )
            formatted_orgs.append(
                {
                    "name": org,
                    "size_bytes": stats["size_bytes"],
                    "size_human": humanize.naturalsize(stats["size_bytes"]),
                    "snapshot_count": stats["snapshot_count"],
                    "percentage": percentage,
                }
            )

        # Sort organizations by size
        top_orgs = sorted(formatted_orgs, key=lambda x: x["size_bytes"], reverse=True)

        return {
            "total_size": total_size,
            "total_size_human": humanize.naturalsize(total_size),
            "repo_count": len(unique_repos),
            "snapshot_count": len(cache_items),
            "file_count": total_files,
            "largest_snapshots": largest_snapshots,  # Show largest individual snapshots
            "organizations": top_orgs,
        }

    def clean_cache(
        self,
        older_than_days: Optional[int] = None,
        min_size_mb: Optional[int] = None,
        dry_run: bool = False,
    ) -> Tuple[int, int, List[Dict]]:
        """
        Clean up cached snapshots (revisions) based on age or size criteria.

        Args:
            older_than_days: Remove snapshots older than this many days.
            min_size_mb: Only consider snapshots larger than this size in MB for removal criteria (removes if >=).
            dry_run: If True, only report what would be removed.

        Returns:
            Tuple of (number of snapshots removed, bytes freed, list of removed items details).
        """
        cache_items = self.scan_cache()  # Get detailed list including paths
        action = "dry_run" if dry_run else "clean_cache"
        self.logger.info(
            f"{action}_started",
            older_than_days=older_than_days,
            min_size_mb=min_size_mb,
        )

        if not cache_items:
            self.logger.info("cache_empty")
            return (0, 0, [])

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        freed_bytes = 0
        removed_count = 0
        removed_items_details = []

        items_to_remove = []

        # --- Filtering Logic ---
        for item in cache_items:
            # Determine if item meets criteria for removal
            meets_age_criteria = False
            if older_than_days is not None:
                if item["last_modified"]:
                    try:
                        last_mod_dt = datetime.datetime.fromisoformat(
                            item["last_modified"]
                        )
                        if last_mod_dt.tzinfo is None:
                            last_mod_dt = last_mod_dt.replace(
                                tzinfo=datetime.timezone.utc
                            )
                        if (now_utc - last_mod_dt).days >= older_than_days:
                            meets_age_criteria = True
                    except ValueError:
                        self.logger.warning(
                            "invalid_date_format_for_clean",
                            repo_id=item["repo_id"],
                            revision=item["revision"],
                            date_str=item["last_modified"],
                        )
                # else: keep if no date? Or remove? Let's default to keeping if date is missing.

            meets_size_criteria = False
            if min_size_mb is not None:
                size_mb = item["size_bytes"] / (1024 * 1024)
                if size_mb >= min_size_mb:
                    meets_size_criteria = True

            # Decide whether to remove based on provided arguments
            should_remove = False
            if older_than_days is not None and min_size_mb is not None:
                # Must meet BOTH age and size criteria
                if meets_age_criteria and meets_size_criteria:
                    should_remove = True
            elif older_than_days is not None:
                # Only age criteria given
                if meets_age_criteria:
                    should_remove = True
            elif min_size_mb is not None:
                # Only size criteria given
                if meets_size_criteria:
                    should_remove = True
            # If neither criteria is given, should_remove remains False

            if should_remove:
                items_to_remove.append(item)
            else:
                self.logger.debug(
                    "keeping_item_criteria_not_met",
                    repo_id=item["repo_id"],
                    revision=item["revision"],
                )

        # --- Execution Phase ---
        if not items_to_remove:
            self.logger.info(f"{action}_no_items_match_criteria")
            return (0, 0, [])

        self.logger.info(f"{action}_items_to_process", count=len(items_to_remove))

        for item in items_to_remove:
            repo_id = item["repo_id"]
            revision = item["revision"]
            size_human = item["size_human"]
            cache_path = item["cache_path"]  # Get the specific snapshot path

            if dry_run:
                self.logger.info(
                    "dry_run_would_remove",
                    repo_id=repo_id,
                    revision=revision,
                    size=size_human,
                    path=cache_path,
                )
                removed_items_details.append(item)  # Add to list for dry run report
                freed_bytes += item["size_bytes"]  # Accumulate potential freed space
                removed_count += 1
            else:
                # Actually remove the directory
                if os.path.exists(cache_path):
                    try:
                        self.logger.info(
                            "removing_snapshot",
                            repo_id=repo_id,
                            revision=revision,
                            size=size_human,
                            path=cache_path,
                        )
                        shutil.rmtree(cache_path)
                        # Verify removal?
                        if not os.path.exists(cache_path):
                            freed_bytes += item["size_bytes"]
                            removed_count += 1
                            removed_items_details.append(
                                item
                            )  # Add successfully removed item
                            self.logger.debug("removal_successful", path=cache_path)
                        else:
                            self.logger.warning(
                                "removal_failed_dir_still_exists", path=cache_path
                            )

                    except Exception as e:
                        self.logger.error(
                            "failed_to_remove_snapshot",
                            repo_id=repo_id,
                            revision=revision,
                            path=cache_path,
                            error=str(e),
                        )
                else:
                    self.logger.warning(
                        "snapshot_path_not_found_for_removal", path=cache_path
                    )

        self.logger.info(
            f"{action}_completed",
            items_processed=removed_count,  # Use actual count for clean, potential for dry-run
            space_affected=humanize.naturalsize(freed_bytes),
        )

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
            # Iterate through top-level directories (categories)
            for category_name in os.listdir(structured_root):
                category_path = os.path.join(structured_root, category_name)
                if category_name.startswith(".") or not os.path.isdir(category_path):
                    continue  # Skip hidden items and files

                category_data = {"size_bytes": 0, "org_count": 0, "organizations": {}}

                # Iterate through organizations/namespaces
                for org_name in os.listdir(category_path):
                    org_path = os.path.join(category_path, org_name)
                    if not os.path.isdir(org_path):
                        continue  # Skip files

                    org_data = {"size_bytes": 0, "repo_count": 0, "repos": []}

                    # Iterate through repositories
                    for repo_name in os.listdir(org_path):
                        repo_path = os.path.join(org_path, repo_name)
                        if not os.path.isdir(repo_path):
                            continue  # Skip files

                        repo_size = 0
                        symlink_count = 0
                        file_count = 0

                        # Walk through the repo directory to calculate size and count symlinks
                        try:
                            for dirpath, dirnames, filenames in os.walk(repo_path):
                                for filename in filenames:
                                    filepath = os.path.join(dirpath, filename)
                                    if os.path.islink(filepath):
                                        symlink_count += 1
                                        # Optionally resolve link to check cache size? No, keep overview simple.
                                    elif os.path.isfile(filepath):
                                        try:
                                            repo_size += os.path.getsize(filepath)
                                            file_count += 1
                                        except OSError:
                                            self.logger.warning(
                                                "could_not_get_size", file=filepath
                                            )

                            org_data["repos"].append(
                                {
                                    "name": repo_name,
                                    "size_bytes": repo_size,
                                    "size_human": humanize.naturalsize(repo_size),
                                    "symlink_count": symlink_count,
                                    "file_count": file_count,
                                    "path": os.path.relpath(repo_path, structured_root),
                                }
                            )
                            org_data["size_bytes"] += repo_size

                        except OSError as walk_err:
                            self.logger.warning(
                                "error_walking_repo_dir",
                                path=repo_path,
                                error=str(walk_err),
                            )

                    if org_data["repos"]:
                        org_data["repo_count"] = len(org_data["repos"])
                        org_data["size_human"] = humanize.naturalsize(
                            org_data["size_bytes"]
                        )
                        org_data["repos"].sort(
                            key=lambda x: x["size_bytes"], reverse=True
                        )  # Sort repos by size
                        category_data["organizations"][org_name] = org_data
                        category_data["size_bytes"] += org_data["size_bytes"]

                if category_data["organizations"]:
                    category_data["org_count"] = len(category_data["organizations"])
                    category_data["size_human"] = humanize.naturalsize(
                        category_data["size_bytes"]
                    )
                    # Sort organizations by size within category?
                    # Convert dict to list for sorting if needed later
                    overview_data["categories"][category_name] = category_data
                    overview_data["total_size"] += category_data["size_bytes"]

            overview_data["total_size_human"] = humanize.naturalsize(
                overview_data["total_size"]
            )
            self.logger.info(
                "organization_overview_complete",
                total_size=overview_data["total_size_human"],
            )
            return overview_data

        except Exception as e:
            self.logger.error(
                "organization_overview_failed", path=structured_root, error=str(e)
            )
            # Return partial data if available? Or empty? Let's return empty on error.
            return {"total_size": 0, "total_size_human": "0 B", "categories": {}}


# =============================================================================
# Command Line Interface Section
# =============================================================================


def _create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="HfHubOrganizer: Manage HuggingFace Hub downloads, cache, and organization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show defaults
    )
    parser.add_argument(
        "--profile",
        help="Profile name to use (defined in config). Overrides default behavior.",
    )
    parser.add_argument(
        "--log-format",
        choices=["console", "json", "structured"],
        default="console",
        help="Logging output format.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG level) logging.",
    )
    parser.add_argument(
        "--config",
        default="~/.config/hf_organizer/config.json",
        help="Path to the configuration file.",
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    # --- Download Command ---
    download_parser = subparsers.add_parser(
        "download", help="Download a full repository or specific file."
    )
    download_parser.add_argument(
        "repo_id", help="Repository ID (e.g., 'google/flan-t5-base', 'gpt2')"
    )
    download_parser.add_argument(
        "--filename", "-f", help="Specific file to download within the repo."
    )
    download_parser.add_argument(
        "--subfolder",
        "-s",
        help="Subfolder within the repository to download from/into.",
    )
    download_parser.add_argument(
        "--revision", "-r", help="Git revision (branch, tag, commit hash) to download."
    )
    download_parser.add_argument(
        "--category",
        choices=["models", "datasets", "spaces"],
        help="Manually specify category for organization (overrides auto-detection).",
    )
    download_parser.add_argument(
        "--base-path", help="Override HF_HOME (cache location) for this command."
    )
    download_parser.add_argument(
        "--out-dir",
        help="Override structured organization root directory for this command.",
    )
    download_parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files from cache instead of symlinking.",
    )
    # Add allow/ignore patterns?
    download_parser.add_argument(
        "--allow-patterns", nargs="*", help="Glob patterns to include (snapshot only)."
    )
    download_parser.add_argument(
        "--ignore-patterns", nargs="*", help="Glob patterns to exclude (snapshot only)."
    )

    # --- Download Recent Command ---
    download_recent_parser = subparsers.add_parser(
        "download-recent", help="Download only files modified recently."
    )
    download_recent_parser.add_argument(
        "repo_id", help="Repository ID (e.g., 'google/flan-t5-base')"
    )
    download_recent_parser.add_argument(
        "--days",
        "-d",
        type=int,
        required=True,
        help="Download files modified within the last N days.",
    )
    download_recent_parser.add_argument(
        "--subfolder", "-s", help="Only consider files within this subfolder."
    )
    download_recent_parser.add_argument(
        "--revision", "-r", help="Git revision (branch, tag, commit hash) to check."
    )
    download_recent_parser.add_argument(
        "--category",
        choices=["models", "datasets", "spaces"],
        help="Manually specify category for organization.",
    )
    download_recent_parser.add_argument(
        "--base-path", help="Override HF_HOME (cache location)."
    )
    download_recent_parser.add_argument(
        "--out-dir", help="Override structured organization root directory."
    )
    download_recent_parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of symlinking."
    )
    # --- ADDED ARGUMENT ---
    download_recent_parser.add_argument(
        "--exclude-repo-pattern",
        help="Skip download if this case-insensitive text is found in the repo_id.",
    )
    # --- END ADDED ARGUMENT ---

    # --- Profile Management Command ---
    profile_parser = subparsers.add_parser(
        "profile", help="Manage configuration profiles."
    )
    profile_subparsers = profile_parser.add_subparsers(
        dest="profile_command", required=True, help="Profile action"
    )

    # List profiles
    profile_subparsers.add_parser("list", help="List available profiles.")

    # Add/Update profile
    add_parser = profile_subparsers.add_parser("add", help="Add or update a profile.")
    add_parser.add_argument("name", help="Profile name.")
    add_parser.add_argument(
        "--base-path", help="Base path for HF cache (HF_HOME). Use '~' for home dir."
    )
    add_parser.add_argument(
        "--out-dir",
        help="Directory for organized files (structured_root). Use '~' for home dir.",
    )
    add_parser.add_argument(
        "--token", help="HuggingFace API token (optional, stored in config)."
    )
    add_parser.add_argument("--description", help="Short description for the profile.")

    # Remove profile
    remove_parser = profile_subparsers.add_parser("remove", help="Remove a profile.")
    remove_parser.add_argument("name", help="Profile name to remove.")

    # --- Cache Management Command ---
    cache_parser = subparsers.add_parser(
        "cache", help="Manage the HuggingFace Hub cache."
    )
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_command", required=True, help="Cache action"
    )

    # Scan cache
    scan_parser = cache_subparsers.add_parser(
        "scan", help="Scan and analyze cache usage."
    )
    scan_parser.add_argument(
        "--json", action="store_true", help="Output results as JSON."
    )

    # Clean cache
    clean_parser = cache_subparsers.add_parser(
        "clean", help="Clean up cached snapshots."
    )
    clean_parser.add_argument(
        "--older-than",
        type=int,
        metavar="DAYS",
        help="Remove snapshots older than N days.",
    )
    clean_parser.add_argument(
        "--min-size",
        type=int,
        metavar="MB",
        help="Remove snapshots larger than N megabytes.",
    )
    clean_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting.",
    )
    clean_parser.add_argument(
        "--json",
        action="store_true",
        help="Output removed items list as JSON (useful for dry-run).",
    )

    # --- List Downloads Command ---
    list_downloads_parser = subparsers.add_parser(
        "list", help="List download history recorded by this tool."
    )
    list_downloads_parser.add_argument(
        "--limit", type=int, default=20, help="Limit number of results."
    )
    list_downloads_parser.add_argument(
        "--category",
        choices=["models", "datasets", "spaces"],
        help="Filter by category.",
    )
    # Add filter by profile? The list is per-profile based on metadata location.
    # list_downloads_parser.add_argument("--filter-profile", help="Show history only for a specific profile name.")
    list_downloads_parser.add_argument(
        "--json", action="store_true", help="Output as JSON."
    )

    # --- Overview Command ---
    overview_parser = subparsers.add_parser(
        "overview", help="Show overview of the organized files directory."
    )
    overview_parser.add_argument("--json", action="store_true", help="Output as JSON.")

    return parser


def main():
    """Main entry point for the CLI."""
    parser = _create_parser()
    args = parser.parse_args()

    # --- Initialize Organizer (most commands need it) ---
    # We initialize it early to handle profile loading based on args.profile
    # For profile commands, some args might be None initially.
    try:
        # Handle potential overrides from command line for specific commands
        base_path_override = getattr(args, "base_path", None)
        out_dir_override = getattr(args, "out_dir", None)

        organizer = HfHubOrganizer(
            profile=args.profile,
            base_path=base_path_override,  # Pass overrides
            structured_root=out_dir_override,  # Pass overrides
            verbose=args.verbose,
            config_path=args.config,
            log_format=args.log_format,
        )
    except ValueError as e:
        # Handle profile not found error during init
        print(f"Error initializing organizer: {e}")
        # Use a basic logger if organizer setup failed
        structlog.get_logger("hfget_cli").error("init_failed", error=str(e))
        exit(1)
    except Exception as e:
        print(f"Unexpected error during initialization: {e}")
        structlog.get_logger("hfget_cli").error(
            "unexpected_init_failed", error=str(e), error_type=type(e).__name__
        )
        exit(1)

    # --- Execute Command ---
    try:
        if args.command == "profile":
            if args.profile_command == "list":
                profiles = organizer.list_profiles()
                if profiles:
                    print("Available profiles:")
                    headers = [
                        "Name",
                        "Description",
                        "Cache Path (HF_HOME)",
                        "Organized Root",
                    ]
                    table = []
                    for name in profiles:
                        p_config = organizer.config["profiles"].get(name, {})
                        desc = p_config.get("description", "N/A")
                        bp = p_config.get(
                            "base_path", f"Default ({organizer.ENV_VARS['HF_HOME']})"
                        )
                        sr = p_config.get(
                            "structured_root",
                            f"Default ({os.path.expanduser('~/huggingface_organized')})",
                        )
                        table.append([name, desc, bp, sr])
                    print(tabulate(table, headers=headers))
                else:
                    print("No profiles configured. Use 'profile add' to create one.")
                    print(f"Config file location: {organizer.config_path}")

            elif args.profile_command == "add":
                organizer.add_profile(
                    name=args.name,
                    base_path=args.base_path,
                    structured_root=args.out_dir,
                    token=args.token,
                    description=args.description,
                )
                print(f"Profile '{args.name}' added/updated successfully.")
                print(f"Config file: {organizer.config_path}")

            elif args.profile_command == "remove":
                organizer.remove_profile(args.name)
                # Confirmation message handled by logger inside method

        elif args.command == "download":
            # Prepare kwargs for snapshot_download/hf_hub_download
            dl_kwargs = {}
            if args.allow_patterns:
                dl_kwargs["allow_patterns"] = args.allow_patterns
            if args.ignore_patterns:
                dl_kwargs["ignore_patterns"] = args.ignore_patterns

            path = organizer.download(
                repo_id=args.repo_id,
                filename=args.filename,
                subfolder=args.subfolder,
                revision=args.revision,
                category=args.category,
                symlink_to_cache=not args.copy,
                **dl_kwargs,
            )
            print(f"\nDownload complete. Organized at: {path}")

        elif args.command == "download-recent":
            # --- ADDED CHECK ---
            if args.exclude_repo_pattern and re.search(
                args.exclude_repo_pattern, args.repo_id, re.IGNORECASE
            ):
                print(
                    f"Skipping repository '{args.repo_id}' because it matches exclusion pattern '{args.exclude_repo_pattern}'."
                )
                organizer.logger.info(
                    "repo_skipped_exclusion",
                    repo_id=args.repo_id,
                    pattern=args.exclude_repo_pattern,
                )
                # Exit gracefully or just return depending on desired behavior when excluded
                return  # Exit the main function, effectively skipping the download
            # --- END ADDED CHECK ---

            path = organizer.download_recent(
                repo_id=args.repo_id,
                days_ago=args.days,
                subfolder=args.subfolder,
                revision=args.revision,
                category=args.category,
                symlink_to_cache=not args.copy,
            )
            print(f"\nRecent file download process complete. Target directory: {path}")

        elif args.command == "cache":
            if args.cache_command == "scan":
                cache_stats = organizer.get_cache_stats()
                if args.json:
                    print(json.dumps(cache_stats, indent=2))
                else:
                    print(
                        f"HF Cache Statistics (Profile: {args.profile or 'Default'}, Path: {organizer.effective_paths['HF_HUB_CACHE']}):"
                    )
                    print(f"-----------------------------------------------------")
                    print(f"Total Size:       {cache_stats['total_size_human']}")
                    print(f"Unique Repos:     {cache_stats['repo_count']}")
                    print(f"Total Snapshots:  {cache_stats['snapshot_count']}")
                    print(f"Total Files:      {cache_stats['file_count']}")
                    print()

                    if cache_stats["largest_snapshots"]:
                        print("Largest snapshots (by revision):")
                        headers = ["Repo ID", "Revision (Snapshot)", "Size"]
                        table = [
                            [s["repo_id"], s["revision"][:12], s["size_human"]]
                            for s in cache_stats["largest_snapshots"]
                        ]
                        print(tabulate(table, headers=headers))
                        print()
                    else:
                        print("No snapshots found in cache scan.")

                    if cache_stats["organizations"]:
                        print("Storage by Organization:")
                        headers = ["Organization", "Size", "Snapshots", "% of Total"]
                        table = []
                        for org in cache_stats["organizations"]:
                            table.append(
                                [
                                    org["name"],
                                    org["size_human"],
                                    org["snapshot_count"],
                                    f"{org['percentage']:.1f}%",
                                ]
                            )
                        print(tabulate(table, headers=headers))

            elif args.cache_command == "clean":
                removed_count, freed_bytes, removed_details = organizer.clean_cache(
                    older_than_days=args.older_than,
                    min_size_mb=args.min_size,
                    dry_run=args.dry_run,
                )
                action_verb = "Would remove" if args.dry_run else "Removed"
                print(
                    f"{action_verb} {removed_count} snapshots, freeing {humanize.naturalsize(freed_bytes)}."
                )

                if args.json:
                    print(json.dumps(removed_details, indent=2))
                elif removed_details:
                    print(f"\n{action_verb.capitalize()} snapshots:")
                    headers = ["Repo ID", "Revision", "Size", "Last Modified"]
                    table = [
                        [
                            i["repo_id"],
                            i["revision"][:12],
                            i["size_human"],
                            i["last_modified"],
                        ]
                        for i in removed_details
                    ]
                    print(tabulate(table, headers=headers))


        elif args.command == "list":
            downloads = organizer.list_downloads(
                limit=args.limit,
                category=args.category,
                # profile_filter=args.filter_profile # Add if implementing cross-profile listing
            )

            if args.json:
                print(json.dumps(downloads, indent=2))
            else:
                if not downloads:
                    print(
                        f"No download history found for profile '{args.profile or 'Default'}'."
                    )
                    print(
                        f"(Metadata file: {os.path.join(organizer.structured_root, '.metadata', 'downloads.json')})"
                    )

                else:
                    print(
                        f"Download History (Profile: {args.profile or 'Default'}, Max: {args.limit}):"
                    )
                    headers = [
                        "Timestamp",
                        "Repo ID",
                        "Type",
                        "Category",
                        "Profile",
                        "Subfolder",
                        "Rel Path",
                    ]
                    table = []
                    for item in downloads:
                        ts = "N/A"
                        try:
                            # Parse and format timestamp robustly
                            dt_obj = datetime.datetime.fromisoformat(
                                item.get("timestamp", "")
                            ).astimezone(None)  # Convert to local
                            ts = dt_obj.strftime("%Y-%m-%d %H:%M")
                        except (ValueError, TypeError):
                            ts = item.get("timestamp", "Invalid Date")[:16]  # Fallback

                        table.append(
                            [
                                ts,
                                item.get("repo_id", "N/A"),
                                item.get("type", "N/A"),
                                item.get("category", "N/A"),
                                item.get("profile", "N/A"),
                                item.get("subfolder", "-"),
                                item.get("relative_path", "N/A"),
                            ]
                        )
                    print(
                        tabulate(
                            table,
                            headers=headers,
                            maxcolwidths=[None, None, 15, None, None, 10, 25],
                        )
                    )  # Adjust widths

        elif args.command == "overview":
            overview = organizer.get_organization_overview()
            if args.json:
                print(json.dumps(overview, indent=2))
            else:
                print(
                    f"Organization Overview (Profile: {args.profile or 'Default'}, Root: {organizer.effective_paths['structured_root']})"
                )
                print(f"-----------------------------------------------------")
                print(
                    f"Total Organized Size: {overview.get('total_size_human', '0 B')}"
                )
                print()

                categories = overview.get("categories", {})
                if not categories:
                    print("No categories found in the organized directory.")
                else:
                    # Sort categories alphabetically for consistent output
                    sorted_categories = sorted(categories.items())

                    for category, cat_data in sorted_categories:
                        print(
                            f"--- Category: {category} ({cat_data.get('size_human', '0 B')}) ---"
                        )

                        orgs_data = cat_data.get("organizations", {})
                        if not orgs_data:
                            print("  No organizations found in this category.")
                            print()
                            continue

                        # Sort organizations by size within the category
                        sorted_orgs = sorted(
                            orgs_data.items(),
                            key=lambda item: item[1].get("size_bytes", 0),
                            reverse=True,
                        )

                        headers = ["Organization", "Size", "Repositories"]
                        table = []
                        for org_name, org_info in sorted_orgs:
                            table.append(
                                [
                                    org_name,
                                    org_info.get("size_human", "0 B"),
                                    org_info.get("repo_count", 0),
                                ]
                            )
                        print(
                            tabulate(table, headers=headers, tablefmt="plain")
                        )  # Use plain for less clutter
                        print()


    except RepositoryNotFoundError as e:
        # Handle specific errors gracefully - This should now work
        organizer.logger.error(
            "command_failed_repo_not_found",
            repo_id=getattr(args, "repo_id", "N/A"),
            error=str(e),
        )
        print(f"Error: Repository not found: {getattr(args, 'repo_id', 'N/A')}")
        exit(1)
    except Exception as e:
        # Log the full traceback for unexpected errors
        organizer.logger.exception(
            "command_failed_unexpected", command=args.command, error=str(e)
        )
        print(f"\nAn unexpected error occurred: {e}")
        print("Check logs or run with --verbose for more details.")
        exit(1)


if __name__ == "__main__":
    main()
