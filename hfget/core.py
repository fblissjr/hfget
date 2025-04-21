# hfget/core.py
import os
import shutil
import json
import time
import datetime
import logging
import re
import fnmatch
from pathlib import Path
from typing import Dict, Optional, Union, List, Any, Tuple, Literal

import structlog
import humanize

from huggingface_hub import (
    HfApi,
    snapshot_download,
    hf_hub_download,
    list_repo_files,
    scan_cache_dir,
    # Note: We use scan_cache_dir().delete_revisions() for cleaning
    # Direct import of HFCacheInfo might be useful for type hints if needed later
    # from huggingface_hub import HFCacheInfo, CommitInfo
)
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    HfHubHTTPError,
    hf_raise_for_status,  # Good practice for checking responses if making raw calls
    GatedRepoError,
    RevisionNotFoundError,
    EntryNotFoundError,
)
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE  # Get default cache path


# --- Configuration Constants ---
# Using Path objects for better path manipulation
CONFIG_DIR = Path("~/.config/hf_organizer").expanduser()
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_STRUCTURED_ROOT = Path("~/huggingface_organized").expanduser()
METADATA_FILENAME = "downloads.json"
METADATA_DIR_NAME = ".metadata"
# Use the constant from huggingface_hub if possible, otherwise fallback
FALLBACK_HF_HOME = (
    Path(HUGGINGFACE_HUB_CACHE).parent
    if HUGGINGFACE_HUB_CACHE
    else Path("~/.cache/huggingface").expanduser()
)

# Mapping from HF repo types to directory names in the organized structure
REPO_TYPE_TO_DIR_NAME: Dict[Literal["model", "dataset", "space"], str] = {
    "model": "models",
    "dataset": "datasets",
    "space": "spaces",
}

# Environment variables this tool specifically influences or reads beyond standard HF ones
HFGET_ENV_OVERRIDES = {"HF_HOME", "HF_TOKEN", "HF_HUB_ENABLE_HF_TRANSFER"}

class HfHubOrganizer:
    """
    Core class to manage HuggingFace Hub downloads, cache, and organization.

    Provides functionality to:
    - Manage configuration profiles (cache location, organized storage root, token).
    - Download models, datasets, or spaces into a structured directory.
    - Choose between symlinking to the cache or copying files.
    - Download only files matching certain patterns within a repository.
    - Scan and clean the Hugging Face cache based on age or size.
    - Track download history performed by this tool.
    - Provide an overview of the organized storage directory.
    """

    # Default values for boolean env vars used if not set elsewhere
    # These align with huggingface_hub's typical desired defaults for performance/convenience
    DEFAULT_BOOLEAN_ENV_VARS = {
        "HF_HUB_ENABLE_HF_TRANSFER": True,
        # Add others if needed, e.g., HF_HUB_DISABLE_TELEMETRY: True
    }

    def __init__(
        self,
        profile: Optional[str] = None,
        base_path_override: Optional[str] = None,  # HF_HOME override
        structured_root_override: Optional[str] = None,
        token_override: Optional[str] = None,
        enable_hf_transfer_override: Optional[bool] = None,
        verbose: bool = False,
        config_path: Optional[str] = None,
        log_format: str = "console",
    ):
        """
        Initializes the HfHubOrganizer.

        Sets up logging, loads configuration, determines the active profile,
        resolves effective paths and settings (considering overrides, profile,
        environment variables, and defaults), and initializes the HfApi client.

        Args:
            profile: Name of the profile to use from the config file.
            base_path_override: Explicitly set the HF_HOME path, overriding profile and env vars.
            structured_root_override: Explicitly set the organized output directory root.
            token_override: Explicitly set the HF token to use.
            enable_hf_transfer_override: Explicitly enable/disable hf_transfer.
            verbose: Enable DEBUG level logging if True.
            config_path: Path to the configuration JSON file.
            log_format: Logging format ('console', 'json', 'structured').
        """
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser()
        self.config_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure config dir exists
        self.logger = self._setup_logger(verbose, log_format)

        try:
            self.config = self._load_config()
        except Exception as e:
            self.logger.error(
                "config_load_failed_init", error=str(e), path=str(self.config_path)
            )
            self.config = {"profiles": {}}  # Start with empty config on load failure

        self.selected_profile = profile
        profile_settings = self._get_profile_settings(profile)

        # Resolve effective settings using a priority order:
        # 1. Explicit override arguments (e.g., base_path_override)
        # 2. Environment variables (e.g., HF_HOME)
        # 3. Profile settings from config file
        # 4. Default values
        self.effective_hf_home = self._resolve_config_value(
            override=base_path_override,
            env_var="HF_HOME",
            profile_setting=profile_settings.get("base_path"),
            default=str(FALLBACK_HF_HOME),  # Convert Path to str for consistency here
            is_path=True,
        )
        self.structured_root = self._resolve_config_value(
            override=structured_root_override,
            env_var=None,  # Not typically an env var
            profile_setting=profile_settings.get("structured_root"),
            default=str(DEFAULT_STRUCTURED_ROOT),
            is_path=True,
        )
        self.effective_token = self._resolve_config_value(
            override=token_override,
            env_var="HF_TOKEN",
            profile_setting=profile_settings.get("token"),
            default=None,
        )
        self.effective_hf_transfer = self._resolve_config_value(
            override=enable_hf_transfer_override,
            env_var="HF_HUB_ENABLE_HF_TRANSFER",
            profile_setting=profile_settings.get("enable_hf_transfer"),
            default=self.DEFAULT_BOOLEAN_ENV_VARS["HF_HUB_ENABLE_HF_TRANSFER"],
            is_bool=True,
        )

        # Set environment variables IF THEY ARE NOT ALREADY SET.
        # This allows users to override profile settings with environment variables.
        # huggingface_hub library reads these env vars internally.
        self._initialize_env_vars()

        # Determine the final cache path (HF_HUB_CACHE defaults to HF_HOME/hub)
        hf_hub_cache_default = Path(self.effective_hf_home) / "hub"
        self.effective_cache_path = Path(
            os.environ.get("HF_HUB_CACHE", str(hf_hub_cache_default))
        )

        # Store resolved paths for easy access and logging
        self.effective_paths = {
            "HF_HOME": str(
                Path(self.effective_hf_home).expanduser()
            ),  # Ensure paths are expanded and strings
            "HF_HUB_CACHE": str(self.effective_cache_path.expanduser()),
            "structured_root": str(Path(self.structured_root).expanduser()),
        }

        # Bind final resolved paths and settings to the logger context
        self.logger = self.logger.bind(
            profile=profile or "Default",
            hf_home=self.effective_paths["HF_HOME"],
            cache=self.effective_paths["HF_HUB_CACHE"],
            org_root=self.effective_paths["structured_root"],
            hf_transfer=self.effective_hf_transfer,
            token_source="explicit"
            if token_override
            else (
                "profile"
                if profile_settings.get("token")
                else ("env" if os.environ.get("HF_TOKEN") else "none")
            ),
        )

        # Initialize the HfApi client *after* setting potential token in env
        try:
            self.api = HfApi(
                token=self.effective_token
            )  # Pass token explicitly for clarity
            # Verify token works if provided (optional but good practice)
            if self.effective_token:
                try:
                    self.api.whoami()
                    self.logger.debug("hf_token_verified")
                except HfHubHTTPError as e:
                    if e.response.status_code == 401:
                        self.logger.warning(
                            "hf_token_invalid",
                            error=str(e),
                            token_source=self.logger._context.get("token_source"),
                        )
                    else:
                        self.logger.warning("hf_api_whoami_failed", error=str(e))

        except Exception as e:
            self.logger.error("hf_api_init_failed", error=str(e))
            # Depending on requirements, might want to raise or handle differently
            raise RuntimeError(f"Failed to initialize HfApi client: {e}") from e

        self.logger.info("organizer_initialized")

    def _resolve_config_value(
        self,
        override: Optional[Any],
        env_var: Optional[str],
        profile_setting: Optional[Any],
        default: Optional[Any],
        is_path: bool = False,
        is_bool: bool = False,
    ) -> Any:
        """Helper to determine the effective value based on priority."""
        value = default

        # 1. Profile Setting (lowest priority besides default)
        if profile_setting is not None:
            value = profile_setting
            source = "profile"
        else:
            source = "default"

        # 2. Environment Variable
        if env_var and env_var in os.environ:
            env_val = os.environ[env_var]
            if is_bool:
                value = env_val.lower() in ["true", "1", "yes"]
            else:
                value = env_val
            source = "env"

        # 3. Explicit Override Argument (highest priority)
        if override is not None:
            value = override
            source = "override"

        # Post-processing for specific types
        if is_path and isinstance(value, str):
            value = str(Path(value).expanduser())  # Return as string after expanding
        elif is_bool:
            # Ensure final value is a boolean, handling potential string values from profile/env
            if isinstance(value, str):
                value = value.lower() in ["true", "1", "yes"]
            else:
                value = bool(value)  # Handle None or other types safely

        # Log the source only if it wasn't the default
        if source != "default":
            log_key = env_var or "setting"
            self.logger.debug(
                f"resolved_{log_key}",
                value=value if env_var != "HF_TOKEN" else "****",
                source=source,
            )  # Mask token

        return value

    def _get_profile_settings(self, profile_name: Optional[str]) -> Dict[str, Any]:
        """Safely retrieves settings for a given profile name."""
        if profile_name:
            if profile_name not in self.config.get("profiles", {}):
                self.logger.error(
                    "profile_not_found",
                    profile=profile_name,
                    available=list(self.config.get("profiles", {}).keys()),
                )
                raise ValueError(
                    f"Profile '{profile_name}' not found in config file {self.config_path}."
                )
            return self.config["profiles"][profile_name]
        return {}

    def _initialize_env_vars(self):
        """
        Set environment variables based on resolved effective values *if not already set*.
        This allows environment variables set outside the tool to take precedence.
        """
        # HF_HOME
        if "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = self.effective_hf_home
            self.logger.debug(
                "env_var_set",
                key="HF_HOME",
                value=self.effective_hf_home,
                source="resolved_config",
            )
        elif os.environ["HF_HOME"] != str(Path(self.effective_hf_home).expanduser()):
            self.logger.warning(
                "env_var_mismatch",
                key="HF_HOME",
                env_value=os.environ["HF_HOME"],
                resolved_value=self.effective_hf_home,
                using="env_value",
            )
            self.effective_hf_home = os.environ[
                "HF_HOME"
            ]  # Update internal state to match env

        # HF_TOKEN
        if "HF_TOKEN" not in os.environ and self.effective_token:
            os.environ["HF_TOKEN"] = self.effective_token
            self.logger.debug(
                "env_var_set", key="HF_TOKEN", value="****", source="resolved_config"
            )
        elif os.environ.get("HF_TOKEN") != self.effective_token:
            self.logger.debug(
                "env_var_mismatch", key="HF_TOKEN", source="env_value_takes_precedence"
            )
            # Don't update self.effective_token here, as it was already resolved with env precedence

        # HF_HUB_ENABLE_HF_TRANSFER
        env_key = "HF_HUB_ENABLE_HF_TRANSFER"
        current_env_val = os.environ.get(env_key)
        target_val_str = "1" if self.effective_hf_transfer else "0"
        if current_env_val is None:
            os.environ[env_key] = target_val_str
            self.logger.debug(
                "env_var_set",
                key=env_key,
                value=target_val_str,
                source="resolved_config",
            )
        elif current_env_val != target_val_str:
            self.logger.warning(
                "env_var_mismatch",
                key=env_key,
                env_value=current_env_val,
                resolved_value=target_val_str,
                using="env_value",
            )
            # Update internal state to match env if it differs
            self.effective_hf_transfer = current_env_val == "1"

    def _setup_logger(self, verbose: bool, format_type: str) -> structlog.BoundLogger:
        """Configures structlog based on verbosity and format."""
        log_level = logging.DEBUG if verbose else logging.INFO

        # Define shared processors
        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(
                fmt="iso", utc=True
            ),  # Use ISO format for better parsing
        ]

        # Configure logging handler and formatter based on format_type
        # Use standard logging to avoid potential conflicts if user also uses structlog
        log_handler = logging.StreamHandler(sys.stdout)

        if format_type == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "event": "%(message)s", "logger": "%(name)s"}'
            )  # Basic JSON format
        else:  # Default to console-friendly
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            )

        log_handler.setFormatter(formatter)

        # Configure the root logger - set level and add handler if not present
        # This affects libraries using standard logging as well (like huggingface_hub)
        root_logger = logging.getLogger()
        if not root_logger.handlers:  # Add handler only if no handlers exist
            root_logger.addHandler(log_handler)
        root_logger.setLevel(log_level)

        # Configure structlog itself
        structlog.configure(
            processors=shared_processors
            + [
                # Processor needed to format output via standard logging
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Return a logger bound to this class instance
        return structlog.get_logger(self.__class__.__name__)


    def _load_config(self) -> Dict[str, Any]:
        """Loads the JSON configuration file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                self.logger.debug("config_loaded", path=str(self.config_path))
                # Basic validation
                if not isinstance(config, dict):
                    raise ValueError("Config root must be a dictionary.")
                if "profiles" not in config:
                    config["profiles"] = {}
                    self.logger.debug(
                        "added_missing_profiles_key", path=str(self.config_path)
                    )
                if not isinstance(config.get("profiles"), dict):
                    self.logger.warning(
                        "invalid_profiles_section_resetting", path=str(self.config_path)
                    )
                    config["profiles"] = {}
                return config
            except json.JSONDecodeError as e:
                self.logger.error(
                    "config_load_failed_invalid_json",
                    path=str(self.config_path),
                    error=str(e),
                )
                raise  # Re-raise critical error
            except Exception as e:
                self.logger.error(
                    "config_load_failed_unexpected",
                    path=str(self.config_path),
                    error=str(e),
                    action="using_empty_config",
                )
                # Fallback to default empty config on unexpected error
                return {"profiles": {}}
        else:
            self.logger.debug(
                "config_file_not_found",
                path=str(self.config_path),
                action="using_empty_config",
            )
            return {"profiles": {}}

    def _save_config(self):
        """Saves the current configuration to the JSON file."""
        try:
            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2, sort_keys=True)
            self.logger.debug("config_saved", path=str(self.config_path))
        except Exception as e:
            # Log error but don't crash the application
            self.logger.error(
                "config_save_failed", path=str(self.config_path), error=str(e)
            )

    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Returns the dictionary of profiles from the configuration."""
        return self.config.get("profiles", {})

    def add_profile(
        self,
        name: str,
        base_path: Optional[str] = None,
        structured_root: Optional[str] = None,
        token: Optional[str] = None,
        enable_hf_transfer: Optional[bool] = None,
        description: Optional[str] = None,
    ):
        """Adds or updates a profile in the configuration."""
        if not name:
            raise ValueError("Profile name cannot be empty.")

        # Initialize 'profiles' key if it doesn't exist
        if "profiles" not in self.config:
            self.config["profiles"] = {}

        # Construct profile data, only including non-None values
        profile_data = {}
        if base_path:
            profile_data["base_path"] = str(Path(base_path).expanduser())
        if structured_root:
            profile_data["structured_root"] = str(Path(structured_root).expanduser())
        if token:
            # Basic check for token format (optional, could be stricter)
            if not isinstance(token, str) or not token.startswith("hf_"):
                self.logger.warning("invalid_token_format_add_profile", profile=name)
            profile_data["token"] = token  # Store token as is
        if (
            enable_hf_transfer is not None
        ):  # Explicitly check for None allows storing 'False'
            profile_data["enable_hf_transfer"] = bool(enable_hf_transfer)
        if description:
            profile_data["description"] = description
        elif (
            name not in self.config["profiles"]
        ):  # Add default description only for new profiles
            profile_data["description"] = f"Profile '{name}'"

        # If profile exists, update only specified fields
        if name in self.config["profiles"]:
            self.config["profiles"][name].update(profile_data)
            log_event = "profile_updated"
        else:
            self.config["profiles"][name] = profile_data
            log_event = "profile_added"

        self._save_config()
        self.logger.info(
            log_event, name=name, details=profile_data
        )  # Log details (mask token if needed)

    def remove_profile(self, name: str) -> bool:
        """Removes a profile from the configuration."""
        if "profiles" in self.config and name in self.config["profiles"]:
            del self.config["profiles"][name]
            self._save_config()
            self.logger.info("profile_removed", name=name)
            return True
        else:
            self.logger.warning("profile_not_found_for_removal", name=name)
            return False

    def _determine_repo_type(self, repo_id: str) -> str:
        """
        Determines the repository type (model, dataset, space) using HfApi.
        Handles errors and falls back to 'model'.
        """
        repo_type_singular = "model"  # Default fallback
        log_ctx = {"repo_id": repo_id}
        try:
            self.logger.debug("determining_repo_type_start", **log_ctx)
            # Use HfApi.repo_info to get the type
            repo_info = self.api.repo_info(
                repo_id=repo_id
            )  # repo_type is already handled by repo_info
            repo_type_from_api = getattr(repo_info, "repo_type", None)

            if repo_type_from_api in ["dataset", "space", "model"]:
                repo_type_singular = repo_type_from_api
                self.logger.debug(
                    "repo_type_detected", type=repo_type_singular, **log_ctx
                )
            elif repo_type_from_api is not None:
                # Log if API returns an unexpected type
                self.logger.warning(
                    "unrecognized_repo_type_api",
                    type=repo_type_from_api,
                    fallback=repo_type_singular,
                    **log_ctx,
                )
            else:
                # Log if repo_info doesn't contain repo_type
                self.logger.warning(
                    "repo_type_missing_api", fallback=repo_type_singular, **log_ctx
                )

        except RepositoryNotFoundError:
            # Repo doesn't exist on the Hub, assume 'model' type for path structure
            self.logger.warning(
                "repo_not_found_api_type_detection",
                fallback=repo_type_singular,
                **log_ctx,
            )
        except GatedRepoError:
            # Cannot access info due to gating, assume 'model'
            self.logger.warning(
                "gated_repo_api_type_detection", fallback=repo_type_singular, **log_ctx
            )
        except HfHubHTTPError as http_err:
            if http_err.response.status_code == 401:
                # Authentication error
                self.logger.error(
                    "authentication_error_api_type_detection",
                    error=str(http_err),
                    **log_ctx,
                )
                # Propagate error, as we can't proceed without knowing the type reliably if it might be private
                raise ValueError(
                    f"Authentication failed for {repo_id}. Cannot determine repository type. Check your HF_TOKEN."
                ) from http_err
            else:
                # Other HTTP errors
                self.logger.warning(
                    "repo_type_detection_http_error",
                    status=http_err.response.status_code,
                    error=str(http_err),
                    fallback=repo_type_singular,
                    **log_ctx,
                )
        except Exception as e:
            # Catch any other unexpected errors during API call
            self.logger.warning(
                "repo_type_detection_failed_unexpected",
                error=str(e),
                error_type=type(e).__name__,
                fallback=repo_type_singular,
                **log_ctx,
            )

        return repo_type_singular

    def _get_organized_path(
        self,
        repo_id: str,
        repo_type: str,
        subfolder: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Constructs the target path within the organized directory structure.
        Example: {structured_root}/models/google/flan-t5-base/model.safetensors
        """
        namespace = (
            "library"  # Default for repos without explicit namespace (like 'gpt2')
        )
        repo_name = repo_id
        if "/" in repo_id:
            namespace, repo_name = repo_id.split("/", 1)

        # Get the directory name for the category (e.g., 'models', 'datasets')
        repo_type_dir_name = REPO_TYPE_TO_DIR_NAME.get(
            repo_type, "models"
        )  # Fallback to models

        # Base path for the specific repo
        org_repo_base_path = (
            Path(self.effective_paths["structured_root"])
            / repo_type_dir_name
            / namespace
            / repo_name
        )

        # Add subfolder if specified
        target_path = org_repo_base_path
        if subfolder:
            target_path = target_path / subfolder

        # Add filename if specified
        if filename:
            # Handle potential edge case where filename might contain path separators
            # Use only the basename for the final part
            target_path = target_path / Path(filename).name

        self.logger.debug(
            "determined_organized_path",
            repo_id=repo_id,
            repo_type=repo_type,
            subfolder=subfolder,
            filename=filename,
            path=str(target_path),
        )
        return target_path

    def _link_or_copy(
        self,
        cache_path: Union[str, Path],
        org_path: Union[str, Path],
        symlink_to_cache: bool,
    ):
        """
        Creates a symlink or copies a file/directory from the cache path to the organized path.
        Handles existing files/links at the destination.

        Args:
            cache_path: Source path in the Hugging Face cache.
            org_path: Destination path in the organized structure.
            symlink_to_cache: If True, create a symlink; otherwise, copy.
        """
        cache_path = Path(cache_path)
        org_path = Path(org_path)
        log_ctx = {
            "source": str(cache_path),
            "target": str(org_path),
            "symlink": symlink_to_cache,
        }
        self.logger.debug("link_or_copy_start", **log_ctx)

        # Ensure the parent directory for the target path exists
        org_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Check and remove existing target if necessary ---
        # Use lexists to check for broken symlinks as well
        if org_path.lexists():
            is_link = org_path.is_symlink()
            should_remove = True
            # Optimization: If it's already the correct symlink, do nothing
            if is_link and symlink_to_cache:
                try:
                    # Resolve both paths for robust comparison
                    if org_path.resolve(strict=True) == cache_path.resolve(strict=True):
                        should_remove = False
                        self.logger.debug("target_already_correct_symlink", **log_ctx)
                except (
                    Exception
                ):  # Handles broken links or inaccessible files during resolve
                    self.logger.debug(
                        "existing_symlink_invalid_proceeding_remove", **log_ctx
                    )

            if should_remove:
                try:
                    if is_link or org_path.is_file():
                        org_path.unlink()  # Remove file or symlink
                        self.logger.debug(
                            "removed_existing_target_file_or_link", **log_ctx
                        )
                    elif org_path.is_dir():
                        shutil.rmtree(org_path)  # Remove directory tree
                        self.logger.debug("removed_existing_target_dir", **log_ctx)
                except OSError as e:
                    self.logger.error(
                        "failed_removing_existing_target", error=str(e), **log_ctx
                    )
                    raise  # Propagate the error as we can't proceed

        # --- Perform the link or copy operation ---
        try:
            if symlink_to_cache:
                # Create a relative symlink if possible for portability, otherwise absolute
                try:
                    # Calculate relative path from org_path's parent to cache_path
                    rel_cache_path = os.path.relpath(cache_path, start=org_path.parent)
                    os.symlink(rel_cache_path, org_path)
                    self.logger.debug(
                        "symlink_created_relative",
                        relative_source=rel_cache_path,
                        **log_ctx,
                    )
                except ValueError:  # Paths are on different drives on Windows
                    abs_cache_path = (
                        cache_path.resolve()
                    )  # Use absolute path as fallback
                    os.symlink(abs_cache_path, org_path)
                    self.logger.debug(
                        "symlink_created_absolute",
                        absolute_source=str(abs_cache_path),
                        **log_ctx,
                    )
            else:
                # Copying files or directories
                if cache_path.is_dir():
                    # copytree copies contents into org_path if it exists, so ensure org_path doesn't exist first
                    # (We already removed it above if necessary)
                    shutil.copytree(
                        cache_path, org_path, symlinks=True
                    )  # symlinks=True preserves internal links if copying a snapshot dir
                    self.logger.debug("directory_copied", **log_ctx)
                elif cache_path.is_file():
                    shutil.copy2(cache_path, org_path)  # copy2 preserves metadata
                    self.logger.debug("file_copied", **log_ctx)
                else:
                    # This case should ideally not happen if hf_hub_download worked
                    self.logger.warning(
                        "copy_source_not_found_or_not_file_or_dir", **log_ctx
                    )

        except OSError as e:
            # Catch symlink creation errors (e.g., permissions on Windows without dev mode)
            if symlink_to_cache and "symbolic link" in str(e).lower():
                self.logger.error(
                    "symlink_failed_os_error",
                    error=str(e),
                    hint="On Windows, symlinks might require administrator privileges or Developer Mode.",
                    **log_ctx,
                )
            else:
                self.logger.error(
                    "link_or_copy_failed_os_error", error=str(e), **log_ctx
                )
            raise  # Propagate error
        except Exception as e:
            self.logger.error("link_or_copy_failed_unexpected", error=str(e), **log_ctx)
            raise  # Propagate error

    def download(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        category: Optional[str] = None,  # User override for category
        symlink_to_cache: bool = True,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        **kwargs,  # Pass extra args like force_download, resume_download to hf_hub funcs
    ) -> str:
        """
        Downloads a repository or a specific file and organizes it.

        Determines the repository type, downloads using appropriate huggingface_hub
        functions, and then either symlinks or copies the result into the
        structured directory. Records the download operation in metadata.

        Args:
            repo_id: The repository ID (e.g., 'google/flan-t5-base').
            filename: Specific file to download. Downloads the whole repo if None.
            subfolder: Subfolder within the repository to target.
            revision: Git revision (branch, tag, commit hash).
            category: Manually specify category ('model', 'dataset', 'space'). Overrides auto-detection.
            symlink_to_cache: If True, symlink to the cache; otherwise, copy.
            allow_patterns: Glob patterns to include (snapshot download only).
            ignore_patterns: Glob patterns to exclude (snapshot download only).
            **kwargs: Additional keyword arguments passed to `hf_hub_download` or `snapshot_download`.
                      Common examples include `force_download=True`, `resume_download=False`.

        Returns:
            The absolute path to the organized file or directory.

        Raises:
            RepositoryNotFoundError: If the repo_id is not found on the Hub.
            HfHubHTTPError: For other HTTP errors from the Hub API.
            ValueError: For invalid arguments.
            OSError: For filesystem errors during linking/copying.
        """
        start_time = time.time()
        log_ctx = {
            "repo_id": repo_id,
            "target": filename or "entire_repo",
            "subfolder": subfolder,
            "revision": revision or "main",  # Log 'main' if revision is None
            "symlink": symlink_to_cache,
            **kwargs,  # Log extra args passed
        }
        self.logger.info("download_job_started", **log_ctx)

        try:
            # 1. Determine Repository Type (model, dataset, space)
            # Use user override if provided, otherwise detect from Hub
            if category and category.lower() in REPO_TYPE_TO_DIR_NAME:
                detected_category = category.lower().rstrip("s")  # Ensure singular form
                self.logger.debug(
                    "using_user_specified_category",
                    category=detected_category,
                    **log_ctx,
                )
            else:
                detected_category = self._determine_repo_type(repo_id)
            log_ctx["category"] = detected_category  # Add detected category to context
            repo_type_arg = detected_category  # Pass to hf_hub functions

            # 2. Determine Target Path in Organized Structure
            # Get the base path for the repo (e.g., .../models/google/flan-t5-base)
            org_repo_path = self._get_organized_path(repo_id, detected_category)
            # Get the specific path for the download (including subfolder/filename)
            final_organized_path = self._get_organized_path(
                repo_id, detected_category, subfolder, filename
            )

            # 3. Download using huggingface_hub
            downloaded_path_in_cache: Optional[str] = None  # Initialize
            if filename:
                # Downloading a single file
                self.logger.debug("downloading_single_file", **log_ctx)
                effective_filename = filename  # Use the provided filename
                effective_subfolder = subfolder  # Use the provided subfolder

                # If filename contains path separators, adjust filename and subfolder for hf_hub_download
                if os.path.sep in filename or ("/" in filename):
                    path_parts = Path(filename).parts
                    effective_filename = path_parts[-1]
                    relative_dir = Path(*path_parts[:-1])
                    effective_subfolder = (
                        str(Path(subfolder) / relative_dir)
                        if subfolder
                        else str(relative_dir)
                    )
                    self.logger.debug(
                        "adjusted_filename_subfolder",
                        filename=effective_filename,
                        subfolder=effective_subfolder,
                        **log_ctx,
                    )

                downloaded_path_in_cache = hf_hub_download(
                    repo_id=repo_id,
                    filename=effective_filename,  # File basename
                    subfolder=effective_subfolder,  # Adjusted subfolder
                    revision=revision,
                    repo_type=repo_type_arg,
                    token=self.api.token,  # Use resolved token
                    cache_dir=self.effective_paths[
                        "HF_HUB_CACHE"
                    ],  # Explicitly pass cache
                    **kwargs,
                )
                # final_organized_path already includes the filename correctly from _get_organized_path
                self.logger.debug(
                    "single_file_downloaded_cache",
                    path=downloaded_path_in_cache,
                    **log_ctx,
                )

            else:
                # Downloading the entire repository (or filtered snapshot)
                self.logger.debug("downloading_snapshot", allow_patterns=allow_patterns, ignore_patterns=ignore_patterns, **log_ctx)
                # Ensure target directory exists for snapshot contents
                final_organized_path.mkdir(parents=True, exist_ok=True)

                downloaded_path_in_cache = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    repo_type=repo_type_arg,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    subfolder=subfolder,  # snapshot_download also supports subfolder
                    token=self.api.token,
                    cache_dir=self.effective_paths[
                        "HF_HUB_CACHE"
                    ],  # Explicitly pass cache
                    **kwargs,
                )
                self.logger.debug(
                    "snapshot_downloaded_cache",
                    path=downloaded_path_in_cache,
                    **log_ctx,
                )

            # 4. Link or Copy from Cache to Organized Directory
            if downloaded_path_in_cache:
                if filename:
                    # Link/copy the single file
                    self._link_or_copy(
                        downloaded_path_in_cache, final_organized_path, symlink_to_cache
                    )
                else:
                    # Link/copy the contents of the snapshot directory
                    # Clear existing directory content first to avoid merging old/new files
                    if final_organized_path.exists():
                        self.logger.debug(
                            "clearing_existing_target_dir",
                            path=str(final_organized_path),
                        )
                        for item in final_organized_path.iterdir():
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()

                    item_count = 0
                    for item_in_cache in Path(downloaded_path_in_cache).iterdir():
                        # Construct target path for each item within the snapshot
                        target_item_path = final_organized_path / item_in_cache.name
                        self._link_or_copy(
                            item_in_cache, target_item_path, symlink_to_cache
                        )
                        item_count += 1
                    self.logger.debug(
                        "snapshot_contents_processed",
                        count=item_count,
                        target=str(final_organized_path),
                        **log_ctx,
                    )
            else:
                # This case should ideally not happen if downloads succeed
                self.logger.warning("download_path_in_cache_is_none", **log_ctx)
                raise RuntimeError(
                    "Download completed but cache path was not returned."
                )

            # 5. Record Metadata
            # Use the base repo path for metadata, not the specific file/subfolder path
            self._save_download_metadata(
                org_repo_path,  # Path(.../models/google/flan-t5-base)
                repo_id,
                detected_category,
                filename if filename else "snapshot",  # Record 'snapshot' if whole repo
                subfolder,
                revision,
            )

            elapsed = time.time() - start_time
            self.logger.info(
                "download_job_completed",
                organized_path=str(final_organized_path),
                elapsed_seconds=round(elapsed, 2),
                **log_ctx,
            )
            return str(final_organized_path)

        # --- Error Handling ---
        except (
            RepositoryNotFoundError,
            GatedRepoError,
            RevisionNotFoundError,
            EntryNotFoundError,
        ) as e:
            # Specific HF errors
            error_type = type(e).__name__
            self.logger.error(
                f"download_failed_{error_type.lower()}", error=str(e), **log_ctx
            )
            raise  # Re-raise the specific error
        except HfHubHTTPError as http_err:
            # General HTTP errors (e.g., 401 Unauthorized, 5xx Server Errors)
            self.logger.error(
                "download_failed_http_error",
                status=http_err.response.status_code,
                error=str(http_err),
                request_id=http_err.request_id,
                **log_ctx,
            )
            # Provide specific hint for 401 errors
            if http_err.response.status_code == 401:
                raise type(http_err)(
                    f"{http_err} - Check your authentication token (HF_TOKEN).",
                    response=http_err.response,
                ) from http_err
            raise http_err
        except ValueError as e:
            # Handle potential ValueErrors from path manipulation or invalid args
            self.logger.error("download_failed_value_error", error=str(e), **log_ctx)
            raise
        except OSError as e:
            # Filesystem errors during link/copy
            self.logger.error(
                "download_failed_os_error", error=str(e), errno=e.errno, **log_ctx
            )
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            self.logger.exception("download_failed_unexpected", error=str(e), **log_ctx)
            raise RuntimeError(
                f"An unexpected error occurred during download: {e}"
            ) from e

    def download_recent(
        self,
        repo_id: str,
        days_ago: int,  # Kept for CLI compatibility, but ignored for filtering
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        category: Optional[str] = None,
        symlink_to_cache: bool = True,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Downloads files from a repository matching subfolder and pattern criteria.

        **NOTE:** This function currently **DOES NOT filter by modification date** (the `days_ago`
        parameter is ignored for filtering). It uses `list_repo_files` which does not
        provide commit dates efficiently. It downloads matching files individually.

        Args:
            repo_id: The repository ID (e.g., 'google/flan-t5-base').
            days_ago: (Ignored for filtering) Number of days used only for logging context.
            subfolder: Only consider files within this subfolder.
            revision: Git revision (branch, tag, commit hash).
            category: Manually specify category ('model', 'dataset', 'space'). Overrides auto-detection.
            symlink_to_cache: If True, symlink to the cache; otherwise, copy.
            allow_patterns: Glob patterns to include files.
            ignore_patterns: Glob patterns to exclude files.
            **kwargs: Additional keyword arguments passed to `hf_hub_download`.

        Returns:
            The absolute path to the organized directory where files were placed.

        Raises:
            RepositoryNotFoundError: If the repo_id is not found on the Hub.
            HfHubHTTPError: For other HTTP errors from the Hub API.
            ValueError: For invalid arguments.
            OSError: For filesystem errors during linking/copying.
        """
        start_time = time.time()
        log_ctx = {
            "repo_id": repo_id,
            "days_ago_param": days_ago,  # Log the parameter even if unused for filtering
            "subfolder": subfolder,
            "revision": revision or "main",
            "symlink": symlink_to_cache,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
            **kwargs,
        }
        self.logger.info("download_recent_job_started", **log_ctx)
        # Explicitly warn that date filtering is skipped
        self.logger.warning(
            "date_filtering_skipped_api_limitation",
            reason="Cannot filter by date efficiently using list_repo_files.",
            days_ago=days_ago,
            **log_ctx,
        )

        try:
            # 1. Determine Repository Type and Base Organized Path
            if category and category.lower() in REPO_TYPE_TO_DIR_NAME:
                detected_category = category.lower().rstrip("s")
                self.logger.debug(
                    "using_user_specified_category",
                    category=detected_category,
                    **log_ctx,
                )
            else:
                detected_category = self._determine_repo_type(repo_id)
            log_ctx["category"] = detected_category
            repo_type_arg = detected_category
            # Get the base path for the repo (e.g., .../models/google/flan-t5-base)
            org_repo_path = self._get_organized_path(repo_id, detected_category)
            # Get the target download path (including subfolder if specified)
            org_download_path = self._get_organized_path(
                repo_id, detected_category, subfolder
            )
            org_download_path.mkdir(
                parents=True, exist_ok=True
            )  # Ensure target dir exists

            # 2. List all files in the repository using list_repo_files
            self.logger.debug("listing_repo_files", **log_ctx)
            all_repo_filenames: List[str] = list_repo_files(
                repo_id=repo_id,
                revision=revision,
                repo_type=repo_type_arg,
                token=self.api.token,
            )
            self.logger.info(
                "repo_files_listed", count=len(all_repo_filenames), **log_ctx
            )

            # 3. Filter files based on subfolder and patterns (Client-side)
            files_to_download: List[str] = []
            for filename in all_repo_filenames:
                # Subfolder check: File must be within the specified subfolder (or anywhere if no subfolder given)
                in_target_subfolder = True
                if subfolder:
                    norm_subfolder = subfolder.strip("/") + "/"
                    # Check if file starts with the subfolder path
                    if not filename.startswith(norm_subfolder):
                        in_target_subfolder = False
                        # self.logger.debug("file_skipped_subfolder_mismatch", file=filename, **log_ctx)

                if not in_target_subfolder:
                    continue

                # Pattern check
                matches_patterns = True
                # Check allow patterns first
                if allow_patterns:
                    matches_patterns = any(
                        fnmatch.fnmatch(filename, pattern) for pattern in allow_patterns
                    )
                # Then check ignore patterns if it still matches (or if no allow patterns were given)
                if matches_patterns and ignore_patterns:
                    if any(
                        fnmatch.fnmatch(filename, pattern)
                        for pattern in ignore_patterns
                    ):
                        matches_patterns = False

                if not matches_patterns:
                    # self.logger.debug("file_skipped_pattern_mismatch", file=filename, **log_ctx)
                    continue

                # If all checks pass, add the file (relative to repo root)
                files_to_download.append(filename)
                # self.logger.debug("file_marked_for_download", file=filename, **log_ctx)

            if not files_to_download:
                self.logger.info("no_files_matched_criteria", **log_ctx)
                # Still save metadata to indicate an attempt was made
                self._save_download_metadata(
                    org_repo_path,
                    repo_id,
                    detected_category,
                    "filtered_recent_nodate",
                    subfolder,
                    revision,
                )
                return str(org_download_path)  # Return the target directory path

            self.logger.info(
                "downloading_filtered_files", count=len(files_to_download), **log_ctx
            )

            # 4. Download each matching file individually
            downloaded_count = 0
            failed_count = 0
            for (
                file_repo_path
            ) in files_to_download:  # file_repo_path is relative to repo root
                try:
                    # Separate directory and filename for hf_hub_download
                    file_basename = Path(file_repo_path).name
                    file_subfolder_in_repo = str(Path(file_repo_path).parent)
                    # Handle root directory case where parent is '.'
                    if file_subfolder_in_repo == ".":
                        file_subfolder_in_repo = None

                    self.logger.debug(
                        "downloading_individual_file", file=file_repo_path, **log_ctx
                    )
                    downloaded_path_in_cache = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_basename,
                        subfolder=file_subfolder_in_repo,  # Subfolder relative to repo root
                        revision=revision,
                        repo_type=repo_type_arg,
                        token=self.api.token,
                        cache_dir=self.effective_paths["HF_HUB_CACHE"],
                        **kwargs,
                    )

                    # Determine the final path in the organized structure
                    # The file's path relative to the *target* subfolder (or root if no subfolder)
                    relative_path_in_target = file_repo_path
                    if subfolder:
                        # Calculate path relative to the *requested* subfolder
                        relative_path_in_target = str(
                            Path(file_repo_path).relative_to(subfolder.strip("/"))
                        )

                    final_organized_path = org_download_path / relative_path_in_target

                    # 5. Link or Copy the downloaded file
                    self._link_or_copy(downloaded_path_in_cache, final_organized_path, symlink_to_cache)
                    downloaded_count += 1
                except Exception as e_file:
                    # Log failure for specific file but continue with others
                    failed_count += 1
                    self.logger.error(
                        "failed_downloading_individual_file",
                        file=file_repo_path,
                        error=str(e_file),
                        **log_ctx,
                    )

            elapsed = time.time() - start_time
            self.logger.info(
                "download_recent_job_completed",
                files_downloaded=downloaded_count,
                files_failed=failed_count,
                target_path=str(org_download_path),
                elapsed_seconds=round(elapsed, 2),
                **log_ctx,
            )
            # 6. Record Metadata (record as a single "filtered" operation)
            self._save_download_metadata(
                org_repo_path,
                repo_id,
                detected_category,
                "filtered_recent_nodate",
                subfolder,
                revision,
            )
            return str(org_download_path)  # Return the target directory path

        # --- Error Handling ---
        except (
            RepositoryNotFoundError,
            GatedRepoError,
            RevisionNotFoundError,
            EntryNotFoundError,
        ) as e:
            error_type = type(e).__name__
            self.logger.error(
                f"download_recent_failed_{error_type.lower()}", error=str(e), **log_ctx
            )
            raise
        except HfHubHTTPError as http_err:
            self.logger.error(
                "download_recent_failed_http_error",
                status=http_err.response.status_code,
                error=str(http_err),
                request_id=http_err.request_id,
                **log_ctx,
            )
            if http_err.response.status_code == 401:
                raise type(http_err)(
                    f"{http_err} - Check your authentication token (HF_TOKEN).",
                    response=http_err.response,
                ) from http_err
            raise http_err
        except ValueError as e:
            self.logger.error(
                "download_recent_failed_value_error", error=str(e), **log_ctx
            )
            raise
        except OSError as e:
            self.logger.error(
                "download_recent_failed_os_error",
                error=str(e),
                errno=e.errno,
                **log_ctx,
            )
            raise
        except Exception as e:
            self.logger.exception(
                "download_recent_failed_unexpected", error=str(e), **log_ctx
            )
            raise RuntimeError(
                f"An unexpected error occurred during recent file download: {e}"
            ) from e

    def _save_download_metadata(
        self,
        org_repo_path: Path,  # The base path for the repo in organized structure
        repo_id: str,
        category: str,
        download_type: str,  # e.g., 'snapshot', 'config.json', 'filtered_recent_nodate'
        subfolder: Optional[str],
        revision: Optional[str],
    ):
        """Saves download metadata to a JSON file within the .metadata directory."""
        # Metadata is stored at the root of the structured_root, not per-repo
        metadata_dir = Path(self.effective_paths["structured_root"]) / METADATA_DIR_NAME
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / METADATA_FILENAME

        metadata = {"downloads": []}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    loaded_data = json.load(f)
                # Basic validation of loaded data
                if isinstance(loaded_data, dict) and isinstance(
                    loaded_data.get("downloads"), list
                ):
                    metadata = loaded_data
                else:
                    self.logger.warning(
                        "invalid_metadata_format_resetting", path=str(metadata_file)
                    )
            except (json.JSONDecodeError, Exception) as e:
                self.logger.error(
                    "failed_loading_metadata", path=str(metadata_file), error=str(e)
                )
                # Continue with empty metadata if loading fails

        # Create the new entry
        # Use UTC time for consistency
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # Calculate relative path from structured_root
        relative_path = str(
            org_repo_path.relative_to(self.effective_paths["structured_root"])
        )

        entry = {
            "timestamp": timestamp,
            "repo_id": repo_id,
            "category": category,
            "type": download_type,  # What was downloaded (file, snapshot, filtered)
            "relative_path": relative_path,  # Path relative to organized root
            "profile": self.selected_profile,  # Record which profile was used
            "revision": revision or "main",
            # Only include subfolder if it was actually provided
            **({"subfolder": subfolder} if subfolder else {}),
        }

        # Prepend the new entry and limit history size
        metadata["downloads"].insert(0, entry)
        max_history = 500  # Configurable?
        metadata["downloads"] = metadata["downloads"][:max_history]

        # Save back to the file
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)  # Keep indent for readability
            self.logger.debug(
                "download_metadata_saved",
                file=str(metadata_file),
                repo_id=repo_id,
                type=download_type,
            )
        except Exception as e:
            self.logger.error(
                "failed_saving_metadata", path=str(metadata_file), error=str(e)
            )

    def list_downloads(
        self,
        limit: Optional[int] = 20,  # Default limit
        category: Optional[str] = None,
        # profile_filter is handled by initializing Organizer with that profile
    ) -> List[Dict[str, Any]]:
        """Lists download history recorded by this tool for the current profile."""
        metadata_file = (
            Path(self.effective_paths["structured_root"])
            / METADATA_DIR_NAME
            / METADATA_FILENAME
        )
        if not metadata_file.exists():
            self.logger.debug(
                "metadata_file_not_found_for_list", path=str(metadata_file)
            )
            return []

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            downloads = metadata.get("downloads", [])
            if not isinstance(downloads, list):
                raise ValueError("Invalid format: 'downloads' key is not a list.")
        except (json.JSONDecodeError, ValueError, Exception) as e:
            self.logger.error(
                "failed_loading_metadata_for_list",
                path=str(metadata_file),
                error=str(e),
            )
            return []

        # Filter by category if provided
        filtered_downloads = downloads
        if category:
            valid_category = category.lower().rstrip("s")
            filtered_downloads = [
                d
                for d in filtered_downloads
                if isinstance(d, dict) and d.get("category") == valid_category
            ]

        # Apply limit
        return filtered_downloads[:limit] if limit and limit > 0 else filtered_downloads

    def scan_cache(self) -> List[Dict[str, Any]]:
        """
        Scans the Hugging Face cache directory using scan_cache_dir.

        Returns:
            A list of dictionaries, each representing a cached snapshot revision,
            sorted by size descending. Includes repo_id, type, revision, size,
            file count, largest file, last modified date, and cache path.
        """
        cache_dir_to_scan = self.effective_paths["HF_HUB_CACHE"]
        self.logger.info("scanning_cache_start", path=cache_dir_to_scan)
        results = []
        try:
            # Use the scan_cache_dir function from huggingface_hub
            scan_result = scan_cache_dir(cache_dir=cache_dir_to_scan)
            self.logger.info(
                "cache_scan_complete_hf_hub",
                repos=scan_result.repos_count,
                size=scan_result.size_on_disk_str,
                warnings=len(scan_result.warnings),
                path=cache_dir_to_scan,
            )
            if scan_result.warnings:
                for warning in scan_result.warnings:
                    self.logger.warning("cache_scan_warning", details=str(warning))

            # Process the results into the desired format
            for repo_info in scan_result.repos:
                for revision_info in repo_info.revisions:
                    # Find largest file within the snapshot (can be slow for huge snapshots)
                    largest_file_info = {"name": "N/A", "size": 0}
                    try:
                        files_in_snapshot = list(
                            revision_info.files
                        )  # Access the files property
                        if files_in_snapshot:
                            largest_file_obj = max(
                                files_in_snapshot, key=lambda f: f.size_on_disk
                            )
                            largest_file_info = {
                                "name": largest_file_obj.file_name,
                                "size": largest_file_obj.size_on_disk,
                            }
                    except Exception as e:
                        self.logger.warning(
                            "failed_finding_largest_file",
                            snapshot_path=str(revision_info.snapshot_path),
                            error=str(e),
                        )

                    results.append(
                        {
                            "repo_id": repo_info.repo_id,
                            "repo_type": repo_info.repo_type,
                            "revision": revision_info.commit_hash,
                            "size_bytes": revision_info.size_on_disk,
                            "size_human": revision_info.size_on_disk_str,
                            "file_count": len(
                                revision_info.files
                            ),  # Use len of files set
                            "largest_file": largest_file_info,
                            "last_modified": revision_info.last_modified.isoformat()
                            if revision_info.last_modified
                            else None,
                            "cache_path": str(revision_info.snapshot_path),
                            "refs": sorted(
                                list(revision_info.refs)
                            ),  # Get associated refs (branches/tags)
                        }
                    )

            # Sort results by size descending
            results.sort(key=lambda x: x["size_bytes"], reverse=True)
            self.logger.info("cache_scan_processing_complete", count=len(results))
            return results

        except Exception as e:
            # Catch potential errors during the scan itself
            self.logger.exception(
                "cache_scan_failed_hf_hub", path=cache_dir_to_scan, error=str(e)
            )
            return []  # Return empty list on failure

    def get_cache_stats(self) -> Dict[str, Any]:
        """Aggregates statistics from the cache scan."""
        self.logger.info("calculating_cache_stats_start")
        # Use the already processed list from scan_cache
        cache_items = self.scan_cache()  # This already logs start/completion

        if not cache_items:
            self.logger.info("cache_stats_empty")
            return {
                "total_size": 0,
                "total_size_human": "0 B",
                "repo_count": 0,
                "snapshot_count": 0,
                "file_count": 0,
                "largest_snapshots": [],
                "organizations": [],
            }

        # Calculate totals directly from the list structure scan_cache provides
        # Note: HFCacheInfo from scan_cache_dir already provides total size and counts
        try:
            scan_info = scan_cache_dir(self.effective_paths["HF_HUB_CACHE"])
            total_size = scan_info.size_on_disk
            total_files = sum(
                len(rev.files) for repo in scan_info.repos for rev in repo.revisions
            )  # More accurate file count
            repo_count = scan_info.repos_count
            snapshot_count = sum(len(repo.revisions) for repo in scan_info.repos)
        except Exception as e:
            self.logger.error(
                "failed_getting_scan_info_for_stats",
                error=str(e),
                action="falling_back_manual_aggregation",
            )
            # Fallback calculations if scan_cache_dir fails again here (unlikely but safe)
            total_size = sum(item["size_bytes"] for item in cache_items)
            total_files = sum(item["file_count"] for item in cache_items)
            unique_repos = {item["repo_id"] for item in cache_items}
            repo_count = len(unique_repos)
            snapshot_count = len(cache_items)

        # Get top 5 largest snapshots (already sorted by scan_cache)
        largest_snapshots = cache_items[:5]

        # Aggregate by organization/namespace
        org_aggregation: Dict[str, Dict[str, Any]] = {}
        for item in cache_items:
            # Determine namespace (org or user, fallback to 'library')
            namespace = "library"
            if "/" in item["repo_id"]:
                namespace = item["repo_id"].split("/")[0]

            if namespace not in org_aggregation:
                org_aggregation[namespace] = {"size_bytes": 0, "snapshot_count": 0}
            # Add size and count for each snapshot
            org_aggregation[namespace]["size_bytes"] += item["size_bytes"]
            org_aggregation[namespace]["snapshot_count"] += 1

        # Format organization data
        formatted_orgs = [
            {
                "name": org,
                "size_bytes": stats["size_bytes"],
                "size_human": humanize.naturalsize(stats["size_bytes"]),
                "snapshot_count": stats["snapshot_count"],
                "percentage": (stats["size_bytes"] / total_size * 100)
                if total_size > 0
                else 0,
            }
            for org, stats in org_aggregation.items()
        ]
        # Sort organizations by size descending
        top_orgs = sorted(formatted_orgs, key=lambda x: x["size_bytes"], reverse=True)

        stats_result = {
            "total_size": total_size,
            "total_size_human": humanize.naturalsize(total_size),
            "repo_count": repo_count,
            "snapshot_count": snapshot_count,
            "file_count": total_files,
            "largest_snapshots": largest_snapshots,  # Contains detailed dicts per snapshot
            "organizations": top_orgs,  # Contains detailed dicts per org
        }
        self.logger.info(
            "cache_stats_calculated", **stats_result
        )  # Log calculated stats
        return stats_result

    def clean_cache(
        self,
        older_than_days: Optional[int] = None,
        min_size_mb: Optional[int] = None,
        dry_run: bool = False,
    ) -> Tuple[int, int, List[Dict]]:
        """
        Cleans the cache by removing snapshot revisions based on age and/or size criteria.

        Uses huggingface_hub's HFCacheInfo.delete_revisions() method for actual deletion.

        Args:
            older_than_days: Remove snapshots older than this many days.
            min_size_mb: Only consider snapshots >= this size (in MB) for removal
                         if combined with older_than_days, otherwise removes snapshots >= this size.
            dry_run: If True, only logs what would be removed without actually deleting.

        Returns:
            A tuple: (number of snapshots removed, total bytes freed, list of removed snapshot details).
        """
        action_verb = "dry_run_clean_cache" if dry_run else "clean_cache"
        log_ctx = {
            "older_than": older_than_days,
            "min_size_mb": min_size_mb,
            "dry_run": dry_run,
        }
        self.logger.info(f"{action_verb}_started", **log_ctx)

        if older_than_days is None and min_size_mb is None:
            self.logger.error(f"{action_verb}_failed_no_criteria", **log_ctx)
            raise ValueError(
                "At least one criteria (--older-than or --min-size) must be provided for cleaning."
            )

        try:
            # Perform a fresh scan to get HFCacheInfo object needed for deletion
            scan_info = scan_cache_dir(self.effective_paths["HF_HUB_CACHE"])
            self.logger.debug(
                "cache_scan_for_clean_complete",
                repos=scan_info.repos_count,
                size=scan_info.size_on_disk_str,
            )
        except Exception as e:
            self.logger.exception(
                "cache_scan_failed_before_clean", error=str(e), **log_ctx
            )
            return (0, 0, [])  # Cannot proceed without scan info

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        revisions_to_delete: List[str] = []
        potential_items_details: List[Dict] = []  # For logging/reporting

        # Iterate through the structured scan results to identify revisions to delete
        for repo_info in scan_info.repos:
            for revision_info in repo_info.revisions:
                meets_criteria = False
                meets_age = False
                meets_size = False

                # Check age criterion
                if older_than_days is not None and revision_info.last_modified:
                    # Ensure last_modified is timezone-aware (assume UTC if not)
                    last_mod_aware = revision_info.last_modified
                    if last_mod_aware.tzinfo is None:
                        last_mod_aware = last_mod_aware.replace(
                            tzinfo=datetime.timezone.utc
                        )
                    age_delta = now_utc - last_mod_aware
                    if age_delta.days >= older_than_days:
                        meets_age = True

                # Check size criterion
                if min_size_mb is not None:
                    size_in_mb = revision_info.size_on_disk / (1024 * 1024)
                    if size_in_mb >= min_size_mb:
                        meets_size = True

                # Determine if criteria are met based on provided args
                if older_than_days is not None and min_size_mb is not None:
                    # Must meet BOTH criteria if both are specified
                    meets_criteria = meets_age and meets_size
                elif older_than_days is not None:
                    # Meets criteria if age matches
                    meets_criteria = meets_age
                elif min_size_mb is not None:
                    # Meets criteria if size matches
                    meets_criteria = meets_size

                if meets_criteria:
                    revisions_to_delete.append(revision_info.commit_hash)
                    potential_items_details.append(
                        {
                            "repo_id": repo_info.repo_id,
                            "revision": revision_info.commit_hash,
                            "size_bytes": revision_info.size_on_disk,
                            "size_human": revision_info.size_on_disk_str,
                            "last_modified": revision_info.last_modified.isoformat()
                            if revision_info.last_modified
                            else None,
                            "refs": sorted(list(revision_info.refs)),
                            "cache_path": str(revision_info.snapshot_path),
                        }
                    )
                # else:
                #     self.logger.debug("keeping_revision_criteria_not_met", repo_id=repo_info.repo_id, revision=revision_info.commit_hash)

        if not revisions_to_delete:
            self.logger.info(f"{action_verb}_no_items_match_criteria", **log_ctx)
            return (0, 0, [])

        self.logger.info(
            f"{action_verb}_revisions_identified",
            count=len(revisions_to_delete),
            **log_ctx,
        )

        if dry_run:
            # In dry run, just report what would be deleted
            total_potential_freed = sum(
                item["size_bytes"] for item in potential_items_details
            )
            self.logger.info(
                "dry_run_clean_cache_would_remove",
                count=len(potential_items_details),
                size=humanize.naturalsize(total_potential_freed),
                **log_ctx,
            )
            # Sort details by size for reporting
            potential_items_details.sort(key=lambda x: x["size_bytes"], reverse=True)
            return (
                len(potential_items_details),
                total_potential_freed,
                potential_items_details,
            )

        # --- Actual Deletion ---
        try:
            # Use HFCacheInfo's delete_revisions method - it handles complexity
            delete_strategy = scan_info.delete_revisions(*revisions_to_delete)
            self.logger.info(
                "cache_delete_strategy_calculated",
                expected_freed_size=delete_strategy.expected_freed_size_str,
                revisions_count=len(delete_strategy.revisions),
                blobs_count=len(delete_strategy.blobs),
                repos_to_delete_count=len(delete_strategy.repos),
                **log_ctx,
            )

            # Execute the deletion plan
            delete_result = delete_strategy.execute()
            freed_bytes = delete_result.get("freed_size", 0)

            self.logger.info(
                "clean_cache_completed",
                space_freed=humanize.naturalsize(freed_bytes),
                revisions_processed_count=len(
                    revisions_to_delete
                ),  # Revisions we *tried* to delete
                **log_ctx,
            )
            # Filter potential_items_details to only include those whose revision hash
            # matches the hashes in the delete_strategy's successful revisions (if available)
            # For simplicity, we'll return the details of *all* revisions we attempted to delete.
            # A more precise approach would re-scan or parse delete_result carefully if needed.
            potential_items_details.sort(key=lambda x: x["size_bytes"], reverse=True)
            return len(revisions_to_delete), freed_bytes, potential_items_details

        except Exception as e:
            self.logger.exception(
                "clean_cache_execution_failed", error=str(e), **log_ctx
            )
            # Return 0, 0 and empty list indicating failure during deletion
            return (0, 0, [])

    def get_organization_overview(self) -> Dict[str, Any]:
        """
        Provides an overview of the organized directory structure.

        Walks the structured_root, calculates sizes (ignoring symlinks),
        and aggregates by category (models, datasets, spaces) and
        organization/namespace.

        Returns:
            A dictionary containing total size and nested category/organization data.
        """
        structured_root_path = Path(self.effective_paths["structured_root"])
        if not structured_root_path.is_dir():
            self.logger.warning(
                "organized_root_not_found", path=str(structured_root_path)
            )
            return {"total_size": 0, "total_size_human": "0 B", "categories": {}}

        self.logger.info(
            "generating_organization_overview", path=str(structured_root_path)
        )
        overview: Dict[str, Any] = {"total_size_bytes": 0, "categories": {}}

        try:
            # Iterate through top-level directories (categories: models, datasets, spaces)
            for cat_path in structured_root_path.iterdir():
                # Skip files and hidden directories like .metadata
                if not cat_path.is_dir() or cat_path.name.startswith("."):
                    continue
                cat_name = cat_path.name
                cat_data: Dict[str, Any] = {"size_bytes": 0, "organizations": {}}
                overview["categories"][cat_name] = cat_data

                # Iterate through organization/namespace directories
                for org_path in cat_path.iterdir():
                    if not org_path.is_dir():
                        continue
                    org_name = org_path.name
                    org_data: Dict[str, Any] = {
                        "size_bytes": 0,
                        "repo_count": 0,
                        "repos": [],
                    }
                    cat_data["organizations"][org_name] = org_data

                    # Iterate through repository directories
                    for repo_path in org_path.iterdir():
                        if not repo_path.is_dir():
                            continue
                        repo_name = repo_path.name
                        repo_size_bytes = 0
                        symlink_count = 0
                        regular_file_count = 0

                        # Walk the repository directory to calculate size of actual files
                        try:
                            for item in repo_path.rglob("*"):  # Recursive glob
                                if item.is_symlink():
                                    symlink_count += 1
                                elif item.is_file():
                                    try:
                                        # Get size of the actual file, not the symlink itself
                                        repo_size_bytes += item.stat().st_size
                                        regular_file_count += 1
                                    except OSError as stat_err:
                                        self.logger.warning(
                                            "could_not_get_size_overview",
                                            file=str(item),
                                            error=str(stat_err),
                                        )
                        except OSError as walk_err:
                            self.logger.warning(
                                "error_walking_repo_dir_overview",
                                path=str(repo_path),
                                error=str(walk_err),
                            )

                        # Only add repo if it contains actual files (size > 0 or symlinks > 0)
                        if (
                            repo_size_bytes > 0
                            or symlink_count > 0
                            or regular_file_count > 0
                        ):
                            org_data["repos"].append(
                                {
                                    "name": repo_name,
                                    "size_bytes": repo_size_bytes,
                                    "size_human": humanize.naturalsize(repo_size_bytes),
                                    "symlink_count": symlink_count,
                                    "file_count": regular_file_count,  # Count of non-symlink files
                                    "path": str(
                                        repo_path.relative_to(structured_root_path)
                                    ),
                                }
                            )
                            org_data["size_bytes"] += (
                                repo_size_bytes  # Add repo size to org total
                            )

                    # Finalize organization data if repos were found
                    if org_data["repos"]:
                        org_data["repo_count"] = len(org_data["repos"])
                        org_data["size_human"] = humanize.naturalsize(
                            org_data["size_bytes"]
                        )
                        org_data["repos"].sort(
                            key=lambda x: x["size_bytes"], reverse=True
                        )  # Sort repos by size
                        cat_data["size_bytes"] += org_data[
                            "size_bytes"
                        ]  # Add org size to category total
                    else:
                        # Remove org if no valid repos found
                        del cat_data["organizations"][org_name]

                # Finalize category data if organizations were found
                if cat_data["organizations"]:
                    cat_data["org_count"] = len(cat_data["organizations"])
                    cat_data["size_human"] = humanize.naturalsize(
                        cat_data["size_bytes"]
                    )
                    overview["total_size_bytes"] += cat_data[
                        "size_bytes"
                    ]  # Add category size to overall total
                else:
                    # Remove category if no valid orgs found
                    del overview["categories"][cat_name]

            # Finalize total size
            overview["total_size_human"] = humanize.naturalsize(
                overview["total_size_bytes"]
            )
            self.logger.info(
                "organization_overview_complete",
                total_size=overview["total_size_human"],
                categories_count=len(overview["categories"]),
            )
            return overview

        except Exception as e:
            # Catch errors during directory iteration
            self.logger.exception(
                "organization_overview_failed",
                path=str(structured_root_path),
                error=str(e),
            )
            # Return a default empty structure on failure
            return {"total_size_bytes": 0, "total_size_human": "0 B", "categories": {}}
