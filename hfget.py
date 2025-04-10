import os
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Optional, Union, List, Any, Tuple
import datetime
import humanize
from tabulate import tabulate

try:
    import structlog
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download, scan_cache_dir
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    missing = []
    for lib in ["structlog", "huggingface_hub", "tabulate", "humanize"]:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        raise ImportError(f"Required libraries not found: {', '.join(missing)}. Install with: pip install {' '.join(missing)}")

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
        log_format: str = "console"
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
                self.logger.error("profile_not_found", profile=profile, 
                                  available=list(self.config.get("profiles", {}).keys()))
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

        # Initialize all environment variables
        self._initialize_env_vars()

        # Keep track of effective paths
        self.effective_paths = {
            "HF_HOME": os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")),
            "HF_HUB_CACHE": os.path.expanduser(os.environ.get("HF_HUB_CACHE", "")),
            "structured_root": self.structured_root
        }

        self.logger.info("initialized", 
                         profile=profile,
                         structured_root=self.structured_root, 
                         hf_home=self.effective_paths["HF_HOME"])

        # Initialize HF API
        self.api = HfApi()

    def _setup_logger(self, verbose: bool, format_type: str) -> structlog.BoundLogger:
        """Set up structured logging."""
        import logging  # Standard library logging for level constants

        # Use actual logging module constants instead of string references
        log_level = logging.DEBUG if verbose else logging.INFO

        # Configure processors based on format
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        ]

        if format_type == "json":
            processors.extend([
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ])
        elif format_type == "structured":
            processors.extend([
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer()
            ])
        else:  # console (default)
            processors.extend([
                structlog.dev.set_exc_info,
                structlog.dev.ConsoleRenderer(colors=True)
            ])

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(log_level),  # Use logging constant
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger()

    def _load_config(self) -> Dict[str, Any]:
        """Load config from disk or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.logger.debug("config_loaded", path=self.config_path)
                    return config
            except json.JSONDecodeError:
                self.logger.warning("invalid_config", path=self.config_path)
                return {"profiles": {}}
        self.logger.debug("default_config_created")
        return {"profiles": {}}

    def _save_config(self):
        """Save current config to disk."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            self.logger.debug("config_saved", path=self.config_path)

    def list_profiles(self) -> List[str]:
        """List all available profiles."""
        return list(self.config.get("profiles", {}).keys())

    def add_profile(
        self,
        name: str,
        base_path: Optional[str] = None,
        structured_root: Optional[str] = None,
        token: Optional[str] = None,
        description: Optional[str] = None
    ):
        """Add or update a profile."""
        if "profiles" not in self.config:
            self.config["profiles"] = {}

        self.config["profiles"][name] = {
            "base_path": base_path,
            "structured_root": structured_root,
            "token": token,
            "description": description or f"Profile created for {name}"
        }

        self._save_config()
        self.logger.info("profile_added", name=name)

    def remove_profile(self, name: str):
        """Remove a profile."""
        if "profiles" in self.config and name in self.config["profiles"]:
            del self.config["profiles"][name]
            self._save_config()
            self.logger.info("profile_removed", name=name)
        else:
            self.logger.warning("profile_not_found", name=name)

    def _initialize_env_vars(self):
        """Initialize all environment variables with defaults."""
        # Process string environment variables
        for key, default_value in self.ENV_VARS.items():
            if key not in os.environ and default_value is not None:
                # Handle variable expansion in default values
                if isinstance(default_value, str) and "${" in default_value:
                    expanded_value = default_value
                    for var_name, var_value in os.environ.items():
                        placeholder = "${" + var_name + "}"
                        if placeholder in expanded_value:
                            expanded_value = expanded_value.replace(placeholder, var_value)
                    os.environ[key] = expanded_value
                else:
                    os.environ[key] = default_value

        # Process boolean environment variables
        for key, default_value in self.BOOLEAN_ENV_VARS.items():
            if key not in os.environ:
                os.environ[key] = "1" if default_value else "0"

    def download(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        category: Optional[str] = None,
        symlink_to_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Download and organize in a human-readable structure.
        """
        start_time = time.time()
        self.logger.info("download_started", repo_id=repo_id, filename=filename or "entire_repo")

        # Determine the category from repo_id if not provided
        if category is None:
            if "/" in repo_id:
                namespace, repo_name = repo_id.split("/", 1)
                # Try to guess the category
                try:
                    repo_type = self.api.repo_type(repo_id)
                    if repo_type == "dataset":
                        category = "datasets"
                    elif repo_type == "space":
                        category = "spaces"
                    else:
                        category = "models"
                    self.logger.debug("category_detected", repo_id=repo_id, category=category)
                except Exception as e:
                    # Fallback if API call fails
                    category = "models"
                    self.logger.warning("category_detection_failed", repo_id=repo_id, error=str(e))
            else:
                category = "models"
                namespace = "huggingface"
                repo_name = repo_id
        else:
            # Parse the repo_id
            if "/" in repo_id:
                namespace, repo_name = repo_id.split("/", 1)
            else:
                namespace = "huggingface" 
                repo_name = repo_id

        # Create organized path
        org_path = os.path.join(
            self.structured_root,
            category,
            namespace,
            repo_name
        )

        # Add subfolder if specified
        if subfolder:
            org_path = os.path.join(org_path, subfolder)

        os.makedirs(org_path, exist_ok=True)
        self.logger.info("organizing_files", path=org_path)

        # Save metadata about this download
        self._save_download_metadata(org_path, repo_id, category, filename)

        # Download file or repo
        try:
            if filename:
                # Download a specific file
                cache_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder,
                    revision=revision,
                    **kwargs
                )

                # Create organized file path
                org_file_path = os.path.join(org_path, os.path.basename(filename))

                if symlink_to_cache:
                    # Create symlink to cache
                    if os.path.exists(org_file_path):
                        os.remove(org_file_path)
                    os.symlink(cache_path, org_file_path)
                    self.logger.debug("symlink_created", source=cache_path, target=org_file_path)
                else:
                    # Copy file
                    shutil.copy2(cache_path, org_file_path)
                    self.logger.debug("file_copied", source=cache_path, target=org_file_path)

                download_path = org_file_path
            else:
                # Download entire repo
                cache_path = snapshot_download(
                    repo_id=repo_id,
                    subfolder=subfolder,
                    revision=revision,
                    **kwargs
                )

                if symlink_to_cache:
                    # Link files individually to maintain correct structure
                    for root, dirs, files in os.walk(cache_path):
                        for file in files:
                            src_file = os.path.join(root, file)
                            rel_path = os.path.relpath(src_file, cache_path)
                            dst_file = os.path.join(org_path, rel_path)

                            # Create parent directory
                            os.makedirs(os.path.dirname(dst_file), exist_ok=True)

                            # Create symlink
                            if os.path.exists(dst_file):
                                os.remove(dst_file)
                            os.symlink(src_file, dst_file)
                    self.logger.debug("symlinks_created", source=cache_path, target=org_path)
                else:
                    # Copy entire repo
                    file_count = 0
                    for item in os.listdir(cache_path):
                        src = os.path.join(cache_path, item)
                        dst = os.path.join(org_path, item)
                        if os.path.isdir(src):
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(src, dst)
                            file_count += sum(len(files) for _, _, files in os.walk(src))
                        else:
                            if os.path.exists(dst):
                                os.remove(dst)
                            shutil.copy2(src, dst)
                            file_count += 1
                    self.logger.debug("files_copied", count=file_count, source=cache_path, target=org_path)

                download_path = org_path

            elapsed = time.time() - start_time
            self.logger.info("download_completed", 
                            path=download_path, 
                            elapsed_seconds=round(elapsed, 2),
                            profile=self.selected_profile)

            return download_path

        except Exception as e:
            self.logger.error("download_failed", 
                             repo_id=repo_id, 
                             error=str(e),
                             error_type=type(e).__name__)
            raise

    def _save_download_metadata(self, path: str, repo_id: str, category: str, filename: Optional[str] = None):
        """Save metadata about downloaded repo/file for future reference."""
        metadata_dir = os.path.join(self.structured_root, ".metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        metadata_file = os.path.join(metadata_dir, "downloads.json")

        # Load existing metadata
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = {"downloads": []}
        else:
            metadata = {"downloads": []}

        # Add new entry
        entry = {
            "repo_id": repo_id,
            "category": category,
            "path": os.path.relpath(path, self.structured_root),
            "timestamp": datetime.datetime.now().isoformat(),
            "profile": self.selected_profile,
        }
        if filename:
            entry["filename"] = filename

        metadata["downloads"].append(entry)

        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def list_downloads(self, limit: Optional[int] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all downloads across profiles with filtering options."""
        metadata_file = os.path.join(self.structured_root, ".metadata", "downloads.json")

        if not os.path.exists(metadata_file):
            self.logger.warning("no_download_history")
            return []

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        downloads = metadata.get("downloads", [])

        # Apply filters
        if category:
            downloads = [d for d in downloads if d.get("category") == category]

        # Sort by timestamp (newest first)
        downloads.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Apply limit
        if limit and limit > 0:
            downloads = downloads[:limit]

        return downloads

    def scan_cache(self) -> List[Dict[str, Any]]:
        """Scan the HF cache and return detailed information."""
        try:
            cache_dir = os.environ.get("HF_HUB_CACHE", os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"))
            cache_dir = os.path.expanduser(cache_dir)

            if not os.path.exists(cache_dir):
                self.logger.warning("cache_dir_not_found", path=cache_dir)
                return []

            self.logger.info("scanning_cache", path=cache_dir)

            # Directly scan the file system instead of relying on scan_cache_dir
            result = []

            # Walk through the cache directory
            for root, dirs, files in os.walk(cache_dir):
                # Skip certain directories like .git
                if ".git" in root or "__pycache__" in root:
                    continue

                # Identify repository information from path
                rel_path = os.path.relpath(root, cache_dir)
                parts = rel_path.split(os.sep)

                # Skip the top level directory
                if rel_path == ".":
                    continue

                # Extract repo_id and revision
                if len(parts) >= 2:
                    # Handle the repo ID format (could be org--repo or just repo)
                    repo_id_part = parts[0]
                    if "--" in repo_id_part:
                        # Convert org--repo format back to org/repo
                        repo_id = repo_id_part.replace("--", "/", 1)
                    else:
                        repo_id = repo_id_part

                    # Get revision (usually the second part of the path)
                    revision = parts[1] if len(parts) > 1 else "main"

                    # Only process directories that have files
                    if files:
                        # Measure directory size
                        dir_size = 0
                        file_count = 0
                        largest_file = {"name": "", "size": 0}

                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.isfile(file_path):
                                file_size = os.path.getsize(file_path)
                                dir_size += file_size
                                file_count += 1

                                if file_size > largest_file["size"]:
                                    largest_file = {"name": file, "size": file_size}

                        # Check if we already have an entry for this repo/revision
                        existing = next((item for item in result if 
                                        item["repo_id"] == repo_id and 
                                        item["revision"] == revision), None)

                        if existing:
                            # Update existing entry
                            existing["size_bytes"] += dir_size
                            existing["file_count"] += file_count
                            if largest_file["size"] > existing["largest_file"]["size"]:
                                existing["largest_file"] = largest_file
                        else:
                            # Create new entry
                            last_modified = os.path.getmtime(root)
                            result.append({
                                "repo_id": repo_id,
                                "revision": revision,
                                "size_bytes": dir_size,
                                "size_human": humanize.naturalsize(dir_size),
                                "file_count": file_count,
                                "largest_file": largest_file,
                                "last_modified": datetime.datetime.fromtimestamp(last_modified).isoformat()
                            })

            # Sort by size (largest first)
            result.sort(key=lambda x: x["size_bytes"], reverse=True)
            self.logger.info("cache_scan_complete", repos_found=len(result))
            return result

        except Exception as e:
            self.logger.error("cache_scan_failed", error=str(e))
            return []


    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache usage."""
        cache_items = self.scan_cache()

        if not cache_items:
            return {
                "total_size": 0,
                "total_size_human": "0 B",
                "repo_count": 0,
                "file_count": 0,
                "largest_repos": []
            }

        total_size = sum(item["size_bytes"] for item in cache_items)
        total_files = sum(item["file_count"] for item in cache_items)

        # Get top 5 largest repos
        largest_repos = sorted(cache_items, key=lambda x: x["size_bytes"], reverse=True)[:5]

        # Group by organization
        orgs = {}
        for item in cache_items:
            repo_id = item["repo_id"]
            if "/" in repo_id:
                org = repo_id.split("/")[0]
            else:
                org = "huggingface"

            if org not in orgs:
                orgs[org] = {"size_bytes": 0, "repo_count": 0}

            orgs[org]["size_bytes"] += item["size_bytes"]
            orgs[org]["repo_count"] += 1

        # Calculate percentages and format sizes
        for org in orgs:
            orgs[org]["size_human"] = humanize.naturalsize(orgs[org]["size_bytes"])
            orgs[org]["percentage"] = (orgs[org]["size_bytes"] / total_size) * 100 if total_size > 0 else 0

        # Sort organizations by size
        top_orgs = sorted(
            [{"name": org, **stats} for org, stats in orgs.items()], 
            key=lambda x: x["size_bytes"], 
            reverse=True
        )

        return {
            "total_size": total_size,
            "total_size_human": humanize.naturalsize(total_size),
            "repo_count": len(cache_items),
            "file_count": total_files,
            "largest_repos": largest_repos,
            "organizations": top_orgs
        }

    def clean_cache(self, older_than_days: Optional[int] = None, min_size_mb: Optional[int] = None) -> Tuple[int, int]:
        """
        Clean up cache entries matching criteria.

        Args:
            older_than_days: Remove entries older than this many days
            min_size_mb: Only consider entries larger than this size in MB

        Returns:
            Tuple of (number of repos removed, bytes freed)
        """
        cache_items = self.scan_cache()
        self.logger.info("clean_cache_started", 
                        older_than_days=older_than_days, 
                        min_size_mb=min_size_mb)

        if not cache_items:
            self.logger.info("cache_empty")
            return (0, 0)

        now = datetime.datetime.now()
        freed_bytes = 0
        removed_count = 0

        for item in cache_items:
            should_remove = True

            # Check age criteria
            if older_than_days is not None:
                last_mod = datetime.datetime.fromisoformat(item["last_modified"])
                age_days = (now - last_mod).days
                if age_days < older_than_days:
                    should_remove = False

            # Check size criteria
            if min_size_mb is not None:
                size_mb = item["size_bytes"] / (1024 * 1024)
                if size_mb < min_size_mb:
                    should_remove = False

            if should_remove:
                try:
                    # Get paths for all files in this repo and remove them
                    cache_dir = os.environ.get("HF_HUB_CACHE", 
                                             os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"))
                    cache_dir = os.path.expanduser(cache_dir)

                    # This is approximate since we don't have direct access to HF's internal structure,
                    # but it's a reasonable approach based on how huggingface_hub organizes files
                    repo_dir = os.path.join(cache_dir, item["repo_id"].replace("/", "--"), item["revision"])

                    if os.path.exists(repo_dir):
                        self.logger.info("removing_repo", 
                                        repo_id=item["repo_id"], 
                                        size=item["size_human"])
                        shutil.rmtree(repo_dir)
                        freed_bytes += item["size_bytes"]
                        removed_count += 1

                except Exception as e:
                    self.logger.error("failed_to_remove", 
                                     repo_id=item["repo_id"], 
                                     error=str(e))

        self.logger.info("clean_cache_completed", 
                        repos_removed=removed_count, 
                        space_freed=humanize.naturalsize(freed_bytes))

        return (removed_count, freed_bytes)

    def get_organization_overview(self) -> Dict[str, Any]:
        """Get an overview of how files are organized in the structured root."""
        if not os.path.exists(self.structured_root):
            self.logger.warning("structured_root_not_found", path=self.structured_root)
            return {"total_size": 0, "categories": {}}

        total_size = 0
        categories = {}

        # Collect data for each category
        for category in os.listdir(self.structured_root):
            category_path = os.path.join(self.structured_root, category)

            # Skip hidden files and directories
            if category.startswith('.') or not os.path.isdir(category_path):
                continue

            category_size = 0
            orgs = {}

            # Analyze each organization within category
            for org in os.listdir(category_path):
                org_path = os.path.join(category_path, org)

                if not os.path.isdir(org_path):
                    continue

                org_size = 0
                repos = []

                # Analyze each repo within org
                for repo in os.listdir(org_path):
                    repo_path = os.path.join(org_path, repo)

                    if not os.path.isdir(repo_path):
                        continue

                    # Calculate repo size
                    repo_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(repo_path)
                        for filename in filenames
                        if os.path.isfile(os.path.join(dirpath, filename)) and not os.path.islink(os.path.join(dirpath, filename))
                    )

                    symlink_count = sum(
                        1 for dirpath, _, filenames in os.walk(repo_path)
                        for filename in filenames
                        if os.path.islink(os.path.join(dirpath, filename))
                    )

                    repos.append({
                        "name": repo,
                        "size_bytes": repo_size,
                        "size_human": humanize.naturalsize(repo_size),
                        "symlink_count": symlink_count,
                        "path": os.path.relpath(repo_path, self.structured_root)
                    })

                    org_size += repo_size

                if repos:
                    orgs[org] = {
                        "size_bytes": org_size,
                        "size_human": humanize.naturalsize(org_size),
                        "repo_count": len(repos),
                        "repos": sorted(repos, key=lambda x: x["size_bytes"], reverse=True)
                    }

                    category_size += org_size

            if orgs:
                categories[category] = {
                    "size_bytes": category_size,
                    "size_human": humanize.naturalsize(category_size),
                    "org_count": len(orgs),
                    "organizations": orgs
                }

                total_size += category_size

        return {
            "total_size": total_size,
            "total_size_human": humanize.naturalsize(total_size),
            "categories": categories
        }

def main():
    import argparse

    parser = argparse.ArgumentParser(description="HuggingFace Hub Organizer")
    # Add profile argument at the top level so it's available for all commands
    parser.add_argument("--profile", help="Profile to use")
    parser.add_argument("--log-format", choices=["console", "json", "structured"], 
                      default="console", help="Logging format")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a repository or file")
    download_parser.add_argument("repo_id", help="Repository ID (e.g., gpt2, facebook/bart-large)")
    download_parser.add_argument("--filename", help="Specific file to download")
    download_parser.add_argument("--category", choices=["models", "datasets", "spaces"], 
                        help="Category for organization")
    download_parser.add_argument("--subfolder", help="Subfolder within the repository")
    # Note: Don't need to add --profile here since it's defined at the parent level
    download_parser.add_argument("--base-path", help="Base path for HF_HOME (overrides profile)")
    download_parser.add_argument("--out-dir", help="Output directory for organized files (overrides profile)")
    download_parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    download_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Profile management
    profile_parser = subparsers.add_parser("profile", help="Manage profiles")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command", help="Profile command")

    # List profiles
    list_parser = profile_subparsers.add_parser("list", help="List available profiles")

    # Add profile
    add_parser = profile_subparsers.add_parser("add", help="Add a new profile")
    add_parser.add_argument("name", help="Profile name")
    add_parser.add_argument("--base-path", help="Base path for HF cache")
    add_parser.add_argument("--out-dir", help="Directory for organized files")
    add_parser.add_argument("--token", help="HuggingFace token")
    add_parser.add_argument("--description", help="Profile description")

    # Remove profile
    remove_parser = profile_subparsers.add_parser("remove", help="Remove a profile")
    remove_parser.add_argument("name", help="Profile name to remove")

    # Cache management
    cache_parser = subparsers.add_parser("cache", help="Manage HF cache")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", help="Cache command")

    # Scan cache
    scan_parser = cache_subparsers.add_parser("scan", help="Scan and analyze cache")
    scan_parser.add_argument("--profile", help="Profile to use")
    scan_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Clean cache
    clean_parser = cache_subparsers.add_parser("clean", help="Clean up cache")
    clean_parser.add_argument("--profile", help="Profile to use")
    clean_parser.add_argument("--older-than", type=int, help="Remove items older than N days")
    clean_parser.add_argument("--min-size", type=int, help="Remove items larger than N MB")
    clean_parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually removing")

    # List downloads
    list_downloads_parser = subparsers.add_parser("list", help="List downloaded repositories")
    list_downloads_parser.add_argument("--profile", help="Profile to use")
    list_downloads_parser.add_argument("--limit", type=int, help="Limit number of results")
    list_downloads_parser.add_argument("--category", choices=["models", "datasets", "spaces"], help="Filter by category")
    list_downloads_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Overview
    overview_parser = subparsers.add_parser("overview", help="Show overview of organized files")
    overview_parser.add_argument("--profile", help="Profile to use")
    overview_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle profile management
    if args.command == "profile":
        organizer = HfHubOrganizer(log_format=args.log_format)

        if args.profile_command == "list":
            profiles = organizer.list_profiles()
            if profiles:
                print("Available profiles:")
                for profile in profiles:
                    desc = organizer.config["profiles"][profile].get("description", "No description")
                    root = organizer.config["profiles"][profile].get("structured_root", "Default")
                    print(f"  - {profile}: {desc} (Root: {root})")
            else:
                print("No profiles configured.")

        elif args.profile_command == "add":
            organizer.add_profile(
                name=args.name,
                base_path=args.base_path,
                structured_root=args.out_dir,
                token=args.token,
                description=args.description
            )
            print(f"Profile '{args.name}' added successfully.")

        elif args.profile_command == "remove":
            organizer.remove_profile(args.name)
            print(f"Profile '{args.name}' removed.")

    # Handle downloads
    elif args.command == "download":
        organizer = HfHubOrganizer(
            profile=args.profile,
            base_path=args.base_path,
            structured_root=args.out_dir,
            verbose=args.verbose,
            log_format=args.log_format
        )

        path = organizer.download(
            repo_id=args.repo_id,
            filename=args.filename,
            subfolder=args.subfolder,
            category=args.category,
            symlink_to_cache=not args.copy
        )

        print(f"Downloaded to: {path}")

    # Handle cache management
    elif args.command == "cache":
        organizer = HfHubOrganizer(profile=args.profile, log_format=args.log_format)

        if args.cache_command == "scan":
            if args.json:
                import json
                print(json.dumps(organizer.scan_cache(), indent=2))
            else:
                cache_stats = organizer.get_cache_stats()

                print(f"HF Cache Statistics:")
                print(f"-------------------")
                print(f"Total size: {cache_stats['total_size_human']}")
                print(f"Repositories: {cache_stats['repo_count']}")
                print(f"Files: {cache_stats['file_count']}")
                print()

                print("Largest repositories:")
                for repo in cache_stats['largest_repos']:
                    print(f"  - {repo['repo_id']} ({repo['size_human']})")
                print()

                print("Storage by organization:")
                headers = ["Organization", "Size", "Repos", "% of Total"]
                table = []
                for org in cache_stats['organizations']:
                    table.append([
                        org['name'],
                        org['size_human'],
                        org['repo_count'],
                        f"{org['percentage']:.1f}%"
                    ])
                print(tabulate(table, headers=headers))

        elif args.cache_command == "clean":
            if args.dry_run:
                cache_items = organizer.scan_cache()
                if not cache_items:
                    print("Cache is empty.")
                    return

                now = datetime.datetime.now()
                total_size = 0
                would_remove = []

                for item in cache_items:
                    should_remove = True

                    if args.older_than is not None:
                        last_mod = datetime.datetime.fromisoformat(item["last_modified"])
                        age_days = (now - last_mod).days
                        if age_days < args.older_than:
                            should_remove = False

                    if args.min_size is not None:
                        size_mb = item["size_bytes"] / (1024 * 1024)
                        if size_mb < args.min_size:
                            should_remove = False

                    if should_remove:
                        would_remove.append(item)
                        total_size += item["size_bytes"]

                print(f"Dry run: Would remove {len(would_remove)} repositories ({humanize.naturalsize(total_size)})")
                if would_remove:
                    print("\nRepositories that would be removed:")
                    for item in would_remove:
                        print(f"  - {item['repo_id']} ({item['size_human']}, last modified: {item['last_modified']})")
            else:
                removed, freed = organizer.clean_cache(
                    older_than_days=args.older_than,
                    min_size_mb=args.min_size
                )
                print(f"Removed {removed} repositories, freed {humanize.naturalsize(freed)}.")

    # Handle downloads list
    elif args.command == "list":
        organizer = HfHubOrganizer(profile=args.profile, log_format=args.log_format)
        downloads = organizer.list_downloads(limit=args.limit, category=args.category)

        if args.json:
            import json
            print(json.dumps(downloads, indent=2))
        else:
            if not downloads:
                print("No downloads found.")
                return

            print(f"Recent downloads ({len(downloads)} total):")
            headers = ["Repo ID", "Category", "Profile", "Date", "Path"]
            table = []
            for item in downloads:
                date = datetime.datetime.fromisoformat(item["timestamp"]).strftime("%Y-%m-%d %H:%M")
                table.append([
                    item["repo_id"],
                    item["category"],
                    item.get("profile", "N/A"),
                    date,
                    item["path"]
                ])
            print(tabulate(table, headers=headers))

    # Handle overview
    elif args.command == "overview":
        organizer = HfHubOrganizer(profile=args.profile, log_format=args.log_format)
        overview = organizer.get_organization_overview()

        if args.json:
            import json
            print(json.dumps(overview, indent=2))
        else:
            print(f"Organization Overview:")
            print(f"---------------------")
            print(f"Total size: {overview['total_size_human']}")
            print()

            # Print categories
            for category, cat_data in overview['categories'].items():
                print(f"Category: {category} ({cat_data['size_human']})")
                print("-" * (10 + len(category) + len(cat_data['size_human'])))

                # Get top organizations by size
                orgs = []
                for org_name, org_data in cat_data['organizations'].items():
                    orgs.append({
                        "name": org_name,
                        "size_bytes": org_data["size_bytes"],
                        "size_human": org_data["size_human"],
                        "repo_count": org_data["repo_count"]
                    })

                orgs.sort(key=lambda x: x["size_bytes"], reverse=True)

                headers = ["Organization", "Size", "Repositories"]
                table = []
                for org in orgs[:5]:  # Show top 5
                    table.append([
                        org['name'],
                        org['size_human'],
                        org['repo_count']
                    ])
                print(tabulate(table, headers=headers))
                print()

if __name__ == "__main__":
    main()
