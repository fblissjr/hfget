# hfget_cli.py
import argparse
import json
import sys
import os
import re
from tabulate import tabulate
import humanize
import datetime # Import datetime for type hint and usage in list command

# Import the core class from the other file
# Assuming hf_organizer_core.py is in the same directory or Python path
from hf_organizer_core import HfHubOrganizer, DEFAULT_CONFIG_PATH, DEFAULT_STRUCTURED_ROOT, FALLBACK_HF_HOME
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import structlog # Use structlog for getting logger

# Get the logger instance (assuming core setup logger)
logger = structlog.get_logger("hfget_cli") # Use structlog to get logger

def _create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="HfHubOrganizer CLI: Manage HuggingFace Hub downloads, cache, and organization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Global options affecting HfHubOrganizer instantiation
    parser.add_argument("--profile", help="Profile name to use (defined in config). Overrides default behavior and environment variables for paths/token.")
    parser.add_argument("--log-format", choices=["console", "json", "structured"],
                      default="console", help="Logging output format.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (DEBUG level) logging.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to the configuration file.")
    parser.add_argument("--no-hf-transfer", action="store_true",
                        help="Disable hf_transfer for downloads (overrides profile/default).")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # --- Download Command ---
    download_parser = subparsers.add_parser("download", help="Download a full repository or specific file.")
    download_parser.add_argument("repo_id", help="Repository ID (e.g., 'google/flan-t5-base', 'gpt2')")
    download_parser.add_argument("--filename", "-f", help="Specific file to download within the repo.")
    download_parser.add_argument("--subfolder", "-s", help="Subfolder within the repository to download from/into.")
    download_parser.add_argument("--revision", "-r", help="Git revision (branch, tag, commit hash) to download.")
    download_parser.add_argument("--category", choices=["models", "datasets", "spaces"],
                        help="Manually specify category for organization (overrides auto-detection).")
    download_parser.add_argument("--base-path", help="Override HF_HOME (cache location) for this command (takes precedence over profile).")
    download_parser.add_argument("--out-dir", help="Override structured organization root directory for this command (takes precedence over profile).")
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
    add_parser.add_argument("--token", help="HuggingFace API token (optional, stored in config). WARNING: Stored in plain text.")
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
    list_downloads_parser.add_argument("--filter-profile", help="Show history only for a specific profile name (uses that profile's metadata file).")
    list_downloads_parser.add_argument("--json", action="store_true", help="Output as JSON.")

    # --- Overview Command ---
    overview_parser = subparsers.add_parser("overview", help="Show overview of the organized files directory.")
    overview_parser.add_argument("--json", action="store_true", help="Output as JSON.")

    return parser

def print_json(data):
    """Helper to print JSON output."""
    print(json.dumps(data, indent=2, default=str)) # Use default=str for datetime

def main():
    """Main entry point for the CLI."""
    parser = _create_parser()
    args = parser.parse_args()

    # --- Initialize Organizer ---
    # Pass CLI args that affect initialization
    try:
        # Determine if hf_transfer should be enabled/disabled based on flag
        # If --no-hf-transfer is present, enable_hf_transfer_flag becomes False
        # Otherwise, it's None, letting the profile/default take precedence
        enable_hf_transfer_flag = not args.no_hf_transfer if args.no_hf_transfer else None

        # Handle overrides for specific commands (download, download-recent)
        base_path_override = getattr(args, 'base_path', None)
        out_dir_override = getattr(args, 'out_dir', None)

        # For profile commands, we might not need a fully functional organizer yet,
        # but we need access to config loading/saving.
        # Let's instantiate anyway, it handles config loading.
        organizer = HfHubOrganizer(
            profile=args.profile,
            base_path=base_path_override,
            structured_root=out_dir_override,
            enable_hf_transfer=enable_hf_transfer_flag,
            verbose=args.verbose,
            config_path=args.config,
            log_format=args.log_format
        )
    except ValueError as e:
        # Handle profile not found during init
        print(f"Error: {e}", file=sys.stderr)
        logger.error("init_failed", error=str(e))
        exit(1)
    except Exception as e:
        print(f"Unexpected error during initialization: {e}", file=sys.stderr)
        logger.exception("unexpected_init_failed", error=str(e))
        exit(1)

    # --- Execute Command ---
    try:
        if args.command == "profile":
            if args.profile_command == "list":
                profiles_data = organizer.list_profiles() # Gets the dict
                if profiles_data:
                    print("Available profiles:")
                    headers = ["Name", "Description", "Cache Path (HF_HOME)", "Organized Root", "HF Transfer"]
                    table = []
                    # Use default values from the class for display if not set in profile
                    default_hft_enabled = HfHubOrganizer.BOOLEAN_ENV_VARS['HF_HUB_ENABLE_HF_TRANSFER']
                    default_hft_str = "Enabled" if default_hft_enabled else "Disabled"
                    # Use the fallback path defined in the core module for display
                    default_hf_home_str = f"Default ({FALLBACK_HF_HOME})"

                    for name, p_config in profiles_data.items():
                        desc = p_config.get("description", "N/A")
                        bp = p_config.get("base_path", default_hf_home_str) # <-- Use fallback string
                        sr = p_config.get("structured_root", f"Default ({DEFAULT_STRUCTURED_ROOT})")
                        hft = p_config.get("enable_hf_transfer")
                        hft_str = "Enabled" if hft is True else ("Disabled" if hft is False else f"Default ({default_hft_str})")
                        table.append([name, desc, bp, sr, hft_str])
                    print(tabulate(table, headers=headers))
                else:
                    print("No profiles configured. Use 'hfget profile add <name> ...' to create one.")
                print(f"\nConfig file location: {organizer.config_path}")

            elif args.profile_command == "add":
                enable_transfer = None
                if args.enable_hf_transfer == 'true': enable_transfer = True
                elif args.enable_hf_transfer == 'false': enable_transfer = False
                organizer.add_profile(
                    name=args.name, base_path=args.base_path, structured_root=args.out_dir,
                    token=args.token, enable_hf_transfer=enable_transfer, description=args.description
                )
                print(f"Profile '{args.name}' added/updated successfully.")

            elif args.profile_command == "remove":
                if organizer.remove_profile(args.name):
                     print(f"Profile '{args.name}' removed.")
                else:
                     print(f"Profile '{args.name}' not found.")

        elif args.command == "download":
             dl_kwargs = {'force_download': args.force_download} if args.force_download else {}
             if args.force_download: dl_kwargs['resume_download'] = False

             path = organizer.download(
                 repo_id=args.repo_id, filename=args.filename, subfolder=args.subfolder,
                 revision=args.revision, category=args.category, symlink_to_cache=not args.copy,
                 allow_patterns=args.allow_patterns, ignore_patterns=args.ignore_patterns,
                 **dl_kwargs
             )
             print(f"\nDownload complete. Organized at: {path}")

        elif args.command == "download-recent":
             # Use re for case-insensitive search
             if args.exclude_repo_pattern and re.search(args.exclude_repo_pattern, args.repo_id, re.IGNORECASE):
                 print(f"Skipping repository '{args.repo_id}' due to exclusion pattern '{args.exclude_repo_pattern}'.")
                 logger.info("repo_skipped_exclusion", repo_id=args.repo_id, pattern=args.exclude_repo_pattern)
                 return # Exit gracefully

             dl_kwargs = {'force_download': args.force_download} if args.force_download else {}
             if args.force_download: dl_kwargs['resume_download'] = False

             path = organizer.download_recent(
                 repo_id=args.repo_id, days_ago=args.days, subfolder=args.subfolder,
                 revision=args.revision, category=args.category, symlink_to_cache=not args.copy,
                 allow_patterns=args.allow_patterns, ignore_patterns=args.ignore_patterns,
                 **dl_kwargs
             )
             print(f"\nRecent file download process complete. Target directory: {path}")

        elif args.command == "cache":
            if args.cache_command == "scan":
                cache_stats = organizer.get_cache_stats()
                if args.json:
                    print_json(cache_stats)
                else:
                    print(f"HF Cache Statistics (Profile: {organizer.selected_profile or 'Default'}, Path: {organizer.effective_paths['HF_HUB_CACHE']}):")
                    print("-" * 60)
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
                    if cache_stats['organizations']:
                         print("Storage by Organization/Namespace:")
                         headers = ["Organization", "Size", "Snapshots", "% of Total"]
                         table = [[o['name'], o['size_human'], o['snapshot_count'], f"{o['percentage']:.1f}%"] for o in cache_stats['organizations']]
                         print(tabulate(table, headers=headers))

            elif args.cache_command == "clean":
                if args.older_than is None and args.min_size is None:
                     parser.error("At least one criteria (--older-than DAYS or --min-size MB) must be provided for cleaning.")
                removed_count, freed_bytes, removed_details = organizer.clean_cache(
                    older_than_days=args.older_than, min_size_mb=args.min_size, dry_run=args.dry_run
                )
                action_verb = "Would remove" if args.dry_run else "Removed"
                print(f"{action_verb} {removed_count} snapshots, freeing {humanize.naturalsize(freed_bytes)}.")
                if args.json: print_json(removed_details)
                elif removed_details:
                     print(f"\n{action_verb.capitalize()} snapshots:")
                     headers = ["Repo ID", "Revision", "Size", "Last Modified"]
                     table = [[i['repo_id'], i['revision'][:12], i['size_human'], i['last_modified']] for i in removed_details]
                     print(tabulate(table, headers=headers))

        elif args.command == "list":
            # If filtering by profile, we need to instantiate an organizer for that profile's root
            list_organizer = organizer
            profile_name_for_list = organizer.selected_profile or 'Default'
            if args.filter_profile and args.filter_profile != organizer.selected_profile:
                 try:
                      print(f"Listing history for profile: {args.filter_profile}")
                      # Need to import datetime here if not already imported globally
                      # import datetime # Already imported at top level
                      list_organizer = HfHubOrganizer(profile=args.filter_profile, verbose=args.verbose, config_path=args.config, log_format=args.log_format)
                      profile_name_for_list = args.filter_profile
                 except ValueError:
                      print(f"Error: Profile '{args.filter_profile}' not found for listing history.", file=sys.stderr)
                      exit(1)

            downloads = list_organizer.list_downloads(limit=args.limit, category=args.category) # Profile filter is handled by which organizer we use

            if args.json:
                print_json(downloads)
            else:
                if not downloads:
                    print(f"No download history found for profile '{profile_name_for_list}'.")
                    meta_path = os.path.join(list_organizer.structured_root, '.metadata', 'downloads.json')
                    print(f"(Metadata file searched: {meta_path})")
                else:
                    print(f"Download History (Profile: {profile_name_for_list}, Max: {args.limit}):")
                    headers = ["Timestamp", "Repo ID", "Type", "Category", "Profile", "Revision", "Subfolder", "Rel Path"]
                    table = []
                    # Need to import datetime here if not already imported globally
                    # import datetime # Already imported at top level
                    for item in downloads:
                        ts = item.get("timestamp", "N/A")
                        try: ts = datetime.datetime.fromisoformat(ts).astimezone(None).strftime("%Y-%m-%d %H:%M")
                        except (ValueError, TypeError): ts = ts[:16] if isinstance(ts, str) else "Invalid"
                        table.append([
                            ts, item.get("repo_id", "N/A"), item.get("type", "N/A"), item.get("category", "N/A"),
                            item.get("profile", "N/A"), item.get("revision", "N/A")[:12], item.get("subfolder", "-"),
                            item.get("relative_path", "N/A")
                        ])
                    print(tabulate(table, headers=headers, maxcolwidths=[None, None, 15, None, None, 12, 10, 25]))

        elif args.command == "overview":
            overview = organizer.get_organization_overview()
            if args.json:
                print_json(overview)
            else:
                print(f"Organization Overview (Profile: {organizer.selected_profile or 'Default'}, Root: {organizer.effective_paths['structured_root']})")
                print("-" * 60)
                print(f"Total Organized Size: {overview.get('total_size_human', '0 B')} (excluding symlinks)")
                print()
                categories = overview.get('categories', {})
                if not categories: print("No categories found.")
                else:
                     for category, cat_data in sorted(categories.items()):
                         print(f"--- Category: {category} ({cat_data.get('size_human', '0 B')}) ---")
                         orgs_data = cat_data.get('organizations', {})
                         if not orgs_data: print("  No organizations/namespaces found.")
                         else:
                              sorted_orgs = sorted(orgs_data.items(), key=lambda i: i[1].get('size_bytes', 0), reverse=True)
                              headers = ["Organization/Namespace", "Size", "Repositories"]
                              table = [[name, info.get('size_human', '0 B'), info.get('repo_count', 0)] for name, info in sorted_orgs]
                              print(tabulate(table, headers=headers, tablefmt="psql")) # Changed format for better alignment
                         print()

    except RepositoryNotFoundError as e:
         logger.error("command_failed_repo_not_found", repo_id=getattr(args, 'repo_id', 'N/A'), error=str(e))
         print(f"Error: Repository not found: {getattr(args, 'repo_id', 'N/A')}", file=sys.stderr)
         exit(1)
    except HfHubHTTPError as http_err:
         logger.error("command_failed_http_error", command=args.command, status=http_err.response.status_code, error=str(http_err))
         print(f"\nAn HTTP error occurred: Status {http_err.response.status_code} - {http_err}", file=sys.stderr)
         if http_err.response.status_code == 401:
              print("Hint: Check if your HF_TOKEN is valid and has the necessary permissions.", file=sys.stderr)
         exit(1)
    except Exception as e:
         logger.exception("command_failed_unexpected", command=args.command, error=str(e))
         print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
         print("Check logs or run with --verbose for more details.", file=sys.stderr)
         exit(1)

if __name__ == "__main__":
    main()
