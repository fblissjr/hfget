# hfget

A command-line tool to simplify managing Hugging Face Hub models, datasets, and spaces, focusing on organized storage, profile management, and cache control.

Built using `huggingface_hub` and `structlog`.

## Features

* **Profiles:** Define different storage configurations (cache location, organized output directory, token).
* **Organized Downloads:** Download entire repos or specific files into a structured directory (`{structured_root}/{category}/{namespace}/{repo_name}`).
* **Symlink or Copy:** Choose between symlinking to the cache (default, saves space) or copying files.
* **Recent File Downloads:** Download only files modified within a specified number of days.
* **Cache Management:** Scan cache usage, view statistics, and clean up old or large snapshots.
* **Download History:** Track downloads performed by this tool (per profile).
* **Structured Logging:** Configurable logging (console, JSON) with verbosity control.

## Installation

1.  Clone the repository or download `hfget.py`.
2.  Install required libraries:
    ```bash
    pip install huggingface_hub structlog tabulate humanize python-dotenv # Optional: dotenv for .env file support
    ```

## Usage

```bash
python hfget.py <command> [<args>]
Configuration ProfilesProfiles allow you to easily switch between different storage setups (e.g., SSD vs. NAS, different projects).Add/Update a Profile:# Example: Profile for an external SSD
python hfget.py profile add ssd --base-path /media/my_ssd/hf_cache --out-dir /media/my_ssd/hf_models --description "External SSD Storage" --token hf_YOUR_TOKEN_optional

# Example: Profile using default locations but with a description
python hfget.py profile add default_project --description "Main project profile"
--base-path: Sets the HF_HOME environment variable for this profile (where the cache lives).--out-dir: Sets the root directory for organized downloads.--token: Optionally store an HF token within the profile config (use with caution).Paths support ~ expansion.List Profiles:python hfget.py profile list
Remove a Profile:python hfget.py profile remove old_profile_name
Using Profiles: Add --profile <profile_name> to other commands (download, cache, list, overview). If --profile is omitted, it uses default paths or environment variables.DownloadingDownload Full Repository:python hfget.py download google/flan-t5-base --profile ssd
Download Specific File:python hfget.py download stabilityai/stable-diffusion-xl-base-1.0 --filename text_encoder/model.safetensors --profile ssd --copy
--copy: Copies the file instead of symlinking (uses more disk space in the output dir).Download Subfolder:python hfget.py download sentence-transformers/all-MiniLM-L6-v2 --subfolder data --profile default_project
Download Only Recently Modified Files: (NEW)# Download files in the main branch modified in the last 7 days
python hfget.py download-recent google/flan-t5-base --days 7 --profile ssd

# Download files in a specific subfolder modified in the last 30 days
python hfget.py download-recent stabilityai/stable-diffusion-xl-base-1.0 --days 30 --subfolder scheduler --profile ssd
Cache ManagementScan Cache Usage:python hfget.py cache scan --profile ssd
# Output as JSON
python hfget.py cache scan --profile ssd --json
Clean Cache:# Remove snapshots older than 60 days
python hfget.py cache clean --profile ssd --older-than 60

# Remove snapshots larger than 5000 MB
python hfget.py cache clean --profile ssd --min-size 5000

# Remove snapshots older than 30 days AND larger than 1000 MB
python hfget.py cache clean --profile ssd --older-than 30 --min-size 1000

# Dry run: See what would be removed without deleting
python hfget.py cache clean --profile ssd --older-than 90 --dry-run
History & OverviewList Recent Downloads (Tracked by this tool):python hfget.py list --profile ssd --limit 15
# Output as JSON
python hfget.py list --profile ssd --json
Get Overview of Organized Directory:python hfget.py overview --profile ssd
# Output as JSON
python hfget.py overview --profile ssd --json > storage_report.json
Global Options--profile <name>: Use a specific configuration profile.--verbose or -v: Enable detailed DEBUG logging.--log-format <format>: Set logging format (console, json, structured).--config <path>: Specify a custom path to the config.json file.Configuration FileThe tool stores profiles in ~/.config/hf_organizer/config.json by default. You can specify a different path using the --config argument.Example config.json:{
  "profiles": {
    "ssd": {
      "base_path": "/media/my_ssd/hf_cache",
      "structured_root": "/media/my_ssd/hf_models",
      "description": "External SSD Storage",
      "token": null
    },
    "default_project": {
      "description": "Main project profile"
    }
  }
}
