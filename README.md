# hfget
simple wrapper around huggingface_hub for managing hugging face data, metadata, and cache

### Create profiles for different storage locations
```
python hfget.py profile add floppy --base-path /mnt/floppydisk/hf_cache --out-dir /mnt/floppydisk/models --description "3.5 Floppy Disk Storage"
```
```
python hfget.py profile add ssd --base-path /media/ohmyssd/hf_cache --out-dir /media/ohmyssd/models --description "External SSD"
```

### List available profiles
`python hfget.py profile list`

### Download using a specific profile
`python hfget.py download best-model-ever-7b --profile ssd --verbose`

### Scan and analyze cache
`python hfget.py cache scan --profile floppy`

### Clean up old/large models
`python hfget.py cache clean --profile ssd --older-than 30 --min-size 1000`

### Dry run to see what would be removed
`python hfget.py cache clean --profile nas --older-than 60 --dry-run`

### List recent downloads
`python hfget.py list --profile ssd --limit 10`

### Filter by category
`python hfget.py list --category models --json`

### Get overview of organized files
`python hfget.py overview --profile floppy`

### Export as JSON for further processing
`python hfget.py overview --json > storage_report.json`
