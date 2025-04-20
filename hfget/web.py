# hfget_web.py
import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Import the core class
from hf_organizer_core import HfHubOrganizer, DEFAULT_CONFIG_PATH
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key") # Change for production

# --- Global Organizer Instance ---
# Instantiate the organizer once when the app starts.
# It will use the default profile unless overridden by env vars.
# For a multi-user or more complex setup, you might instantiate per request
# or use Flask's application context (g).
try:
    # You might want to configure the profile used by the web app via env vars
    # For simplicity, using default profile here.
    # Set verbose=True for web logs if desired
    organizer = HfHubOrganizer(
        config_path=os.environ.get("HFGET_CONFIG_PATH", DEFAULT_CONFIG_PATH),
        verbose=True, # Make web logs verbose by default
        log_format="console" # Or 'json'/'structured'
    )
    logger = organizer.logger # Use the logger from the organizer instance
except Exception as e:
    print(f"FATAL: Could not initialize HfHubOrganizer for web app: {e}", file=sys.stderr)
    # Optionally log to a fallback logger if organizer.logger failed
    # exit(1) # Exit if core component fails

# --- Helper ---
def safe_str_to_bool(s):
    if isinstance(s, str):
        return s.lower() in ['true', '1', 't', 'y', 'yes']
    return bool(s)

# --- Routes ---
@app.route('/')
def index():
    """Home page, shows overview."""
    try:
        overview_data = organizer.get_organization_overview()
    except Exception as e:
        logger.exception("web_overview_failed", error=str(e))
        flash(f"Error getting overview: {e}", "error")
        overview_data = {"total_size_human": "Error", "categories": {}}
    return render_template('index.html', overview=overview_data)

@app.route('/profiles')
def list_profiles():
    """List configured profiles."""
    try:
        profiles_data = organizer.list_profiles()
    except Exception as e:
        logger.exception("web_list_profiles_failed", error=str(e))
        flash(f"Error listing profiles: {e}", "error")
        profiles_data = {}
    return render_template('profiles.html', profiles=profiles_data)

@app.route('/cache')
def cache_stats():
    """Display cache statistics."""
    try:
        stats = organizer.get_cache_stats()
    except Exception as e:
        logger.exception("web_cache_stats_failed", error=str(e))
        flash(f"Error getting cache stats: {e}", "error")
        stats = None
    return render_template('cache.html', stats=stats)

@app.route('/history')
def download_history():
    """Display download history."""
    limit = request.args.get('limit', 20, type=int)
    category = request.args.get('category')
    profile = request.args.get('profile') # Allow filtering by profile via query param
    try:
        # If filtering by profile, instantiate a temporary organizer for that profile's root
        list_organizer_instance = organizer
        if profile and profile != organizer.selected_profile:
             try:
                  list_organizer_instance = HfHubOrganizer(profile=profile, verbose=True, config_path=organizer.config_path)
             except ValueError:
                  flash(f"Profile '{profile}' not found for history view.", "warning")
                  # Fallback to default organizer or show error?
                  # Let's show history for default profile in this case
                  list_organizer_instance = organizer
                  profile = organizer.selected_profile # Update profile var for template

        history = list_organizer_instance.list_downloads(limit=limit, category=category)
    except Exception as e:
        logger.exception("web_history_failed", error=str(e))
        flash(f"Error getting download history: {e}", "error")
        history = []
    return render_template('history.html', history=history, limit=limit, selected_category=category, selected_profile=profile)

@app.route('/download', methods=['GET', 'POST'])
def download_repo():
    """Handle download form and trigger download."""
    if request.method == 'POST':
        repo_id = request.form.get('repo_id')
        filename = request.form.get('filename') or None # Empty string becomes None
        subfolder = request.form.get('subfolder') or None
        revision = request.form.get('revision') or None
        category = request.form.get('category') or None
        symlink = safe_str_to_bool(request.form.get('symlink', 'on')) # Checkbox value is 'on' if checked
        force = safe_str_to_bool(request.form.get('force', 'off'))
        allow_patterns = request.form.get('allow_patterns')
        ignore_patterns = request.form.get('ignore_patterns')

        allow_list = allow_patterns.split(',') if allow_patterns else None
        ignore_list = ignore_patterns.split(',') if ignore_patterns else None

        if not repo_id:
            flash("Repository ID is required.", "error")
            return redirect(url_for('download_repo'))

        try:
            logger.info("web_download_request", repo_id=repo_id, filename=filename, subfolder=subfolder, revision=revision, symlink=symlink, force=force)
            # Run download synchronously (will block the request)
            dl_kwargs = {'force_download': force} if force else {}
            if force: dl_kwargs['resume_download'] = False

            # Use the global organizer instance
            organized_path = organizer.download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                revision=revision,
                category=category,
                symlink_to_cache=symlink,
                allow_patterns=allow_list,
                ignore_patterns=ignore_list,
                **dl_kwargs
            )
            flash(f"Download started for {repo_id}. Organized at: {organized_path}", "success")
            # Redirect back to form or to a status page? Redirecting back for simplicity.
            return redirect(url_for('download_repo'))

        except (RepositoryNotFoundError, HfHubHTTPError, ValueError, Exception) as e:
            logger.exception("web_download_failed", repo_id=repo_id, error=str(e))
            flash(f"Error downloading {repo_id}: {e}", "error")
            return redirect(url_for('download_repo')) # Redirect back to form with error

    # GET request: just show the form
    return render_template('download_form.html')


# --- Templates (should be in a 'templates' folder) ---

# templates/base.html (Example base template)
"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}HF Organizer{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 5rem; }
        .flash-error { background-color: #f8d7da; color: #842029; border-color: #f5c2c7; padding: 1rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .flash-success { background-color: #d1e7dd; color: #0f5132; border-color: #badbcc; padding: 1rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .flash-warning { background-color: #fff3cd; color: #664d03; border-color: #ffecb5; padding: 1rem; margin-bottom: 1rem; border: 1px solid transparent; border-radius: .25rem; }
        .table th { white-space: nowrap; }
    </style>
  </head>
  <body>
    <nav class="navbar navbar-expand-md navbar-dark bg-dark fixed-top">
      <div class="container-fluid">
        <a class="navbar-brand" href="{{ url_for('index') }}">HF Organizer</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarCollapse" aria-controls="navbarCollapse" aria-expanded="false" aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarCollapse">
          <ul class="navbar-nav me-auto mb-2 mb-md-0">
            <li class="nav-item"><a class="nav-link" href="{{ url_for('index') }}">Overview</a></li>
            <li class="nav-item"><a class="nav-link" href="{{ url_for('download_repo') }}">Download</a></li>
            <li class="nav-item"><a class="nav-link" href="{{ url_for('cache_stats') }}">Cache</a></li>
            <li class="nav-item"><a class="nav-link" href="{{ url_for('download_history') }}">History</a></li>
            <li class="nav-item"><a class="nav-link" href="{{ url_for('list_profiles') }}">Profiles</a></li>
          </ul>
        </div>
      </div>
    </nav>

    <main class="container">
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
          {% for category, message in messages %}
            <div class="flash-{{ category }}">{{ message }}</div>
          {% endfor %}
        {% endif %}
      {% endwith %}
      {% block content %}{% endblock %}
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

# templates/index.html
"""
{% extends 'base.html' %}
{% block title %}Overview - HF Organizer{% endblock %}

{% block content %}
  <h1>Organization Overview</h1>
  <p>Profile: <strong>{{ organizer.selected_profile or 'Default' }}</strong></p>
  <p>Root Directory: <code>{{ organizer.effective_paths['structured_root'] }}</code></p>
  <p>Total Size (excluding symlinks): <strong>{{ overview.total_size_human | default('N/A') }}</strong></p>
  <hr>

  {% if overview.categories %}
    {% for category, cat_data in overview.categories | dictsort %}
      <h2>{{ category | capitalize }} ({{ cat_data.size_human | default('0 B') }})</h2>
      {% if cat_data.organizations %}
        <table class="table table-sm table-striped">
          <thead>
            <tr><th>Organization/Namespace</th><th>Size</th><th>Repositories</th></tr>
          </thead>
          <tbody>
            {% for org_name, org_info in cat_data.organizations | dictsort(by='value.size_bytes', reverse=true) %}
            <tr>
              <td>{{ org_name }}</td>
              <td>{{ org_info.size_human | default('0 B') }}</td>
              <td>{{ org_info.repo_count | default(0) }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      {% else %}
        <p class="text-muted">No organizations/namespaces found in this category.</p>
      {% endif %}
      <br>
    {% endfor %}
  {% else %}
    <p class="text-muted">No categories found in the organized directory.</p>
  {% endif %}
{% endblock %}
"""

# templates/profiles.html
"""
{% extends 'base.html' %}
{% block title %}Profiles - HF Organizer{% endblock %}

{% block content %}
  <h1>Configuration Profiles</h1>
  <p>Config file: <code>{{ organizer.config_path }}</code></p>

  {% if profiles %}
    <table class="table table-striped">
      <thead>
        <tr>
          <th>Name</th>
          <th>Description</th>
          <th>Cache Path (HF_HOME)</th>
          <th>Organized Root</th>
          <th>HF Transfer</th>
        </tr>
      </thead>
      <tbody>
        {% for name, config in profiles | dictsort %}
        <tr>
          <td><strong>{{ name }}</strong> {% if name == organizer.selected_profile %}<span class="badge bg-success">Active</span>{% endif %}</td>
          <td>{{ config.description | default('N/A') }}</td>
          <td>{{ config.base_path | default('Default (~/.cache/huggingface)') }}</td>
          <td>{{ config.structured_root | default('Default (~/huggingface_organized)') }}</td>
          <td>
            {% if config.enable_hf_transfer is true %}Enabled
            {% elif config.enable_hf_transfer is false %}Disabled
            {% else %}Default ({% if organizer.BOOLEAN_ENV_VARS['HF_HUB_ENABLE_HF_TRANSFER'] %}Enabled{% else %}Disabled{% endif %})
            {% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <p class="text-muted">No profiles configured. Use the CLI 'hfget profile add ...' to create one.</p>
  {% endif %}
{% endblock %}
"""

# templates/cache.html
"""
{% extends 'base.html' %}
{% block title %}Cache Stats - HF Organizer{% endblock %}

{% block content %}
  <h1>Cache Statistics</h1>
  <p>Profile: <strong>{{ organizer.selected_profile or 'Default' }}</strong></p>
  <p>Cache Path: <code>{{ organizer.effective_paths['HF_HUB_CACHE'] }}</code></p>

  {% if stats %}
    <div class="row">
      <div class="col-md-6">
        <ul class="list-group mb-3">
          <li class="list-group-item d-flex justify-content-between align-items-center">
            Total Size: <strong>{{ stats.total_size_human }}</strong>
          </li>
          <li class="list-group-item d-flex justify-content-between align-items-center">
            Unique Repos: <strong>{{ stats.repo_count }}</strong>
          </li>
          <li class="list-group-item d-flex justify-content-between align-items-center">
            Total Snapshots: <strong>{{ stats.snapshot_count }}</strong>
          </li>
           <li class="list-group-item d-flex justify-content-between align-items-center">
            Total Files: <strong>{{ stats.file_count }}</strong>
          </li>
        </ul>
      </div>
    </div>

    {% if stats.largest_snapshots %}
      <h2>Largest Snapshots</h2>
      <table class="table table-sm table-striped">
        <thead><tr><th>Repo ID</th><th>Revision</th><th>Size</th><th>Last Modified</th></tr></thead>
        <tbody>
          {% for s in stats.largest_snapshots %}
          <tr>
            <td>{{ s.repo_id }}</td>
            <td><code>{{ s.revision[:12] }}</code></td>
            <td>{{ s.size_human }}</td>
            <td>{{ s.last_modified | default('N/A') }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    {% endif %}

    {% if stats.organizations %}
      <h2>Storage by Organization/Namespace</h2>
      <table class="table table-sm table-striped">
        <thead><tr><th>Organization</th><th>Size</th><th>Snapshots</th><th>% of Total</th></tr></thead>
        <tbody>
          {% for o in stats.organizations %}
          <tr>
            <td>{{ o.name }}</td>
            <td>{{ o.size_human }}</td>
            <td>{{ o.snapshot_count }}</td>
            <td>{{ "%.1f" | format(o.percentage) }}%</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    {% endif %}

  {% else %}
     <p class="text-muted">Could not retrieve cache statistics.</p>
  {% endif %}
{% endblock %}
"""

# templates/history.html
"""
{% extends 'base.html' %}
{% block title %}Download History - HF Organizer{% endblock %}

{% block content %}
  <h1>Download History</h1>
   <p>Profile: <strong>{{ selected_profile or 'Default' }}</strong> (Showing last {{ limit }} entries)</p>
   {# Add filtering options here if needed #}

  {% if history %}
    <table class="table table-sm table-striped table-hover">
      <thead>
        <tr>
          <th>Timestamp (Local)</th>
          <th>Repo ID</th>
          <th>Type</th>
          <th>Category</th>
          <th>Profile</th>
          <th>Revision</th>
          <th>Subfolder</th>
          <th>Relative Path</th>
        </tr>
      </thead>
      <tbody>
        {% for item in history %}
        <tr>
          <td>{{ item.timestamp[:16] if item.timestamp else 'N/A' }}</td> {# Basic formatting #}
          <td>{{ item.repo_id | default('N/A') }}</td>
          <td>{{ item.type | default('N/A') }}</td>
          <td>{{ item.category | default('N/A') }}</td>
          <td>{{ item.profile | default('N/A') }}</td>
          <td><code>{{ (item.revision | default('N/A'))[:12] }}</code></td>
          <td>{{ item.subfolder | default('-') }}</td>
          <td>{{ item.relative_path | default('N/A') }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <p class="text-muted">No download history found for this profile.</p>
  {% endif %}
{% endblock %}
"""

# templates/download_form.html
"""
{% extends 'base.html' %}
{% block title %}Download - HF Organizer{% endblock %}

{% block content %}
  <h1>Download Repository or File</h1>
  <p>Using profile: <strong>{{ organizer.selected_profile or 'Default' }}</strong></p>

  <form method="POST" action="{{ url_for('download_repo') }}">
    <div class="mb-3">
      <label for="repo_id" class="form-label">Repository ID *</label>
      <input type="text" class="form-control" id="repo_id" name="repo_id" placeholder="e.g., google/flan-t5-base or gpt2" required>
    </div>
    <div class="mb-3">
      <label for="filename" class="form-label">Filename (Optional)</label>
      <input type="text" class="form-control" id="filename" name="filename" placeholder="e.g., config.json (download entire repo if blank)">
    </div>
     <div class="mb-3">
      <label for="subfolder" class="form-label">Subfolder (Optional)</label>
      <input type="text" class="form-control" id="subfolder" name="subfolder" placeholder="e.g., models/text-generation">
    </div>
    <div class="mb-3">
      <label for="revision" class="form-label">Revision (Optional)</label>
      <input type="text" class="form-control" id="revision" name="revision" placeholder="e.g., main, v1.0, commit_hash">
    </div>
     <div class="mb-3">
      <label for="category" class="form-label">Category (Optional)</label>
      <select class="form-select" id="category" name="category">
        <option value="">Auto-detect</option>
        <option value="models">Models</option>
        <option value="datasets">Datasets</option>
        <option value="spaces">Spaces</option>
      </select>
    </div>
     <div class="mb-3">
      <label for="allow_patterns" class="form-label">Allow Patterns (Snapshot only, comma-separated)</label>
      <input type="text" class="form-control" id="allow_patterns" name="allow_patterns" placeholder="e.g., *.json,data/*">
    </div>
     <div class="mb-3">
      <label for="ignore_patterns" class="form-label">Ignore Patterns (Snapshot only, comma-separated)</label>
      <input type="text" class="form-control" id="ignore_patterns" name="ignore_patterns" placeholder="e.g., *.safetensors,logs/*">
    </div>
    <div class="form-check mb-3">
      <input class="form-check-input" type="checkbox" value="on" id="symlink" name="symlink" checked>
      <label class="form-check-label" for="symlink">
        Symlink to cache (uncheck to copy files)
      </label>
    </div>
     <div class="form-check mb-3">
      <input class="form-check-input" type="checkbox" value="on" id="force" name="force">
      <label class="form-check-label" for="force">
        Force download (ignore cache)
      </label>
    </div>
    <button type="submit" class="btn btn-primary">Start Download</button>
  </form>
{% endblock %}
"""


# --- Run Development Server ---
if __name__ == '__main__':
    # Run with flask run --debug for development
    # For production, use a proper WSGI server like gunicorn or waitress
    # Example: gunicorn --bind 0.0.0.0:5555 hfget_web:app
    app.run(debug=True, host='0.0.0.0', port=5555)
