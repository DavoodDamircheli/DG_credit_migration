"""Config loading and result saving utilities."""
import yaml


def load_config(path):
    """Load a YAML config file. Returns dict."""
    with open(path) as f:
        return yaml.safe_load(f)
