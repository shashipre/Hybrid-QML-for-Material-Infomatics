"""Configuration loader module."""
import yaml
from typing import Any, Dict, Optional

def _update_dict(d: dict, u: dict) -> dict:
    """Recursively updates a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = _update_dict(d[k], v)
        else:
            d[k] = v
    return d

def get_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and applies optional overrides.
    
    Args:
        path: Path to the YAML file.
        overrides: Optional dictionary of overrides to apply.
        
    Returns:
        A nested dictionary representing the configuration.
    """
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    if config is None:
        config = {}
        
    if overrides:
        _update_dict(config, overrides)
        
    return config
