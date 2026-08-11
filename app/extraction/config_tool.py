"""
Dynamic Configuration Management & Calibration Tool.
Allows creating, updating, listing, and deleting YAML document configurations and fields.
"""

from typing import Dict, Any, List, Optional
import os
import yaml


def get_config_dir() -> str:
    """Return path to configs directory."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs")


def list_all_configs() -> Dict[str, Dict[str, Any]]:
    """
    List all YAML configuration files in configs/ directory and return their contents.
    """
    cfg_dir = get_config_dir()
    configs = {}
    if os.path.exists(cfg_dir):
        for fname in os.listdir(cfg_dir):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                fpath = os.path.join(cfg_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        configs[fname] = yaml.safe_load(f) or {}
                except Exception:
                    configs[fname] = {}
    return configs


def get_single_config(config_name: str) -> Dict[str, Any]:
    """
    Get content of a single YAML configuration file.
    """
    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        config_name = f"{config_name}.yaml"
    
    fpath = os.path.join(get_config_dir(), config_name)
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def create_new_config(config_name: str, document_type: str, language: str = "en") -> bool:
    """
    Create a new YAML configuration file.
    """
    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        config_name = f"{config_name}.yaml"

    fpath = os.path.join(get_config_dir(), config_name)
    data = {
        "document_type": document_type,
        "language": language,
        "fields": {}
    }
    
    os.makedirs(get_config_dir(), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return True


def save_field_config(
    config_name: str,
    field_key: str,
    label: str,
    language: str = "en",
    strategy: str = "region",
    region: Optional[Dict[str, float]] = None,
    anchor: Optional[Dict[str, Any]] = None,
    normalization: str = "none",
    validator: str = "none"
) -> bool:
    """
    Add or update a field configuration entry in a specified YAML file.
    """
    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        config_name = f"{config_name}.yaml"

    fpath = os.path.join(get_config_dir(), config_name)
    data = get_single_config(config_name)
    
    if "fields" not in data or not isinstance(data["fields"], dict):
        data["fields"] = {}

    field_data: Dict[str, Any] = {
        "label": label or field_key.replace("_", " ").title(),
        "language": language,
        "strategy": strategy,
        "normalization": normalization,
        "validator": validator
    }

    if region:
        field_data["region"] = {
            "x1": round(float(region.get("x1", 0.0)), 3),
            "y1": round(float(region.get("y1", 0.0)), 3),
            "x2": round(float(region.get("x2", 1.0)), 3),
            "y2": round(float(region.get("y2", 1.0)), 3)
        }

    if anchor and anchor.get("keyword"):
        field_data["anchor"] = {
            "keyword": anchor.get("keyword"),
            "direction": anchor.get("direction", "right"),
            "fallback_to_region": anchor.get("fallback_to_region", True)
        }

    data["fields"][field_key] = field_data

    os.makedirs(get_config_dir(), exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return True


def update_field_region_config(
    config_path: str,
    field_name: str,
    region: Dict[str, float],
    label: Optional[str] = None,
    strategy: Optional[str] = "region",
    normalization: Optional[str] = "none",
    validator: Optional[str] = "none"
) -> bool:
    """Backward compatibility alias for save_field_config."""
    return save_field_config(
        config_name=config_path,
        field_key=field_name,
        label=label or field_name.replace("_", " ").title(),
        strategy=strategy or "region",
        region=region,
        normalization=normalization or "none",
        validator=validator or "none"
    )


def delete_field_config(config_name: str, field_key: str) -> bool:
    """
    Delete a field entry from a configuration file.
    """
    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        config_name = f"{config_name}.yaml"

    fpath = os.path.join(get_config_dir(), config_name)
    data = get_single_config(config_name)
    
    if "fields" in data and field_key in data["fields"]:
        del data["fields"][field_key]
        with open(fpath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return True
    return False
