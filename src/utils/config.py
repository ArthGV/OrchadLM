import yaml

class Config:
    """Wraps a dict and allows attribute access. Missing keys return None."""
    def __init__(self, d: dict):
        for k, v in d.items():
            v = scientific_to_float(v) if isinstance(v, str) else v
            setattr(self, k.lower(), Config(v) if isinstance(v, dict) else v)

    def get(self, key, default=None):
        return getattr(self, key.lower(), default)

    def __repr__(self):
        return str(vars(self))
    
def scientific_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    return Config(raw)