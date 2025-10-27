import os
from typing import List


def load_api_keys() -> set[str]:
    """Load API keys from environment variable or file."""
    keys = set()

    # Load from environment variable (comma-separated)
    env_keys = os.environ.get("MLX_API_KEYS", "")
    if env_keys:
        keys.update(k.strip() for k in env_keys.split(",") if k.strip())

    # Load from file (one key per line)
    api_key_file = os.environ.get("MLX_API_KEY_FILE", ".api_keys")
    if os.path.exists(api_key_file):
        with open(api_key_file, "r") as f:
            keys.update(
                line.strip() for line in f if line.strip() and not line.startswith("#")
            )

    return keys


def check_stop_sequences(text: str, stop_sequences: List[str]) -> tuple[bool, str]:
    """Check if text contains any stop sequences and trim if found."""
    for stop_seq in stop_sequences:
        if stop_seq in text:
            # Find the position and trim
            pos = text.find(stop_seq)
            return True, text[:pos]
    return False, text
