import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val


OUTPUT_DIR = Path(get("OUTPUT_DIR", "./output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
