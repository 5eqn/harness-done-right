"""
Configuration helpers for HDR runtime behavior.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


CONFIG_PATH = Path("~/.hdr/config.yaml").expanduser()
DEFAULT_ANTHROPIC_MODEL = "claude-4.6-sonnet"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_VERIFY_CACHE_DIR = Path("/tmp/claude/hdr_verify_cache")


@dataclass(frozen=True, slots=True)
class VerifyConfig:
    """Resolved configuration used by BaseContract.verify()."""

    anthropic_auth_token: str
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    verify_cache_dir: Path = DEFAULT_VERIFY_CACHE_DIR


def config_template() -> str:
    """Return the starter config content written on first use."""
    return f"""# HDR verification config
# Fill in the Anthropic API token before calling BaseContract.verify().
anthropic_auth_token: ""
anthropic_model: "{DEFAULT_ANTHROPIC_MODEL}"
anthropic_base_url: "{DEFAULT_ANTHROPIC_BASE_URL}"
verify_cache_dir: "{DEFAULT_VERIFY_CACHE_DIR}"
"""


def load_verify_config() -> VerifyConfig:
    """
    Load verification settings from ~/.hdr/config.yaml.

    If the file does not exist, create a template and raise with guidance.
    """
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(config_template())
        raise EnvironmentError(
            f"HDR config created at {CONFIG_PATH}. "
            "Please fill in anthropic_auth_token in ~/.hdr/config.yaml and rerun."
        )

    raw_config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    if not isinstance(raw_config, dict):
        raise EnvironmentError(f"HDR config at {CONFIG_PATH} must be a YAML mapping.")

    config = VerifyConfig(
        anthropic_auth_token=str(raw_config.get("anthropic_auth_token", "")).strip(),
        anthropic_model=str(
            raw_config.get("anthropic_model", DEFAULT_ANTHROPIC_MODEL)
        ).strip(),
        anthropic_base_url=str(
            raw_config.get("anthropic_base_url", DEFAULT_ANTHROPIC_BASE_URL)
        ).strip(),
        verify_cache_dir=Path(
            str(raw_config.get("verify_cache_dir", DEFAULT_VERIFY_CACHE_DIR))
        ).expanduser(),
    )

    if not config.anthropic_auth_token:
        raise EnvironmentError(
            f"anthropic_auth_token is empty in {CONFIG_PATH}. "
            "Please fill it in in ~/.hdr/config.yaml and rerun."
        )

    return config
