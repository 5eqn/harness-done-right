"""
Configuration helpers for HDR runtime behavior.
"""

from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path("~/.hdr/config.yaml").expanduser()
DEFAULT_ANTHROPIC_MODEL = "claude-4.6-sonnet"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"


@dataclass(frozen=True, slots=True)
class VerifyConfig:
    """Resolved configuration used by Task.verify()."""

    anthropic_auth_token: str
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL


def config_template() -> str:
    """Return the starter config content written on first use."""
    return f"""# HDR verification config
# Fill in the Anthropic API token before calling Task.verify().
anthropic_auth_token: ""
anthropic_model: "{DEFAULT_ANTHROPIC_MODEL}"
anthropic_base_url: "{DEFAULT_ANTHROPIC_BASE_URL}"
"""


def _parse_config(content: str) -> dict[str, str]:
    parsed_config: dict[str, str] = {}

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        parsed_config[key.strip()] = value

    return parsed_config


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

    raw_config = _parse_config(CONFIG_PATH.read_text())
    config = VerifyConfig(
        anthropic_auth_token=raw_config.get("anthropic_auth_token", "").strip(),
        anthropic_model=raw_config.get(
            "anthropic_model", DEFAULT_ANTHROPIC_MODEL
        ).strip(),
        anthropic_base_url=raw_config.get(
            "anthropic_base_url", DEFAULT_ANTHROPIC_BASE_URL
        ).strip(),
    )

    if not config.anthropic_auth_token:
        raise EnvironmentError(
            f"anthropic_auth_token is empty in {CONFIG_PATH}. "
            "Please fill it in in ~/.hdr/config.yaml and rerun."
        )

    return config
