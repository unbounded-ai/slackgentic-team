from __future__ import annotations

import os
import platform
import plistlib
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SERVICE_NAME = "slackgentic-team"
MACOS_LABEL = "com.slackgentic-team.daemon"
DEFAULT_SERVICE_PATHS = (
    "~/.local/bin",
    "~/.cargo/bin",
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)
TRANSIENT_PATH_MARKERS = (
    "/.codex/tmp/",
    "/var/run/com.apple.security.cryptexd/",
    "/node_modules/@openai/codex/node_modules/",
)


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    executable: Path
    args: list[str]
    working_directory: Path
    log_dir: Path

    @property
    def stdout_path(self) -> Path:
        return self.log_dir / f"{self.name}.out.log"

    @property
    def stderr_path(self) -> Path:
        return self.log_dir / f"{self.name}.err.log"


def build_service_spec(
    name: str = DEFAULT_SERVICE_NAME,
    executable: Path | None = None,
    working_directory: Path | None = None,
    config_file: Path | None = None,
) -> ServiceSpec:
    resolved_executable = executable or _current_slackgentic_executable()
    args = ["slack", "serve"]
    if config_file:
        args.extend(["--config-file", str(config_file.expanduser().resolve())])
    return ServiceSpec(
        name=name,
        executable=resolved_executable,
        args=args,
        working_directory=(working_directory or Path.cwd()).resolve(),
        log_dir=Path.home() / ".slackgentic-team" / "logs",
    )


def install_service(spec: ServiceSpec) -> Path:
    system = platform.system().lower()
    if system == "darwin":
        return _install_launchd(spec)
    if system == "linux":
        return _install_systemd(spec)
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def uninstall_service(name: str = DEFAULT_SERVICE_NAME) -> Path:
    system = platform.system().lower()
    if system == "darwin":
        return _uninstall_launchd()
    if system == "linux":
        return _uninstall_systemd(name)
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def service_status(name: str = DEFAULT_SERVICE_NAME) -> int:
    system = platform.system().lower()
    if system == "darwin":
        completed = subprocess.run(["launchctl", "list", MACOS_LABEL], check=False)
        return completed.returncode
    if system == "linux":
        completed = subprocess.run(["systemctl", "--user", "status", f"{name}.service"])
        return completed.returncode
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def render_service(spec: ServiceSpec) -> tuple[Path, str | bytes]:
    system = platform.system().lower()
    if system == "darwin":
        return _launchd_path(), render_launchd_plist(spec)
    if system == "linux":
        return _systemd_path(spec.name), render_systemd_unit(spec)
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def render_launchd_plist(spec: ServiceSpec) -> bytes:
    payload = {
        "Label": MACOS_LABEL,
        "ProgramArguments": [str(spec.executable), *spec.args],
        "WorkingDirectory": str(spec.working_directory),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(spec.stdout_path),
        "StandardErrorPath": str(spec.stderr_path),
        "EnvironmentVariables": {
            "PATH": service_environment_path(),
        },
    }
    return plistlib.dumps(payload, sort_keys=True)


def render_systemd_unit(spec: ServiceSpec) -> str:
    command = shlex.join([str(spec.executable), *spec.args])
    return "\n".join(
        [
            "[Unit]",
            "Description=Slackgentic Team Slack daemon",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={spec.working_directory}",
            f"ExecStart={command}",
            _systemd_environment_line("PATH", service_environment_path()),
            "Restart=always",
            "RestartSec=5",
            f"StandardOutput=append:{spec.stdout_path}",
            f"StandardError=append:{spec.stderr_path}",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def _install_launchd(spec: ServiceSpec) -> Path:
    path, content = render_service(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    assert isinstance(content, bytes)
    path.write_bytes(content)
    subprocess.run(["launchctl", "unload", str(path)], check=False)
    subprocess.run(["launchctl", "load", str(path)], check=True)
    return path


def _uninstall_launchd() -> Path:
    path = _launchd_path()
    subprocess.run(["launchctl", "unload", str(path)], check=False)
    if path.exists():
        path.unlink()
    return path


def _install_systemd(spec: ServiceSpec) -> Path:
    path, content = render_service(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    assert isinstance(content, str)
    path.write_text(content)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", f"{spec.name}.service"], check=True)
    return path


def _uninstall_systemd(name: str) -> Path:
    path = _systemd_path(name)
    subprocess.run(["systemctl", "--user", "disable", "--now", f"{name}.service"], check=False)
    if path.exists():
        path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    return path


def _launchd_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{MACOS_LABEL}.plist"


def _systemd_path(name: str) -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{name}.service"


def _current_slackgentic_executable() -> Path:
    current = Path(sys.argv[0])
    if current.exists():
        return current.resolve()
    found = shutil.which("slackgentic")
    if found:
        return Path(found).resolve()
    raise RuntimeError("Could not locate slackgentic executable")


def service_environment_path(environ_path: str | None = None) -> str:
    paths: list[str] = []
    for command in ("codex", "claude"):
        found = shutil.which(command)
        if found:
            paths.append(str(Path(found).parent))
    paths.extend(DEFAULT_SERVICE_PATHS)

    source_path = environ_path if environ_path is not None else os.environ.get("PATH", "")
    paths.extend(source_path.split(os.pathsep))

    stable_paths: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        if not raw_path:
            continue
        expanded = str(Path(raw_path).expanduser())
        if _is_transient_path(expanded) or expanded in seen:
            continue
        seen.add(expanded)
        stable_paths.append(expanded)
    return os.pathsep.join(stable_paths)


def _is_transient_path(path: str) -> bool:
    return any(marker in path for marker in TRANSIENT_PATH_MARKERS)


def _systemd_environment_line(name: str, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'Environment="{name}={escaped}"'
