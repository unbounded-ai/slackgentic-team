from __future__ import annotations

import os
import platform
import plistlib
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SERVICE_NAME = "slackgentic-team"
MACOS_LABEL = "com.slackgentic-team.daemon"
MACOS_CODEX_APP_SERVER_LABEL = "com.slackgentic-team.codex-app-server"
CODEX_APP_SERVER_SERVICE_SUFFIX = "codex-app-server"
DEFAULT_CODEX_APP_SERVER_URL = "ws://127.0.0.1:47684"
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
SAFE_SERVICE_ENVIRONMENT = {
    "SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART": "false",
}
FALSE_ENV_VALUES = {"0", "false", "no", "off"}


class UnsafeServiceRestartError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    executable: Path
    args: list[str]
    working_directory: Path
    log_dir: Path
    label: str = MACOS_LABEL
    description: str = "Slackgentic Team Slack daemon"
    environment: dict[str, str] = field(default_factory=lambda: dict(SAFE_SERVICE_ENVIRONMENT))

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


def build_codex_app_server_service_spec(
    name: str = DEFAULT_SERVICE_NAME,
    executable: Path | None = None,
    working_directory: Path | None = None,
    url: str = DEFAULT_CODEX_APP_SERVER_URL,
) -> ServiceSpec:
    resolved_executable = executable or _current_codex_executable()
    return ServiceSpec(
        name=_codex_app_server_service_name(name),
        executable=resolved_executable,
        args=["app-server", "--listen", url],
        working_directory=(working_directory or Path.cwd()).resolve(),
        log_dir=Path.home() / ".slackgentic-team" / "logs",
        label=MACOS_CODEX_APP_SERVER_LABEL,
        description="Slackgentic Team Codex app-server",
        environment={},
    )


def install_service(spec: ServiceSpec) -> Path:
    system = platform.system().lower()
    if system == "darwin":
        return _install_launchd(spec)
    if system == "linux":
        return _install_systemd(spec)
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def install_services(specs: list[ServiceSpec]) -> list[Path]:
    system = platform.system().lower()
    if system == "darwin":
        return _install_launchd_services(specs)
    if system == "linux":
        return _install_systemd_services(specs)
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def uninstall_service(name: str = DEFAULT_SERVICE_NAME) -> Path:
    system = platform.system().lower()
    if system == "darwin":
        return _uninstall_launchd(MACOS_LABEL)
    if system == "linux":
        return _uninstall_systemd(name)
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def uninstall_services(
    name: str = DEFAULT_SERVICE_NAME,
    *,
    include_codex_app_server: bool = True,
) -> list[Path]:
    system = platform.system().lower()
    if system == "darwin":
        labels = [MACOS_LABEL]
        if include_codex_app_server:
            labels.append(MACOS_CODEX_APP_SERVER_LABEL)
        return [_uninstall_launchd(label) for label in labels]
    if system == "linux":
        names = [name]
        if include_codex_app_server:
            names.append(_codex_app_server_service_name(name))
        return [_uninstall_systemd(service_name) for service_name in names]
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def start_services(
    name: str = DEFAULT_SERVICE_NAME,
    *,
    include_codex_app_server: bool = True,
) -> list[int]:
    system = platform.system().lower()
    if system == "darwin":
        return [
            _start_launchd(label)
            for label in _service_labels(
                include_codex_app_server=include_codex_app_server,
                codex_first=True,
            )
        ]
    if system == "linux":
        return [
            _start_systemd(service_name)
            for service_name in _service_names(
                name,
                include_codex_app_server=include_codex_app_server,
                codex_first=True,
            )
        ]
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def restart_service(name: str = DEFAULT_SERVICE_NAME, *, force: bool = False) -> int:
    system = platform.system().lower()
    if system == "darwin":
        return _restart_launchd(force=force)
    if system == "linux":
        return _restart_systemd(name, force=force)
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def service_statuses(
    name: str = DEFAULT_SERVICE_NAME,
    *,
    include_codex_app_server: bool = True,
) -> list[int]:
    system = platform.system().lower()
    if system == "darwin":
        labels = [MACOS_LABEL]
        if include_codex_app_server:
            labels.append(MACOS_CODEX_APP_SERVER_LABEL)
        return [_launchd_status(label) for label in labels]
    if system == "linux":
        names = [name]
        if include_codex_app_server:
            names.append(_codex_app_server_service_name(name))
        return [_systemd_status(service_name) for service_name in names]
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def service_status(name: str = DEFAULT_SERVICE_NAME) -> int:
    system = platform.system().lower()
    if system == "darwin":
        return _launchd_status(MACOS_LABEL)
    if system == "linux":
        return _systemd_status(name)
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def render_services(specs: list[ServiceSpec]) -> list[tuple[Path, str | bytes]]:
    return [render_service(spec) for spec in specs]


def render_service(spec: ServiceSpec) -> tuple[Path, str | bytes]:
    system = platform.system().lower()
    if system == "darwin":
        return _launchd_path(spec.label), render_launchd_plist(spec)
    if system == "linux":
        return _systemd_path(spec.name), render_systemd_unit(spec)
    raise RuntimeError(f"Unsupported service platform: {platform.system()}")


def render_launchd_plist(spec: ServiceSpec) -> bytes:
    environment = {
        "PATH": service_environment_path(),
        **spec.environment,
    }
    payload = {
        "Label": spec.label,
        "ProgramArguments": [str(spec.executable), *spec.args],
        "WorkingDirectory": str(spec.working_directory),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(spec.stdout_path),
        "StandardErrorPath": str(spec.stderr_path),
        "EnvironmentVariables": environment,
    }
    return plistlib.dumps(payload, sort_keys=True)


def render_systemd_unit(spec: ServiceSpec) -> str:
    command = shlex.join([str(spec.executable), *spec.args])
    environment_lines = [
        _systemd_environment_line("PATH", service_environment_path()),
        *[
            _systemd_environment_line(name, value)
            for name, value in sorted(spec.environment.items())
        ],
    ]
    return "\n".join(
        [
            "[Unit]",
            f"Description={spec.description}",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={spec.working_directory}",
            f"ExecStart={command}",
            *environment_lines,
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
    _bootout_launchd(spec.label)
    _bootstrap_launchd(spec.label)
    _settle_launchd_services([spec])
    return path


def _install_launchd_services(specs: list[ServiceSpec]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        path, content = render_service(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        assert isinstance(content, bytes)
        path.write_bytes(content)
        paths.append(path)

    for spec in specs:
        _bootout_launchd(spec.label)
    ordered_specs = _codex_first_specs(specs)
    for spec in ordered_specs:
        _bootstrap_launchd(spec.label)
    _settle_launchd_services(ordered_specs)
    return paths


def _uninstall_launchd(label: str) -> Path:
    path = _launchd_path(label)
    _bootout_launchd(label)
    if path.exists():
        path.unlink()
    return path


def _start_launchd(label: str) -> int:
    path = _launchd_path(label)
    if path.exists():
        _bootstrap_launchd(label, ignore_already_loaded=True)
    completed = subprocess.run(
        ["launchctl", "kickstart", _launchd_target(label)],
        check=False,
    )
    return completed.returncode


def _restart_launchd(*, force: bool = False) -> int:
    if not force:
        issue = _launchd_restart_safety_issue(_launchd_path(MACOS_LABEL))
        if issue:
            raise UnsafeServiceRestartError(issue)
    completed = subprocess.run(
        ["launchctl", "kickstart", "-k", _launchd_target(MACOS_LABEL)],
        check=False,
    )
    return completed.returncode


def _install_systemd(spec: ServiceSpec) -> Path:
    path, content = render_service(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    assert isinstance(content, str)
    path.write_text(content)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", f"{spec.name}.service"], check=True)
    return path


def _install_systemd_services(specs: list[ServiceSpec]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        path, content = render_service(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        assert isinstance(content, str)
        path.write_text(content)
        paths.append(path)

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    for spec in specs:
        subprocess.run(["systemctl", "--user", "stop", f"{spec.name}.service"], check=False)
    for spec in _codex_first_specs(specs):
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", f"{spec.name}.service"],
            check=True,
        )
    return paths


def _uninstall_systemd(name: str) -> Path:
    path = _systemd_path(name)
    subprocess.run(["systemctl", "--user", "disable", "--now", f"{name}.service"], check=False)
    if path.exists():
        path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    return path


def _start_systemd(name: str) -> int:
    completed = subprocess.run(
        ["systemctl", "--user", "start", f"{name}.service"],
        check=False,
    )
    return completed.returncode


def _restart_systemd(name: str, *, force: bool = False) -> int:
    if not force:
        issue = _systemd_restart_safety_issue(_systemd_path(name))
        if issue:
            raise UnsafeServiceRestartError(issue)
    completed = subprocess.run(
        ["systemctl", "--user", "restart", f"{name}.service"],
        check=False,
    )
    return completed.returncode


def _bootout_launchd(label: str) -> None:
    completed = subprocess.run(
        ["launchctl", "bootout", _launchd_target(label)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0 or _launchd_service_missing(completed):
        return
    raise RuntimeError(_launchctl_failure_message("bootout", completed))


def _bootstrap_launchd(label: str, *, ignore_already_loaded: bool = False) -> None:
    completed = subprocess.run(
        ["launchctl", "bootstrap", _launchd_domain(), str(_launchd_path(label))],
        check=False,
        capture_output=True,
        text=True,
    )
    if _launchd_bootstrap_succeeded(label, completed, ignore_already_loaded):
        return

    retry = _retry_bootstrap_launchd(label)
    if _launchd_bootstrap_succeeded(label, retry, ignore_already_loaded=True):
        return

    raise RuntimeError(_launchctl_failure_message("bootstrap", retry or completed))


def _retry_bootstrap_launchd(label: str) -> subprocess.CompletedProcess[str] | None:
    time.sleep(0.25)
    _bootout_launchd(label)
    completed = subprocess.run(
        ["launchctl", "bootstrap", _launchd_domain(), str(_launchd_path(label))],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed


def _launchd_bootstrap_succeeded(
    label: str,
    completed: subprocess.CompletedProcess[str] | None,
    ignore_already_loaded: bool,
) -> bool:
    if completed is None:
        return False
    if completed.returncode == 0:
        return True
    if ignore_already_loaded and _launchd_service_already_loaded(completed):
        return True
    return _launchd_service_is_loaded(label)


def _settle_launchd_services(specs: list[ServiceSpec]) -> None:
    time.sleep(0.25)
    for spec in specs:
        _start_launchd(spec.label)
    time.sleep(0.25)
    missing = [spec.label for spec in specs if not _launchd_service_is_loaded(spec.label)]
    if missing:
        raise RuntimeError(f"launchd services did not stay loaded: {', '.join(missing)}")


def _launchd_domain() -> str:
    return f"gui/{os.getuid()}"


def _launchd_target(label: str) -> str:
    return f"{_launchd_domain()}/{label}"


def _launchd_path(label: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def _systemd_path(name: str) -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{name}.service"


def _codex_app_server_service_name(name: str) -> str:
    return f"{name}-{CODEX_APP_SERVER_SERVICE_SUFFIX}"


def _service_labels(
    *,
    include_codex_app_server: bool = True,
    codex_first: bool = False,
) -> list[str]:
    labels = [MACOS_LABEL]
    if include_codex_app_server:
        labels.append(MACOS_CODEX_APP_SERVER_LABEL)
    if codex_first:
        labels.reverse()
    return labels


def _service_names(
    name: str,
    *,
    include_codex_app_server: bool = True,
    codex_first: bool = False,
) -> list[str]:
    names = [name]
    if include_codex_app_server:
        names.append(_codex_app_server_service_name(name))
    if codex_first:
        names.reverse()
    return names


def _codex_first_specs(specs: list[ServiceSpec]) -> list[ServiceSpec]:
    return sorted(
        specs,
        key=lambda spec: (
            0
            if spec.label == MACOS_CODEX_APP_SERVER_LABEL
            or spec.name.endswith(f"-{CODEX_APP_SERVER_SERVICE_SUFFIX}")
            else 1
        ),
    )


def _current_slackgentic_executable() -> Path:
    current = Path(sys.argv[0])
    if current.exists():
        return current.resolve()
    found = shutil.which("slackgentic")
    if found:
        return Path(found).resolve()
    raise RuntimeError("Could not locate slackgentic executable")


def _current_codex_executable() -> Path:
    found = shutil.which("codex")
    if found:
        return Path(found).resolve()
    return Path("codex")


def _launchd_status(label: str) -> int:
    completed = subprocess.run(["launchctl", "list", label], check=False)
    return completed.returncode


def _systemd_status(name: str) -> int:
    completed = subprocess.run(["systemctl", "--user", "status", f"{name}.service"])
    return completed.returncode


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


def _launchd_restart_safety_issue(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = plistlib.loads(path.read_bytes())
    except Exception as exc:
        return f"could not inspect installed launchd plist at {path}: {exc}"
    env = payload.get("EnvironmentVariables")
    if not isinstance(env, dict):
        env = {}
    value = env.get("SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART")
    if _is_false_env_value(value):
        return None
    return _unsafe_restart_message(path)


def _systemd_restart_safety_issue(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        values = _systemd_environment_values(path.read_text())
    except Exception as exc:
        return f"could not inspect installed systemd unit at {path}: {exc}"
    if _is_false_env_value(values.get("SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART")):
        return None
    return _unsafe_restart_message(path)


def _unsafe_restart_message(path: Path) -> str:
    return (
        f"{path} does not set SLACKGENTIC_CODEX_APP_SERVER_AUTOSTART=false. "
        "Restarting this service may stop the Codex app-server and disconnect "
        "`codex --remote` sessions. Reinstall the service with the current "
        "`slackgentic service install` first, or re-run restart with --force "
        "after accepting that risk."
    )


def _is_false_env_value(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in FALSE_ENV_VALUES


def _launchd_service_missing(completed: subprocess.CompletedProcess[str]) -> bool:
    output = _launchctl_output(completed).lower()
    return (
        completed.returncode == 3
        or "no such process" in output
        or "could not find specified service" in output
    )


def _launchd_service_already_loaded(completed: subprocess.CompletedProcess[str]) -> bool:
    output = _launchctl_output(completed).lower()
    return (
        completed.returncode == 5
        or "already bootstrapped" in output
        or "service is already loaded" in output
    )


def _launchd_service_is_loaded(label: str) -> bool:
    completed = subprocess.run(
        ["launchctl", "print", _launchd_target(label)],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _launchctl_failure_message(
    action: str,
    completed: subprocess.CompletedProcess[str],
) -> str:
    detail = _launchctl_output(completed)
    message = f"launchctl {action} failed with exit code {completed.returncode}"
    if detail:
        message = f"{message}: {detail}"
    return message


def _launchctl_output(completed: subprocess.CompletedProcess[str]) -> str:
    parts = [
        part.strip()
        for part in (completed.stderr, completed.stdout)
        if isinstance(part, str) and part.strip()
    ]
    return "\n".join(parts)


def _systemd_environment_values(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line.startswith("Environment="):
            continue
        payload = line.split("=", 1)[1]
        for assignment in shlex.split(payload):
            if "=" not in assignment:
                continue
            name, value = assignment.split("=", 1)
            values[name] = value
    return values


def _systemd_environment_line(name: str, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'Environment="{name}={escaped}"'
