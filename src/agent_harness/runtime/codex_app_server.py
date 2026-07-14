from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import signal
import socket
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_harness.config import AgentCommandConfig
from agent_harness.updates import is_newer_version

LOGGER = logging.getLogger(__name__)
DEFAULT_CODEX_APP_SERVER_URL = "ws://127.0.0.1:47684"
DEFAULT_CODEX_VERSION_CHECK_SECONDS = 2.0
CODEX_VERSION_TIMEOUT_SECONDS = 5.0
CODEX_VERSION_PATTERN = re.compile(
    r"(?<![0-9A-Za-z])(?P<version>\d+(?:\.\d+)+(?:[-+][0-9A-Za-z.-]+)?)"
)


class CodexAppServerError(RuntimeError):
    pass


ServerRequestHandler = Callable[[dict[str, Any]], Any]


class CodexAppServerClient:
    def __init__(
        self,
        url: str,
        timeout_seconds: float = 8.0,
        server_request_handler: ServerRequestHandler | None = None,
        listener_idle_timeout_seconds: float = 1800.0,
    ):
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.server_request_handler = server_request_handler
        self.listener_idle_timeout_seconds = listener_idle_timeout_seconds

    def send_to_thread(
        self,
        thread_id: str,
        text: str,
        cwd: Path | None = None,
        *,
        approval_policy: str | None = None,
        sandbox: str | None = None,
        sandbox_policy: dict[str, Any] | None = None,
    ) -> bool:
        if not thread_id or not text.strip():
            return False
        rpc = _JsonRpcWebSocket(self.url, timeout_seconds=self.timeout_seconds)
        try:
            rpc.connect()
            rpc.request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "slackgentic",
                        "title": "Slackgentic",
                        "version": "0.1.0",
                    },
                    "capabilities": {
                        "experimentalApi": True,
                        "optOutNotificationMethods": [
                            "command/exec/outputDelta",
                            "item/agentMessage/delta",
                            "item/plan/delta",
                            "item/fileChange/outputDelta",
                            "item/reasoning/summaryTextDelta",
                            "item/reasoning/textDelta",
                        ],
                    },
                },
                server_request_handler=self.server_request_handler,
            )
            rpc.notify("initialized")
            resume_params: dict[str, Any] = {
                "threadId": thread_id,
                "excludeTurns": True,
                "persistExtendedHistory": True,
            }
            if cwd is not None:
                resume_params["cwd"] = str(cwd)
            if approval_policy is not None:
                resume_params["approvalPolicy"] = approval_policy
            if sandbox is not None:
                resume_params["sandbox"] = sandbox
            rpc.request(
                "thread/resume",
                resume_params,
                server_request_handler=self.server_request_handler,
            )
            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "input": [
                    {
                        "type": "text",
                        "text": text,
                        "text_elements": [],
                    }
                ],
            }
            if approval_policy is not None:
                turn_params["approvalPolicy"] = approval_policy
            if sandbox_policy is not None:
                turn_params["sandboxPolicy"] = sandbox_policy
            rpc.request(
                "turn/start",
                turn_params,
                server_request_handler=self.server_request_handler,
            )
        except Exception:
            rpc.close()
            raise
        if self.server_request_handler is None:
            rpc.close()
        else:
            listener = threading.Thread(
                target=self._listen_for_server_requests,
                args=(rpc, thread_id),
                daemon=True,
                name=f"slackgentic-codex-app-server-{thread_id[:8]}",
            )
            listener.start()
        return True

    def _listen_for_server_requests(self, rpc: _JsonRpcWebSocket, thread_id: str) -> None:
        deadline = time.monotonic() + self.listener_idle_timeout_seconds
        try:
            while time.monotonic() < deadline:
                message = rpc.recv_json(timeout_seconds=1.0)
                if message is None:
                    continue
                if _is_server_request(message):
                    rpc.handle_server_request(message, self.server_request_handler)
                    deadline = time.monotonic() + self.listener_idle_timeout_seconds
                    continue
                if _is_terminal_thread_notification(message, thread_id):
                    return
        except Exception:
            LOGGER.debug("Codex app-server listener stopped", exc_info=True)
        finally:
            rpc.close()


@dataclass
class CodexAppServerManager:
    commands: AgentCommandConfig
    url: str = DEFAULT_CODEX_APP_SERVER_URL
    startup_timeout_seconds: float = 8.0

    def __post_init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> str | None:
        if not self.url or self.url.lower() == "off":
            return None
        if _health_ready(self.url):
            return self.url
        installation = resolve_codex_installation(self.commands.codex_binary)
        codex_binary = (
            str(installation.executable) if installation is not None else self.commands.codex_binary
        )
        try:
            self._process = subprocess.Popen(
                [
                    codex_binary,
                    "app-server",
                    "--listen",
                    self.url,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError:
            LOGGER.exception("failed to start Codex app-server")
            return None
        _drain_stream(self._process.stdout, logging.INFO)
        _drain_stream(self._process.stderr, logging.WARNING)
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                LOGGER.warning(
                    "Codex app-server exited early with code %s",
                    self._process.returncode,
                )
                return None
            if _health_ready(self.url):
                return self.url
            time.sleep(0.1)
        LOGGER.warning("Codex app-server did not become ready at %s", self.url)
        return None

    def close(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        _stop_codex_process(process)


@dataclass(frozen=True)
class CodexInstallation:
    executable: Path
    version: str
    signature: tuple[Any, ...]


def resolve_codex_installation(
    command: str,
    *,
    environ_path: str | None = None,
) -> CodexInstallation | None:
    """Resolve the newest usable Codex binary visible to the service.

    A service can outlive changes to the user's shell PATH, and Codex can be
    installed by more than one package manager. For a bare command name, check
    every matching executable on PATH and prefer the newest reported version.
    Explicit paths remain pinned to the requested installation.
    """

    best: CodexInstallation | None = None
    for candidate in _codex_candidate_paths(command, environ_path=environ_path):
        version = _codex_version(candidate)
        if version is None:
            continue
        installation = CodexInstallation(
            executable=candidate,
            version=version,
            signature=(*_codex_file_signature(candidate), version),
        )
        if best is None or is_newer_version(installation.version, best.version):
            best = installation
    return best


class CodexAppServerSupervisor:
    """Keep the app-server aligned with upgrades to the Codex executable."""

    def __init__(
        self,
        codex_binary: str = "codex",
        url: str = DEFAULT_CODEX_APP_SERVER_URL,
        *,
        check_interval_seconds: float = DEFAULT_CODEX_VERSION_CHECK_SECONDS,
        startup_timeout_seconds: float = 8.0,
    ):
        self.codex_binary = codex_binary
        self.url = url
        self.check_interval_seconds = max(0.1, check_interval_seconds)
        self.startup_timeout_seconds = startup_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._installation: CodexInstallation | None = None
        self._candidate_snapshot: tuple[Any, ...] | None = None

    def run(self, stop: threading.Event) -> int:
        try:
            self.reconcile(force=True)
            while not stop.wait(self.check_interval_seconds):
                process_missing = self._process is None
                process_exited = self._process is not None and self._process.poll() is not None
                snapshot = _codex_candidate_snapshot(self.codex_binary)
                if process_missing or process_exited or snapshot != self._candidate_snapshot:
                    self.reconcile(force=process_exited, snapshot=snapshot)
        finally:
            self.close()
        return 0

    def reconcile(
        self,
        *,
        force: bool = False,
        snapshot: tuple[Any, ...] | None = None,
    ) -> bool:
        if snapshot is None:
            snapshot = _codex_candidate_snapshot(self.codex_binary)
        self._candidate_snapshot = snapshot
        installation = resolve_codex_installation(self.codex_binary)
        if installation is None:
            self._candidate_snapshot = None
            LOGGER.error("could not find a usable Codex executable: %s", self.codex_binary)
            return False
        process_running = self._process is not None and self._process.poll() is None
        if (
            process_running
            and not force
            and self._installation is not None
            and installation.signature == self._installation.signature
        ):
            return True
        if process_running:
            LOGGER.info(
                "restarting Codex app-server after Codex changed from %s to %s",
                self._installation.version if self._installation else "unknown",
                installation.version,
            )
            self._stop_process()
        return self._start_process(installation)

    def close(self) -> None:
        self._stop_process()

    def _start_process(self, installation: CodexInstallation) -> bool:
        try:
            self._process = subprocess.Popen(
                [str(installation.executable), "app-server", "--listen", self.url],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError:
            LOGGER.exception("failed to start Codex app-server with %s", installation.executable)
            self._process = None
            return False
        self._installation = installation
        _drain_stream(self._process.stdout, logging.INFO)
        _drain_stream(self._process.stderr, logging.WARNING)
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                LOGGER.warning(
                    "Codex app-server %s exited early with code %s",
                    installation.version,
                    self._process.returncode,
                )
                return False
            if _health_ready(self.url):
                LOGGER.info(
                    "Codex app-server %s is ready at %s",
                    installation.version,
                    self.url,
                )
                return True
            time.sleep(0.1)
        LOGGER.warning("Codex app-server did not become ready at %s", self.url)
        self._stop_process()
        return False

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        _stop_codex_process(process)


def _stop_codex_process(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, OSError):
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, OSError):
            process.kill()
        process.wait(timeout=5)


def run_codex_app_server_supervisor(
    codex_binary: str = "codex",
    url: str = DEFAULT_CODEX_APP_SERVER_URL,
    *,
    check_interval_seconds: float = DEFAULT_CODEX_VERSION_CHECK_SECONDS,
) -> int:
    stop = threading.Event()
    supervisor = CodexAppServerSupervisor(
        codex_binary,
        url,
        check_interval_seconds=check_interval_seconds,
    )

    def request_stop(_signum=None, _frame=None) -> None:
        stop.set()

    previous_handlers: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.signal(sig, request_stop)
        except (ValueError, OSError):
            continue
    try:
        return supervisor.run(stop)
    finally:
        for sig, previous in previous_handlers.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError, TypeError):
                continue


def _codex_candidate_paths(
    command: str,
    *,
    environ_path: str | None = None,
) -> list[Path]:
    requested = Path(command).expanduser()
    if requested.is_absolute() or len(requested.parts) > 1:
        candidates = [requested]
    else:
        path_value = environ_path if environ_path is not None else os.environ.get("PATH", "")
        candidates = [
            Path(directory).expanduser() / command for directory in path_value.split(os.pathsep)
        ]
    usable: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        absolute = candidate.absolute()
        if absolute in seen:
            continue
        seen.add(absolute)
        usable.append(absolute)
    return usable


def _codex_version(executable: Path) -> str | None:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=CODEX_VERSION_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    match = CODEX_VERSION_PATTERN.search(f"{completed.stdout}\n{completed.stderr}")
    return match.group("version") if match else None


def _codex_file_signature(executable: Path) -> tuple[Any, ...]:
    try:
        link_stat = executable.lstat()
        target = executable.resolve(strict=True)
        target_stat = target.stat()
    except OSError:
        return (str(executable),)
    return (
        str(executable),
        link_stat.st_dev,
        link_stat.st_ino,
        link_stat.st_mtime_ns,
        link_stat.st_size,
        str(target),
        target_stat.st_dev,
        target_stat.st_ino,
        target_stat.st_mtime_ns,
        target_stat.st_size,
    )


def _codex_candidate_snapshot(command: str) -> tuple[Any, ...]:
    return tuple(_codex_file_signature(candidate) for candidate in _codex_candidate_paths(command))


class _JsonRpcWebSocket:
    def __init__(self, url: str, timeout_seconds: float):
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._next_id = 1
        self._socket: socket.socket | None = None

    def __enter__(self) -> _JsonRpcWebSocket:
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def request(
        self,
        method: str,
        params: Any,
        server_request_handler: ServerRequestHandler | None = None,
    ) -> Any:
        message_id = self._next_id
        self._next_id += 1
        self._send_json({"id": message_id, "method": method, "params": params})
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            message = self.recv_json(deadline - time.monotonic())
            if message is None:
                continue
            if _is_server_request(message):
                self.handle_server_request(message, server_request_handler)
                continue
            if message.get("id") != message_id:
                continue
            if "error" in message:
                error = message.get("error")
                if isinstance(error, dict):
                    raise CodexAppServerError(str(error.get("message") or error))
                raise CodexAppServerError(str(error))
            return message.get("result")
        raise CodexAppServerError(f"timed out waiting for {method}")

    def notify(self, method: str, params: Any | None = None) -> None:
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = params
        self._send_json(payload)

    def connect(self) -> None:
        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme != "ws" or not parsed.hostname or not parsed.port:
            raise CodexAppServerError(f"unsupported Codex app-server URL: {self.url}")
        sock = socket.create_connection(
            (parsed.hostname, parsed.port),
            timeout=self.timeout_seconds,
        )
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        sock.sendall(request.encode("ascii"))
        response = _read_http_response(sock)
        if not response.startswith("HTTP/1.1 101") and not response.startswith("HTTP/1.0 101"):
            sock.close()
            raise CodexAppServerError("Codex app-server did not accept websocket upgrade")
        expected_accept = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        ).decode("ascii")
        if f"sec-websocket-accept: {expected_accept.lower()}" not in response.lower():
            sock.close()
            raise CodexAppServerError("Codex app-server websocket accept header did not match")
        self._socket = sock

    def handle_server_request(
        self,
        message: dict[str, Any],
        server_request_handler: ServerRequestHandler | None,
    ) -> None:
        request_id = message.get("id")
        if server_request_handler is None:
            self._send_json(
                {
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"unsupported server request: {message.get('method')}",
                    },
                }
            )
            return
        try:
            result = server_request_handler(message)
        except Exception as exc:
            LOGGER.exception("failed to handle Codex app-server request")
            self._send_json(
                {
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": str(exc),
                    },
                }
            )
            return
        self._send_json({"id": request_id, "result": result})

    def _send_json(self, message: dict[str, Any]) -> None:
        self._send_text(json.dumps(message, separators=(",", ":")))

    def _send_text(self, text: str) -> None:
        if self._socket is None:
            raise CodexAppServerError("websocket is not connected")
        payload = text.encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.extend([0x80 | 126])
            header.extend(struct.pack("!H", length))
        else:
            header.extend([0x80 | 127])
            header.extend(struct.pack("!Q", length))
        header.extend(mask)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._socket.sendall(bytes(header) + masked)

    def recv_json(self, timeout_seconds: float) -> dict[str, Any] | None:
        frame = self._recv_frame(timeout_seconds)
        if frame is None:
            return None
        opcode, payload = frame
        if opcode == 8:
            raise CodexAppServerError("Codex app-server closed websocket")
        if opcode == 9:
            self._send_pong(payload)
            return None
        if opcode != 1:
            return None
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return decoded if isinstance(decoded, dict) else None

    def _recv_frame(self, timeout_seconds: float) -> tuple[int, bytes] | None:
        if self._socket is None:
            raise CodexAppServerError("websocket is not connected")
        self._socket.settimeout(max(0.01, timeout_seconds))
        try:
            header = _recv_exact(self._socket, 2)
        except TimeoutError:
            return None
        first, second = header
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", _recv_exact(self._socket, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", _recv_exact(self._socket, 8))[0]
        mask = _recv_exact(self._socket, 4) if masked else b""
        payload = _recv_exact(self._socket, length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, payload

    def _send_pong(self, payload: bytes) -> None:
        if self._socket is None:
            return
        header = bytearray([0x8A])
        length = len(payload)
        if length >= 126:
            return
        mask = os.urandom(4)
        header.append(0x80 | length)
        header.extend(mask)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._socket.sendall(bytes(header) + masked)


def _health_ready(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "ws" or not parsed.hostname or not parsed.port:
        return False
    health_url = f"http://{parsed.hostname}:{parsed.port}/healthz"
    try:
        with urllib.request.urlopen(health_url, timeout=0.5) as response:
            return 200 <= response.status < 300
    except OSError:
        return False


def _read_http_response(sock: socket.socket) -> str:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\r\n\r\n" in b"".join(chunks):
            break
    return b"".join(chunks).decode("iso-8859-1", errors="replace")


def _is_server_request(message: dict[str, Any]) -> bool:
    return "id" in message and isinstance(message.get("method"), str)


def _is_terminal_thread_notification(message: dict[str, Any], thread_id: str) -> bool:
    method = message.get("method")
    if method not in {"turn/completed", "thread/closed"}:
        return False
    params = message.get("params")
    if not isinstance(params, dict):
        return False
    return params.get("threadId") == thread_id


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise CodexAppServerError("websocket ended while reading frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _drain_stream(stream, level: int) -> None:
    if stream is None:
        return

    def run() -> None:
        for line in stream:
            LOGGER.log(level, "codex app-server: %s", line.rstrip())

    threading.Thread(target=run, daemon=True, name="slackgentic-codex-app-server-log").start()
