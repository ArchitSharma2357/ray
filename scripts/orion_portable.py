#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
FRONTEND_SCRIPT = SCRIPTS_DIR / "orion_frontend.py"
CONFIG_FILE = ROOT_DIR / "config.json"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
VENV_DIR = ROOT_DIR / ".venv"
BUILD_DIR = ROOT_DIR / "build"
OUT_DIR = ROOT_DIR / "out"
CACHE_DIR = OUT_DIR / "cache"
TEMP_DIR = OUT_DIR / "temp"
IMPORTS_DIR = OUT_DIR / "imports"
ENV_SH_FILE = SCRIPTS_DIR / "orion_env.sh"

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8092,
    "auto_open_browser": True,
}


class SetupError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[orion-portable] {message}", flush=True)


def _command_display(cmd: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _normalized_path(path: Path) -> str:
    text = str(path.resolve()).replace("\\", "/").rstrip("/")
    if os.name == "nt":
        return text.lower()
    return text


def in_virtualenv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def venv_python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def resolve_python_for_dependencies() -> str:
    if in_virtualenv():
        return sys.executable

    venv_python = venv_python_path()
    if not venv_python.exists():
        log(f"Creating project virtual environment: {VENV_DIR}")
        run_checked([sys.executable, "-m", "venv", str(VENV_DIR)])
    if not venv_python.exists():
        raise SetupError(
            f"Virtual environment python was not created as expected: {venv_python}"
        )
    return str(venv_python)


def resolve_python_for_runtime() -> str:
    if in_virtualenv():
        return sys.executable
    venv_python = venv_python_path()
    candidates: list[str] = []
    if venv_python.exists():
        candidates.append(str(venv_python))
    candidates.append(sys.executable)

    for candidate in candidates:
        probe = [
            candidate,
            "-c",
            "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('PIL') else 1)",
        ]
        result = subprocess.run(probe, cwd=str(ROOT_DIR), check=False)
        if result.returncode == 0:
            return candidate
    return candidates[0]


def reset_stale_build_cache() -> None:
    cache_file = BUILD_DIR / "CMakeCache.txt"
    if not cache_file.exists():
        return

    try:
        lines = cache_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return

    source_line = next(
        (line for line in lines if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")),
        None,
    )
    if not source_line:
        return

    configured_source = source_line.split("=", 1)[1].strip()
    if not configured_source:
        return

    if _normalized_path(Path(configured_source)) == _normalized_path(ROOT_DIR):
        return

    log("Detected stale CMake cache from a different source path; resetting cache.")
    try:
        cache_file.unlink()
    except OSError:
        pass
    shutil.rmtree(BUILD_DIR / "CMakeFiles", ignore_errors=True)


def run_checked(cmd: list[str], *, cwd: Path = ROOT_DIR) -> None:
    log(f"$ {_command_display(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_python_version() -> None:
    if sys.version_info < (3, 9):
        raise SetupError(
            "Python 3.9+ is required. Install Python 3.9 or newer and re-run install."
        )


def load_config() -> Dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    if not CONFIG_FILE.exists():
        return config

    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"Failed to read config file {CONFIG_FILE}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SetupError(f"Config file {CONFIG_FILE} must contain a JSON object.")

    if isinstance(raw.get("host"), str) and raw["host"].strip():
        config["host"] = raw["host"].strip()
    if isinstance(raw.get("port"), int):
        config["port"] = raw["port"]
    if isinstance(raw.get("auto_open_browser"), bool):
        config["auto_open_browser"] = raw["auto_open_browser"]

    return config


def ensure_required_directories() -> None:
    for path in (OUT_DIR, CACHE_DIR, TEMP_DIR, IMPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_env_files(host: str, port: int) -> None:
    ENV_SH_FILE.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f'export ORION_FRONTEND_HOST="${{ORION_FRONTEND_HOST:-{host}}}"',
                f'export ORION_FRONTEND_PORT="${{ORION_FRONTEND_PORT:-{port}}}"',
                'export ORION_DEFAULT_BACKEND="${ORION_DEFAULT_BACKEND:-auto}"',
                "",
            ]
        ),
        encoding="utf-8",
    )



def resolve_renderer_binary(required: bool = True) -> Optional[Path]:
    candidates: list[Path] = []

    override = os.environ.get("ORION_BINARY_PATH", "").strip()
    if override:
        custom = Path(override).expanduser()
        if not custom.is_absolute():
            custom = (ROOT_DIR / custom).resolve()
        else:
            custom = custom.resolve()
        candidates.append(custom)

    candidates.extend(
        [
            BUILD_DIR / "orion_raytracer",
            BUILD_DIR / "orion_raytracer.exe",
            BUILD_DIR / "Release" / "orion_raytracer",
            BUILD_DIR / "Release" / "orion_raytracer.exe",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    if not required:
        return None

    searched = "\n".join(f"  - {item}" for item in candidates)
    raise SetupError(
        "Renderer binary not found.\n"
        f"Searched:\n{searched}\n"
        "Run install first: python scripts/orion_portable.py install"
    )


def detect_os() -> str:
    system = platform.system().strip()
    normalized = system.lower()
    if normalized.startswith("win"):
        return "windows"
    if normalized == "linux":
        return "linux"
    return normalized


def ensure_tool(name: str, help_text: str) -> None:
    if shutil.which(name) is None:
        raise SetupError(f"Missing required tool `{name}`. {help_text}")


def choose_cmake_generator(os_name: str) -> Optional[str]:
    ensure_tool("cmake", "Install CMake and ensure it is available in PATH.")

    generator_override = os.environ.get("ORION_CMAKE_GENERATOR", "").strip()
    if generator_override:
        log(f"Using CMake generator from ORION_CMAKE_GENERATOR: {generator_override}")
        return generator_override

    if shutil.which("ninja"):
        return "Ninja"

    if os_name == "windows":
        # On Windows, allow CMake to auto-detect an installed Visual Studio generator.
        return None

    if shutil.which("make") is None:
        raise SetupError(
            "No build generator found. Install `make` or `ninja` before running install."
        )
    return None


def print_gpu_toolchain_status() -> None:
    nvcc = shutil.which("nvcc")
    smi = shutil.which("nvidia-smi")
    if nvcc:
        log(f"CUDA toolkit detected: {nvcc}")
        return
    if smi:
        log("NVIDIA runtime detected, but CUDA toolkit not found. CPU build may be used.")
        return
    log("No CUDA toolkit detected. Build will use CPU backend unless CUDA tools are available.")


def install_python_dependencies(skip_pip: bool) -> None:
    if skip_pip:
        log("Skipping Python dependency install (--skip-pip).")
        return
    if not REQUIREMENTS_FILE.exists():
        raise SetupError(f"Missing requirements file: {REQUIREMENTS_FILE}")

    python_bin = resolve_python_for_dependencies()
    run_checked([python_bin, "-m", "pip", "--version"])
    try:
        run_checked(
            [
                python_bin,
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                "-r",
                str(REQUIREMENTS_FILE),
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise SetupError(
            "Failed to install Python dependencies. "
            "If your environment blocks package downloads, run install with --skip-pip "
            "after pre-installing dependencies."
        ) from exc


def configure_and_build(generator: Optional[str]) -> None:
    reset_stale_build_cache()
    configure_cmd = [
        "cmake",
        "-S",
        str(ROOT_DIR),
        "-B",
        str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if generator:
        configure_cmd.extend(["-G", generator])
    run_checked(configure_cmd)

    build_cmd = [
        "cmake",
        "--build",
        str(BUILD_DIR),
        "--config",
        "Release",
        "--parallel",
    ]
    run_checked(build_cmd)


def fix_permissions() -> None:
    if os.name == "nt":
        return

    candidates = [
        ROOT_DIR / "install.sh",
        ROOT_DIR / "run.sh",
        ROOT_DIR / "run_frontend.sh",
        SCRIPTS_DIR / "install.sh",
        SCRIPTS_DIR / "run.sh",
        SCRIPTS_DIR / "orion_portable.py",
        ENV_SH_FILE,
    ]
    for candidate in candidates:
        if candidate.exists():
            mode = candidate.stat().st_mode
            candidate.chmod(mode | 0o111)

    binary = resolve_renderer_binary(required=False)
    if binary is not None:
        mode = binary.stat().st_mode
        binary.chmod(mode | 0o111)


def parse_port(port_value: object) -> int:
    try:
        port = int(port_value)
    except (TypeError, ValueError) as exc:
        raise SetupError(f"Invalid port `{port_value}`. Port must be an integer.") from exc
    if port < 1 or port > 65535:
        raise SetupError(f"Invalid port `{port}`. Valid range is 1-65535.")
    return port


def run_install(skip_pip: bool) -> None:
    # Install pipeline: validate toolchain, prepare runtime dirs/env, then build binaries.
    ensure_python_version()
    os_name = detect_os()
    log(f"Detected OS: {platform.system()}")
    generator = choose_cmake_generator(os_name)
    print_gpu_toolchain_status()

    config = load_config()
    host = str(config["host"])
    port = parse_port(config["port"])

    ensure_required_directories()
    write_env_files(host, port)
    install_python_dependencies(skip_pip=skip_pip)
    configure_and_build(generator)
    binary = resolve_renderer_binary(required=True)
    fix_permissions()

    log(f"Renderer binary ready: {binary}")
    log(f"Frontend output directory: {OUT_DIR}")
    log("Install complete.")


def start_backend_process(host: str, port: int, binary: Path) -> subprocess.Popen:
    if not FRONTEND_SCRIPT.exists():
        raise SetupError(f"Frontend server script missing: {FRONTEND_SCRIPT}")

    env = os.environ.copy()
    env.setdefault("ORION_FRONTEND_HOST", host)
    env.setdefault("ORION_FRONTEND_PORT", str(port))
    env["ORION_BINARY_PATH"] = str(binary)

    runtime_python = resolve_python_for_runtime()
    command = [
        runtime_python,
        str(FRONTEND_SCRIPT),
        "--host",
        host,
        "--port",
        str(port),
    ]
    log(f"$ {_command_display(command)}")
    return subprocess.Popen(command, cwd=str(ROOT_DIR), env=env)


def run_browser(url: str) -> None:
    try:
        opened = webbrowser.open(url, new=0, autoraise=True)
    except Exception as exc:  # broad: browser availability differs by platform
        log(f"Could not open browser automatically: {exc}")
        return
    if not opened:
        log("Browser did not open automatically. Open the URL manually.")


def ensure_port_available(host: str, port: int) -> None:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        raise SetupError(
            f"Cannot open a TCP socket on this system. Networking appears unavailable. ({exc})"
        ) from exc
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise SetupError(
            f"Cannot bind to {host}:{port}. Choose another port or stop the process using it. ({exc})"
        ) from exc
    finally:
        probe.close()


def run_backend(host: str, port: int, auto_open_browser: bool) -> None:
    # Runtime pipeline: ensure prerequisites, launch backend, then keep process supervised.
    ensure_python_version()
    ensure_required_directories()
    ensure_port_available(host, port)

    binary = resolve_renderer_binary(required=False)
    if binary is None:
        log("Renderer binary not found. Building before startup...")
        generator = choose_cmake_generator(detect_os())
        configure_and_build(generator)
        binary = resolve_renderer_binary(required=True)

    url = f"http://{host}:{port}"
    log("Starting Orion Studio web server...")
    log("Frontend assets are served by the backend HTTP server.")
    log("Render backend mode defaults to auto (GPU when available, CPU fallback).")
    log(f"Renderer binary: {binary}")
    log(f"URL: {url}")

    proc: Optional[subprocess.Popen] = None
    try:
        proc = start_backend_process(host, port, binary)
        time.sleep(1.0)
        if proc.poll() is not None:
            raise SetupError(
                f"Backend server exited early with code {proc.returncode}. Check logs above."
            )
        if auto_open_browser:
            run_browser(url)
        return_code = proc.wait()
        if return_code != 0:
            raise SetupError(f"Backend server exited with code {return_code}.")
    except KeyboardInterrupt:
        log("Shutdown requested, stopping backend...")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-platform installer/runner for Orion Studio web app."
    )
    subparsers = parser.add_subparsers(dest="command")

    install_cmd = subparsers.add_parser("install", help="Install dependencies and build binaries.")
    install_cmd.add_argument(
        "--skip-pip",
        action="store_true",
        help="Skip Python pip dependency installation.",
    )

    run_cmd = subparsers.add_parser("run", help="Run Orion Studio web app.")
    run_cmd.add_argument("--host", help="Host bind address (default: config/env).")
    run_cmd.add_argument("--port", type=int, help="Port (default: config/env).")
    run_cmd.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "install":
            run_install(skip_pip=bool(args.skip_pip))
            return 0

        if args.command == "run":
            # CLI args override env/config for convenient local customization.
            config = load_config()
            host = (
                str(args.host).strip()
                if args.host
                else str(os.environ.get("ORION_FRONTEND_HOST", config["host"])).strip()
            )
            if not host:
                raise SetupError("Host cannot be empty.")
            port = (
                int(args.port)
                if args.port is not None
                else parse_port(os.environ.get("ORION_FRONTEND_PORT", config["port"]))
            )
            auto_open_browser = (
                not bool(args.no_browser)
                and bool(config.get("auto_open_browser", True))
            )
            run_backend(host, port, auto_open_browser=auto_open_browser)
            return 0

        raise SetupError(f"Unknown command: {args.command}")
    except subprocess.CalledProcessError as exc:
        log(f"Command failed with exit code {exc.returncode}: {_command_display(exc.cmd)}")
        return exc.returncode if exc.returncode else 1
    except SetupError as exc:
        log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
