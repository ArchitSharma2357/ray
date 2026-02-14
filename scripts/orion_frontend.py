#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import io
import json
import math
import mimetypes
import os
import struct
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "build"
OUT_DIR = PROJECT_ROOT / "out"
CACHE_DIR = OUT_DIR / "cache"
TEMP_DIR = OUT_DIR / "temp"
OBJ_IMPORTS_DIR = OUT_DIR / "imports"
HTML_PATH = PROJECT_ROOT / "frontend" / "index.html"

MODE_RENDER = "render"
MODE_LIVE = "live"
BACKENDS = {"auto", "cpu", "gpu"}
DEFAULT_DEMO = "scene_editor"
DEMOS: List[Dict[str, object]] = [
    {
        "id": "scene_editor",
        "label": "Editable Environment",
        "description": "Live editable primitive scene (sphere, cube, plane, light).",
        "gpu_compatible": True,
    },
]
DEMO_IDS = {str(item["id"]) for item in DEMOS}
LIVE_PROFILES: Dict[str, Dict[str, object]] = {
    "speed": {
        "live_samples": 24,
        "live_depth": 10,
        "live_interval_ms": 24,
        "live_preview_scale": 0.84,
        "live_preview_samples": 8,
        "live_preview_depth": 8,
        "live_refine_every": 7,
    },
    "balanced": {
        "live_samples": 64,
        "live_depth": 14,
        "live_interval_ms": 50,
        "live_preview_scale": 0.92,
        "live_preview_samples": 20,
        "live_preview_depth": 12,
        "live_refine_every": 3,
    },
    "ultra": {
        "live_samples": 128,
        "live_depth": 18,
        "live_interval_ms": 90,
        "live_preview_scale": 1.0,
        "live_preview_samples": 36,
        "live_preview_depth": 14,
        "live_refine_every": 2,
    },
}
DEFAULT_LIVE_PROFILE = "speed"

DEFAULT_CAMERA_POS = (6.8, 4.2, 9.5)
DEFAULT_CAMERA_YAW = -35.57
DEFAULT_CAMERA_PITCH = -15.32
DEFAULT_CAMERA_FOV = 34.0

OUTPUT_EXTENSIONS = {".ppm", ".png", ".jpg", ".jpeg", ".bmp", ".hdr", ".exr"}
LIVE_ACCUM_OUTPUT_EXTENSIONS = {".ppm", ".png", ".jpg", ".jpeg", ".bmp"}
PREVIEW_CACHE_MAX_ENTRIES = 24
STATUS_LOG_MAX_LINES = 320
MAX_THREADS = 64
TELEMETRY_REFRESH_SEC = 0.8


def normalize_demo(value: object) -> str:
    text = str(value if value is not None else DEFAULT_DEMO).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    if text not in DEMO_IDS:
        choices = ", ".join(sorted(DEMO_IDS))
        raise ValueError(f"Demo must be one of: {choices}.")
    return text


def normalize_live_profile(value: object) -> str:
    text = str(value if value is not None else DEFAULT_LIVE_PROFILE).strip().lower()
    if text not in LIVE_PROFILES:
        return DEFAULT_LIVE_PROFILE
    return text


def normalize_live_profile_ui(value: object) -> str:
    text = str(value if value is not None else DEFAULT_LIVE_PROFILE).strip().lower()
    if text == "custom":
        return "custom"
    if text in LIVE_PROFILES:
        return text
    return DEFAULT_LIVE_PROFILE


def is_gpu_compatible_demo(demo_id: str) -> bool:
    for item in DEMOS:
        if str(item.get("id")) == demo_id:
            return bool(item.get("gpu_compatible", False))
    return False


def gpu_compatible_demo_ids() -> List[str]:
    return [str(item["id"]) for item in DEMOS if bool(item.get("gpu_compatible", False))]


def json_error(message: str) -> Dict[str, str]:
    return {"error": message}


def clamp_int(name: str, value: object, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return result


def clamp_float(name: str, value: object, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return result


def parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        return int(value) != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def parse_ppm_to_bmp_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    n = len(data)
    idx = 0

    def next_token() -> bytes:
        nonlocal idx

        while idx < n:
            c = data[idx]
            if c in b" \t\r\n":
                idx += 1
                continue
            if c == ord("#"):
                idx += 1
                while idx < n and data[idx] not in (ord("\n"), ord("\r")):
                    idx += 1
                continue
            break

        if idx >= n:
            raise ValueError("Invalid PPM header")

        start = idx
        while idx < n and data[idx] not in b" \t\r\n":
            idx += 1
        return data[start:idx]

    magic = next_token()
    if magic != b"P6":
        raise ValueError("Only binary PPM (P6) is supported")

    width = int(next_token())
    height = int(next_token())
    max_value = int(next_token())

    if width <= 0 or height <= 0:
        raise ValueError("Invalid PPM dimensions")

    if max_value <= 0 or max_value > 255:
        raise ValueError("Unsupported PPM max value")

    while idx < n:
        c = data[idx]
        if c in b" \t\r\n":
            idx += 1
            continue
        if c == ord("#"):
            idx += 1
            while idx < n and data[idx] not in (ord("\n"), ord("\r")):
                idx += 1
            continue
        break

    pixel_count = width * height
    needed = pixel_count * 3
    if n - idx < needed:
        raise ValueError("PPM pixel data is truncated")

    rgb = data[idx:idx + needed]

    row_stride = width * 3
    row_pad = (4 - (row_stride % 4)) % 4
    bmp_img_size = (row_stride + row_pad) * height
    file_size = 14 + 40 + bmp_img_size

    file_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    dib_header = struct.pack(
        "<IIIHHIIIIII",
        40,
        width,
        height,
        1,
        24,
        0,
        bmp_img_size,
        2835,
        2835,
        0,
        0,
    )

    out = bytearray()
    out.extend(file_header)
    out.extend(dib_header)

    for y in range(height - 1, -1, -1):
        row = rgb[y * row_stride:(y + 1) * row_stride]
        for x in range(0, row_stride, 3):
            r = row[x]
            g = row[x + 1]
            b = row[x + 2]
            if max_value != 255:
                r = (r * 255) // max_value
                g = (g * 255) // max_value
                b = (b * 255) // max_value
            out.extend((b, g, r))
        if row_pad:
            out.extend(b"\x00" * row_pad)

    return bytes(out)


def parse_ppm_to_png_bytes(path: Path) -> bytes:
    """Convert PPM image to PNG bytes using PIL if available."""
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL is required for PNG conversion")
    
    try:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            png_buffer = io.BytesIO()
            img_rgb.save(png_buffer, format="PNG")
            return png_buffer.getvalue()
    except Exception as e:
        raise RuntimeError(f"Failed to convert PPM to PNG: {e}")


def convert_with_retries(converter, path: Path, *, attempts: int = 6, base_delay_sec: float = 0.03) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return converter(path)
        except Exception as exc:  # broad: converter-specific decode failures
            last_error = exc
            if attempt < attempts:
                time.sleep(base_delay_sec * attempt)
    raise RuntimeError(str(last_error) if last_error is not None else "Unknown conversion failure")


def read_ppm_rgb8(path: Path) -> Tuple[int, int, bytes]:
    data = path.read_bytes()
    n = len(data)
    idx = 0

    def next_token() -> bytes:
        nonlocal idx

        while idx < n:
            c = data[idx]
            if c in b" \t\r\n":
                idx += 1
                continue
            if c == ord("#"):
                idx += 1
                while idx < n and data[idx] not in (ord("\n"), ord("\r")):
                    idx += 1
                continue
            break

        if idx >= n:
            raise ValueError("Invalid PPM header")

        start = idx
        while idx < n and data[idx] not in b" \t\r\n":
            idx += 1
        return data[start:idx]

    magic = next_token()
    if magic != b"P6":
        raise ValueError("Only binary PPM (P6) is supported")

    width = int(next_token())
    height = int(next_token())
    max_value = int(next_token())

    if width <= 0 or height <= 0:
        raise ValueError("Invalid PPM dimensions")

    if max_value <= 0 or max_value > 255:
        raise ValueError("Unsupported PPM max value")

    while idx < n:
        c = data[idx]
        if c in b" \t\r\n":
            idx += 1
            continue
        if c == ord("#"):
            idx += 1
            while idx < n and data[idx] not in (ord("\n"), ord("\r")):
                idx += 1
            continue
        break

    needed = width * height * 3
    if n - idx < needed:
        raise ValueError("PPM pixel data is truncated")

    rgb = data[idx:idx + needed]
    if max_value == 255:
        return width, height, rgb

    scaled = bytearray(needed)
    for i in range(needed):
        scaled[i] = (rgb[i] * 255) // max_value
    return width, height, bytes(scaled)


def write_ppm_rgb8(path: Path, width: int, height: int, rgb: bytes) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size.")
    expected = width * height * 3
    if len(rgb) != expected:
        raise ValueError("RGB payload size does not match dimensions.")

    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + rgb)


def load_rgb8_image(path: Path) -> Tuple[int, int, bytes]:
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        return read_ppm_rgb8(path)

    if not PIL_AVAILABLE:
        raise RuntimeError("PIL is required for non-PPM live accumulation output.")

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        return width, height, rgb.tobytes()


def save_rgb8_image(path: Path, width: int, height: int, rgb: bytes) -> None:
    suffix = path.suffix.lower()
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        if suffix == ".ppm":
            write_ppm_rgb8(temp_path, width, height, rgb)
            temp_path.replace(path)
            return

        if not PIL_AVAILABLE:
            raise RuntimeError("PIL is required for non-PPM live accumulation output.")

        fmt_map = {
            ".png": "PNG",
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".bmp": "BMP",
        }
        fmt = fmt_map.get(suffix)
        if fmt is None:
            raise ValueError(f"Unsupported live accumulation output format: {suffix}")

        image = Image.frombytes("RGB", (width, height), rgb)
        image.save(temp_path, format=fmt)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def blend_accumulated_rgb8(
    accumulated_rgb: bytearray,
    frame_rgb: bytes,
    previous_sample_count: int,
    frame_samples: int,
) -> int:
    if previous_sample_count < 0 or frame_samples <= 0:
        raise ValueError("Invalid sample counts for accumulation.")
    if len(accumulated_rgb) != len(frame_rgb):
        raise ValueError("Accumulation and frame buffers must match.")

    if previous_sample_count == 0:
        accumulated_rgb[:] = frame_rgb
        return frame_samples

    total_samples = previous_sample_count + frame_samples
    prev_weight = previous_sample_count
    frame_weight = frame_samples

    # accumulated_color = (previous_color * (sample_count - 1) + new_sample) / sample_count
    # generalized to weighted sample batches:
    # next = (previous * prev_samples + current * frame_samples) / (prev_samples + frame_samples)
    for i in range(len(accumulated_rgb)):
        accumulated_rgb[i] = (
            accumulated_rgb[i] * prev_weight + frame_rgb[i] * frame_weight + total_samples // 2
        ) // total_samples

    return total_samples


@dataclass
class OrionFrontendState:
    # Process/thread-safe runtime state shared by HTTP handlers and worker threads.
    root_dir: Path
    out_dir: Path

    process: Optional[subprocess.Popen] = None
    worker_thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    movement_event: threading.Event = field(default_factory=threading.Event)
    live_settings_event: threading.Event = field(default_factory=threading.Event)

    running: bool = False
    mode: str = MODE_LIVE
    demo: str = DEFAULT_DEMO
    live_frame: int = 0
    live_phase: str = "idle"
    live_profile: str = DEFAULT_LIVE_PROFILE
    live_fps: float = 0.0
    live_last_frame_ms: float = 0.0
    live_preview_scale: float = 1.0
    live_preview_samples: int = 1
    live_quality_samples: int = 1
    live_output_name: str = ""
    live_add_to_recent: bool = True
    live_target_width: int = 1280
    live_target_height: int = 720
    live_target_samples: int = 64
    live_target_depth: int = 14
    live_target_threads: int = 8
    live_target_interval_ms: int = 24
    live_target_preview_scale: float = 0.84
    live_target_preview_samples: int = 8
    live_target_preview_depth: int = 8
    live_target_refine_every: int = 3
    live_output_path: str = "out/live_preview.ppm"
    backend_requested: str = "auto"
    backend_active: str = "unknown"

    command: List[str] = field(default_factory=list)
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=5000))

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    last_return_code: Optional[int] = None
    last_error: str = ""
    last_output: Optional[str] = None
    preview_png_cache: Dict[str, Tuple[int, int, bytes]] = field(default_factory=dict)
    preview_bmp_cache: Dict[str, Tuple[int, int, bytes]] = field(default_factory=dict)
    gpu_probe_status: str = "unknown"
    gpu_probe_message: str = "GPU probe not run yet."
    gpu_probe_checked_at: float = 0.0
    scene_spec: str = ""
    obj_path: str = ""
    scene_revision: int = 0
    camera_pos_x: float = DEFAULT_CAMERA_POS[0]
    camera_pos_y: float = DEFAULT_CAMERA_POS[1]
    camera_pos_z: float = DEFAULT_CAMERA_POS[2]
    camera_yaw: float = DEFAULT_CAMERA_YAW
    camera_pitch: float = DEFAULT_CAMERA_PITCH
    camera_fov: float = DEFAULT_CAMERA_FOV
    last_motion_ts: float = 0.0
    telemetry_cache: Dict[str, object] = field(default_factory=dict)
    telemetry_checked_at: float = 0.0
    cpu_prev_total_jiffies: int = 0
    cpu_prev_idle_jiffies: int = 0

    lock: threading.Lock = field(default_factory=threading.Lock)

    def list_outputs(self) -> List[str]:
        with self.lock:
            running_live = self.running and self.mode == MODE_LIVE
            hide_live_output = running_live and (not self.live_add_to_recent)
            live_output_name = self.live_output_name

        self.out_dir.mkdir(parents=True, exist_ok=True)
        files: List[Tuple[float, Path]] = []
        for path in self.out_dir.iterdir():
            if not path.is_file():
                continue
            if path.name.startswith(".") or path.name.endswith(".tmp"):
                # Hide transient/internal files from output list and thumbnails.
                continue
            if path.suffix.lower() not in OUTPUT_EXTENSIONS:
                continue
            if hide_live_output and live_output_name and path.name == live_output_name:
                continue
            try:
                mtime = float(path.stat().st_mtime)
            except OSError:
                continue
            files.append((mtime, path))

        files.sort(key=lambda item: item[0], reverse=True)
        return [path.name for _, path in files]

    def status(self) -> Dict[str, object]:
        with self.lock:
            now = time.time()
            start_time = self.start_time
            end_time = self.end_time
            running = self.running
            mode = self.mode
            demo = self.demo
            live_frame = self.live_frame
            live_phase = self.live_phase
            live_profile = self.live_profile
            live_fps = self.live_fps
            live_last_frame_ms = self.live_last_frame_ms
            live_preview_scale = self.live_preview_scale
            live_preview_samples = self.live_preview_samples
            live_quality_samples = self.live_quality_samples
            live_target_width = self.live_target_width
            live_target_height = self.live_target_height
            live_target_samples = self.live_target_samples
            live_target_depth = self.live_target_depth
            live_target_threads = self.live_target_threads
            live_target_interval_ms = self.live_target_interval_ms
            live_target_preview_scale = self.live_target_preview_scale
            live_target_preview_samples = self.live_target_preview_samples
            live_target_preview_depth = self.live_target_preview_depth
            live_target_refine_every = self.live_target_refine_every
            live_add_to_recent = self.live_add_to_recent
            live_output_name = self.live_output_name
            backend_requested = self.backend_requested
            backend_active = self.backend_active
            command = list(self.command)
            logs = list(self.logs)[-STATUS_LOG_MAX_LINES:]
            last_return_code = self.last_return_code
            last_error = self.last_error
            last_output = self.last_output
            scene_revision = self.scene_revision
            obj_path = self.obj_path
            camera_pos = [self.camera_pos_x, self.camera_pos_y, self.camera_pos_z]
            camera_yaw = self.camera_yaw
            camera_pitch = self.camera_pitch
            camera_fov = self.camera_fov

        telemetry = self.telemetry()

        if start_time is None:
            duration = 0.0
        elif running:
            duration = now - start_time
        else:
            end_t = end_time if end_time is not None else now
            duration = max(0.0, end_t - start_time)

        return {
            "running": running,
            "mode": mode,
            "demo": demo,
            "live_frame": live_frame,
            "live_phase": live_phase,
            "live_profile": live_profile,
            "live_fps": live_fps,
            "live_last_frame_ms": live_last_frame_ms,
            "live_preview_scale": live_preview_scale,
            "live_preview_samples": live_preview_samples,
            "live_quality_samples": live_quality_samples,
            "live_target_width": live_target_width,
            "live_target_height": live_target_height,
            "live_target_samples": live_target_samples,
            "live_target_depth": live_target_depth,
            "live_target_threads": live_target_threads,
            "live_target_interval_ms": live_target_interval_ms,
            "live_target_preview_scale": live_target_preview_scale,
            "live_target_preview_samples": live_target_preview_samples,
            "live_target_preview_depth": live_target_preview_depth,
            "live_target_refine_every": live_target_refine_every,
            "live_add_to_recent": live_add_to_recent,
            "live_output_name": live_output_name,
            "backend_requested": backend_requested,
            "backend_active": backend_active,
            "command": command,
            "logs": logs,
            "duration_sec": duration,
            "last_return_code": last_return_code,
            "last_error": last_error,
            "last_output": last_output,
            "scene_revision": scene_revision,
            "obj_path": obj_path,
            "camera_pos": camera_pos,
            "camera_yaw": camera_yaw,
            "camera_pitch": camera_pitch,
            "camera_fov": camera_fov,
            "telemetry": telemetry,
            "outputs": self.list_outputs(),
        }

    @staticmethod
    def _read_cpu_jiffies() -> Optional[Tuple[int, int]]:
        if os.name == "nt":
            class FILETIME(ctypes.Structure):
                _fields_ = [
                    ("dwLowDateTime", ctypes.c_ulong),
                    ("dwHighDateTime", ctypes.c_ulong),
                ]

            idle = FILETIME()
            kernel = FILETIME()
            user = FILETIME()
            ok = ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
                ctypes.byref(idle),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
            if not ok:
                return None

            def _to_int(value: FILETIME) -> int:
                return (int(value.dwHighDateTime) << 32) + int(value.dwLowDateTime)

            idle_ticks = _to_int(idle)
            kernel_ticks = _to_int(kernel)
            user_ticks = _to_int(user)
            total_ticks = kernel_ticks + user_ticks
            return total_ticks, idle_ticks

        try:
            with open("/proc/stat", "r", encoding="utf-8") as handle:
                first = handle.readline().strip()
        except OSError:
            return None
        if not first:
            return None
        parts = first.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return None
        try:
            fields = [int(token) for token in parts[1:]]
        except ValueError:
            return None
        if not fields:
            return None
        total = sum(fields)
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        return total, idle

    @staticmethod
    def _read_system_memory_percent() -> Optional[float]:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
                ctypes.byref(status)
            )
            if not ok:
                return None
            total = float(status.ullTotalPhys)
            available = float(status.ullAvailPhys)
            if total <= 0:
                return None
            used = max(0.0, total - available)
            return max(0.0, min(100.0, (used / total) * 100.0))

        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                total_kb: Optional[float] = None
                available_kb: Optional[float] = None
                for line in handle:
                    if line.startswith("MemTotal:"):
                        total_kb = float(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        available_kb = float(line.split()[1])
                    if total_kb is not None and available_kb is not None:
                        break
        except (OSError, ValueError, IndexError):
            total_kb = None
            available_kb = None
        if total_kb is None or available_kb is None or total_kb <= 0:
            try:
                page_size = os.sysconf("SC_PAGE_SIZE")
                total_pages = os.sysconf("SC_PHYS_PAGES")
                avail_pages = os.sysconf("SC_AVPHYS_PAGES")
                if page_size <= 0 or total_pages <= 0 or avail_pages < 0:
                    return None
                total_bytes = float(page_size * total_pages)
                available_bytes = float(page_size * avail_pages)
                used_bytes = max(0.0, total_bytes - available_bytes)
                return max(0.0, min(100.0, (used_bytes / total_bytes) * 100.0))
            except (AttributeError, OSError, ValueError):
                return None
        used = max(0.0, total_kb - available_kb)
        return max(0.0, min(100.0, (used / total_kb) * 100.0))

    @staticmethod
    def _read_process_rss_mb() -> Optional[float]:
        if os.name == "nt":
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            process = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            if not ok:
                return None
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)

        try:
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1]) / 1024.0
                        return None
        except (OSError, ValueError):
            pass
        try:
            import resource  # Unix-only fallback

            rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == "darwin":
                return rss / (1024.0 * 1024.0)
            return rss / 1024.0
        except Exception:
            return None
        return None

    @staticmethod
    def _probe_gpu_telemetry() -> Dict[str, object]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        unavailable = {
            "gpu_available": False,
            "gpu_util_percent": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
            "gpu_temp_c": None,
        }
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=0.35,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return unavailable
        if completed.returncode != 0:
            return unavailable

        rows: List[Tuple[float, float, float, float]] = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                util = float(parts[0])
                mem_used = float(parts[1])
                mem_total = float(parts[2])
                temp_c = float(parts[3])
            except ValueError:
                continue
            if mem_total <= 0:
                continue
            rows.append((util, mem_used, mem_total, temp_c))

        if not rows:
            return unavailable

        util_avg = sum(item[0] for item in rows) / len(rows)
        mem_used_sum = sum(item[1] for item in rows)
        mem_total_sum = sum(item[2] for item in rows)
        temp_max = max(item[3] for item in rows)

        return {
            "gpu_available": True,
            "gpu_util_percent": max(0.0, min(100.0, util_avg)),
            "gpu_mem_used_mb": max(0.0, mem_used_sum),
            "gpu_mem_total_mb": max(0.0, mem_total_sum),
            "gpu_temp_c": temp_max if temp_max >= 0 else None,
        }

    def telemetry(self, force: bool = False) -> Dict[str, object]:
        now = time.time()
        with self.lock:
            cached = dict(self.telemetry_cache) if self.telemetry_cache else {}
            checked_at = self.telemetry_checked_at
            prev_total = self.cpu_prev_total_jiffies
            prev_idle = self.cpu_prev_idle_jiffies

        if cached and not force and checked_at > 0 and (now - checked_at) < TELEMETRY_REFRESH_SEC:
            return cached

        cpu_percent: Optional[float] = None
        jiffies = self._read_cpu_jiffies()
        if jiffies is not None:
            total, idle = jiffies
            if prev_total > 0 and total > prev_total:
                delta_total = total - prev_total
                delta_idle = max(0, idle - prev_idle)
                if delta_total > 0:
                    cpu_percent = max(0.0, min(100.0, (1.0 - (delta_idle / delta_total)) * 100.0))
        else:
            total = 0
            idle = 0

        metrics: Dict[str, object] = {
            "cpu_percent": cpu_percent,
            "mem_percent": self._read_system_memory_percent(),
            "rss_mb": self._read_process_rss_mb(),
            "updated_at": now,
        }
        metrics.update(self._probe_gpu_telemetry())

        with self.lock:
            if jiffies is not None:
                self.cpu_prev_total_jiffies = total
                self.cpu_prev_idle_jiffies = idle
            self.telemetry_cache = metrics
            self.telemetry_checked_at = now

        return dict(metrics)

    def gpu_probe(self, force: bool = False) -> Dict[str, object]:
        now = time.time()
        with self.lock:
            cached_status = self.gpu_probe_status
            cached_message = self.gpu_probe_message
            cached_checked_at = self.gpu_probe_checked_at

        # Reuse probe results briefly to avoid repeatedly launching full probes.
        if not force and cached_checked_at > 0 and (now - cached_checked_at) < 60.0:
            return {
                "status": cached_status,
                "message": cached_message,
                "checked_at": cached_checked_at,
            }

        binary = self._orion_binary()
        if not binary.exists() or not binary.is_file():
            status = "unavailable"
            message = f"Binary missing: {binary}. Build first with `cmake --build {BUILD_DIR} -j`."
            checked_at = time.time()
            with self.lock:
                self.gpu_probe_status = status
                self.gpu_probe_message = message
                self.gpu_probe_checked_at = checked_at
            return {"status": status, "message": message, "checked_at": checked_at}

        probe_output = self.out_dir / ".gpu_probe.ppm"
        cmd = [
            str(binary),
            "--backend", "gpu",
            "--demo", "scene_editor",
            "--width", "64",
            "--height", "64",
            "--samples", "1",
            "--depth", "2",
            "--threads", "1",
            "--seed", "11",
            "--output", str(probe_output),
            "--quiet",
        ]

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.root_dir),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            text = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
            if completed.returncode == 0:
                status = "ok"
                message = "GPU rendering is available."
            else:
                status = "failed"
                detail = ""
                for line in text.splitlines():
                    lower = line.lower()
                    if "gpu render failed" in lower or "cuda" in lower or "gpu backend requested" in lower:
                        detail = line.strip()
                        break
                if not detail:
                    detail = f"GPU probe exited with code {completed.returncode}."
                message = detail
        except subprocess.TimeoutExpired:
            status = "failed"
            message = "GPU probe timed out while testing CUDA rendering."
        except OSError as exc:
            status = "failed"
            message = f"Failed to execute GPU probe: {exc}"
        finally:
            try:
                probe_output.unlink(missing_ok=True)
            except OSError:
                pass

        checked_at = time.time()
        with self.lock:
            self.gpu_probe_status = status
            self.gpu_probe_message = message
            self.gpu_probe_checked_at = checked_at
        return {"status": status, "message": message, "checked_at": checked_at}

    def update_scene(self, scene_spec: str, obj_path: Optional[str] = None) -> Dict[str, object]:
        spec = str(scene_spec).strip()
        if len(spec) > 120000:
            raise ValueError("Scene specification is too large.")
        next_obj_path: Optional[str] = None
        if obj_path is not None:
            next_obj_path = self._resolve_obj_path(obj_path)

        now = time.time()
        changed = False
        should_reset = False
        with self.lock:
            obj_changed = next_obj_path is not None and next_obj_path != self.obj_path
            if spec != self.scene_spec or obj_changed:
                self.scene_spec = spec
                if next_obj_path is not None:
                    self.obj_path = next_obj_path
                self.scene_revision += 1
                self.last_motion_ts = now
                changed = True
                if self.running and self.mode == MODE_LIVE:
                    self.movement_event.set()
                    should_reset = True
                    self.logs.append("[live] scene updated; accumulation reset.")

        if changed and not should_reset:
            if next_obj_path is not None:
                if next_obj_path:
                    self._append_log(f"[scene] updated (OBJ={Path(next_obj_path).name})")
                else:
                    self._append_log("[scene] updated (OBJ cleared)")
            else:
                self._append_log("[scene] updated")
        return self.status()

    def update_camera(self, payload: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("Camera payload must be an object.")

        with self.lock:
            fallback = (
                (self.camera_pos_x, self.camera_pos_y, self.camera_pos_z),
                self.camera_yaw,
                self.camera_pitch,
                self.camera_fov,
            )
            running_live = self.running and self.mode == MODE_LIVE

        parsed = self._parse_camera_payload(
            payload,
            fallback_pos=fallback[0],
            fallback_yaw=fallback[1],
            fallback_pitch=fallback[2],
            fallback_fov=fallback[3],
        )

        changed = False
        now = time.time()
        with self.lock:
            eps = 1e-5
            changed = (
                abs(self.camera_pos_x - parsed["camera_pos_x"]) > eps
                or abs(self.camera_pos_y - parsed["camera_pos_y"]) > eps
                or abs(self.camera_pos_z - parsed["camera_pos_z"]) > eps
                or abs(self.camera_yaw - parsed["camera_yaw"]) > eps
                or abs(self.camera_pitch - parsed["camera_pitch"]) > eps
                or abs(self.camera_fov - parsed["camera_fov"]) > eps
            )
            if changed:
                self.camera_pos_x = parsed["camera_pos_x"]
                self.camera_pos_y = parsed["camera_pos_y"]
                self.camera_pos_z = parsed["camera_pos_z"]
                self.camera_yaw = parsed["camera_yaw"]
                self.camera_pitch = parsed["camera_pitch"]
                self.camera_fov = parsed["camera_fov"]
                self.last_motion_ts = now
                if running_live:
                    self.movement_event.set()

        return {"updated": changed}

    def update_live_settings(self, payload: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("Live settings payload must be an object.")

        with self.lock:
            if not (self.running and self.mode == MODE_LIVE):
                raise ValueError("Live settings can only be changed while live rendering is running.")

            current_width = self.live_target_width
            current_height = self.live_target_height
            current_samples = self.live_target_samples
            current_depth = self.live_target_depth
            current_threads = self.live_target_threads
            current_interval_ms = self.live_target_interval_ms
            current_preview_scale = self.live_target_preview_scale
            current_preview_samples = self.live_target_preview_samples
            current_preview_depth = self.live_target_preview_depth
            current_refine_every = self.live_target_refine_every
            current_profile = normalize_live_profile_ui(self.live_profile)
            current_output_path = self.live_output_path
            current_add_to_recent = self.live_add_to_recent

        next_profile = normalize_live_profile_ui(payload.get("live_profile", current_profile))
        profile_selected = "live_profile" in payload
        profile_defaults = LIVE_PROFILES[next_profile] if next_profile in LIVE_PROFILES else None

        width = clamp_int("Width", payload.get("width", current_width), 64, 16384)
        height = clamp_int("Height", payload.get("height", current_height), 64, 16384)
        threads = clamp_int("Threads", payload.get("threads", current_threads), 1, MAX_THREADS)

        live_samples_raw = payload.get("live_samples", payload.get("samples"))
        if live_samples_raw is None:
            if profile_selected and profile_defaults is not None:
                live_samples_raw = profile_defaults["live_samples"]
            else:
                live_samples_raw = current_samples
        live_samples = clamp_int("Live samples", live_samples_raw, 1, 512)

        live_depth_raw = payload.get("live_depth", payload.get("depth"))
        if live_depth_raw is None:
            if profile_selected and profile_defaults is not None:
                live_depth_raw = profile_defaults["live_depth"]
            else:
                live_depth_raw = current_depth
        live_depth = clamp_int("Live depth", live_depth_raw, 1, 64)

        interval_raw = payload.get("live_interval_ms")
        if interval_raw is None:
            if profile_selected and profile_defaults is not None:
                interval_raw = profile_defaults["live_interval_ms"]
            else:
                interval_raw = current_interval_ms
        interval_ms = clamp_int("Live interval", interval_raw, 0, 10000)

        preview_scale_raw = payload.get("live_preview_scale")
        if preview_scale_raw is None:
            if profile_selected and profile_defaults is not None:
                preview_scale_raw = profile_defaults["live_preview_scale"]
            else:
                preview_scale_raw = current_preview_scale
        preview_scale = clamp_float("Live preview scale", preview_scale_raw, 0.8, 1.0)

        preview_samples_raw = payload.get("live_preview_samples")
        if preview_samples_raw is None:
            if profile_selected and profile_defaults is not None:
                preview_samples_raw = profile_defaults["live_preview_samples"]
            else:
                preview_samples_raw = current_preview_samples
        preview_samples = clamp_int("Live preview samples", preview_samples_raw, 1, 64)

        preview_depth_raw = payload.get("live_preview_depth")
        if preview_depth_raw is None:
            if profile_selected and profile_defaults is not None:
                preview_depth_raw = profile_defaults["live_preview_depth"]
            else:
                preview_depth_raw = current_preview_depth
        preview_depth = clamp_int("Live preview depth", preview_depth_raw, 1, 64)

        refine_every_raw = payload.get("live_refine_every")
        if refine_every_raw is None:
            if profile_selected and profile_defaults is not None:
                refine_every_raw = profile_defaults["live_refine_every"]
            else:
                refine_every_raw = current_refine_every
        refine_every = clamp_int("Live refine every", refine_every_raw, 1, 120)

        live_add_to_recent = parse_bool(
            payload.get("live_add_to_recent", current_add_to_recent),
            default=current_add_to_recent,
        )

        output_path = self._resolve_output_path(payload.get("live_output", current_output_path), current_output_path)
        output_suffix = output_path.suffix.lower()
        if output_suffix not in LIVE_ACCUM_OUTPUT_EXTENSIONS:
            choices = ", ".join(sorted(LIVE_ACCUM_OUTPUT_EXTENSIONS))
            raise ValueError(
                f"Live output format `{output_suffix or '(none)'}` is not supported for stable accumulation. "
                f"Use one of: {choices}."
            )

        reset_required = (
            width != current_width
            or height != current_height
            or live_depth != current_depth
            or preview_depth != current_preview_depth
            or abs(preview_scale - current_preview_scale) > 1e-8
        )

        with self.lock:
            if not (self.running and self.mode == MODE_LIVE):
                raise ValueError("Live settings can only be changed while live rendering is running.")

            self.live_target_width = width
            self.live_target_height = height
            self.live_target_samples = live_samples
            self.live_target_depth = live_depth
            self.live_target_threads = threads
            self.live_target_interval_ms = interval_ms
            self.live_target_preview_scale = preview_scale
            self.live_target_preview_samples = preview_samples
            self.live_target_preview_depth = preview_depth
            self.live_target_refine_every = refine_every
            self.live_profile = next_profile
            self.live_add_to_recent = live_add_to_recent
            self.live_output_path = str(output_path)
            self.live_output_name = output_path.name
            if reset_required:
                self.live_settings_event.set()
                self.logs.append("[live] runtime settings updated; accumulation reset.")
            else:
                self.logs.append("[live] runtime settings updated.")

        output_path_ui = str(output_path.relative_to(self.root_dir))
        applied = {
            "width": width,
            "height": height,
            "threads": threads,
            "live_samples": live_samples,
            "live_depth": live_depth,
            "live_interval_ms": interval_ms,
            "live_preview_scale": preview_scale,
            "live_preview_samples": preview_samples,
            "live_preview_depth": preview_depth,
            "live_refine_every": refine_every,
            "live_profile": next_profile,
            "live_add_to_recent": live_add_to_recent,
            "live_output": output_path_ui,
        }
        return {"applied": applied, "reset": reset_required, "status": self.status()}

    def _append_log(self, line: str) -> None:
        text = line.rstrip("\n")
        with self.lock:
            self.logs.append(text)

    def _backend_guess_for_request(self, backend: str, demo: str) -> str:
        if backend == "cpu":
            return "cpu"
        if backend == "gpu":
            return "gpu"
        demo_meta = next((item for item in DEMOS if str(item.get("id")) == demo), None)
        if demo_meta is not None and not bool(demo_meta.get("gpu_compatible", False)):
            return "cpu"
        return "gpu"

    def _update_backend_from_output_line(self, text: str) -> None:
        normalized = text.lower()
        backend: Optional[str] = None

        if "| backend=cpu" in normalized:
            backend = "cpu"
        elif "| backend=gpu" in normalized:
            backend = "gpu"
        elif "falling back to cpu" in normalized:
            backend = "cpu"
        elif "switching to cpu backend" in normalized:
            backend = "cpu"
        elif "gpu render failed" in normalized:
            backend = "cpu"
        elif "gpu render unavailable" in normalized:
            backend = "cpu"

        if backend is None:
            return

        with self.lock:
            self.backend_active = backend

    def _orion_binary(self) -> Path:
        override = os.environ.get("ORION_BINARY_PATH", "").strip()
        if override:
            custom = Path(override).expanduser()
            if not custom.is_absolute():
                custom = (self.root_dir / custom).resolve()
            else:
                custom = custom.resolve()
            return custom

        build_dir = self.root_dir / "build"
        candidates = [
            build_dir / "orion_raytracer",
            build_dir / "orion_raytracer.exe",
            build_dir / "Release" / "orion_raytracer",
            build_dir / "Release" / "orion_raytracer.exe",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return candidates[0]

    def _resolve_output_path(self, raw_output: object, default_name: str) -> Path:
        output_text = str(raw_output if raw_output is not None else "").strip() or default_name
        output_path = Path(output_text)

        if not output_path.is_absolute():
            output_path = self.root_dir / output_path
        output_path = output_path.resolve()

        # Hard boundary: never allow writes outside the configured output directory.
        if self.out_dir not in output_path.parents:
            raise ValueError("Output path must be inside `orion-raytracer/out/`.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _obj_imports_dir(self) -> Path:
        path = OBJ_IMPORTS_DIR.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_obj_path(self, raw_obj_path: object) -> str:
        text = str(raw_obj_path if raw_obj_path is not None else "").strip()
        if not text:
            return ""

        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = (self.root_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        imports_dir = self._obj_imports_dir()
        if imports_dir not in candidate.parents:
            raise ValueError("OBJ path must be inside `orion-raytracer/out/imports/`.")
        if candidate.suffix.lower() != ".obj":
            raise ValueError("OBJ path must end with `.obj`.")
        if not candidate.exists() or not candidate.is_file():
            raise ValueError(f"OBJ file not found: {candidate}")
        return str(candidate)

    def _sanitize_asset_name(self, raw_name: object, used_names: Optional[set[str]] = None) -> str:
        text = str(raw_name if raw_name is not None else "").replace("\\", "/").strip()
        name = Path(text).name if text else "asset.bin"
        if not name:
            name = "asset.bin"

        safe = "".join(ch if (ch.isalnum() or ch in {".", "_", "-"}) else "_" for ch in name)
        safe = safe.strip("._")
        if not safe:
            safe = "asset.bin"

        stem = Path(safe).stem or "asset"
        suffix = Path(safe).suffix[:12]
        if len(stem) > 56:
            stem = stem[:56]
        candidate = f"{stem}{suffix}"

        if used_names is None:
            return candidate

        taken = {entry.lower() for entry in used_names}
        if candidate.lower() not in taken:
            used_names.add(candidate)
            return candidate

        index = 2
        while True:
            alt = f"{stem}_{index}{suffix}"
            if alt.lower() not in taken:
                used_names.add(alt)
                return alt
            index += 1

    def _convert_texture_to_ppm(self, source_path: Path) -> Optional[Path]:
        if source_path.suffix.lower() == ".ppm":
            return source_path
        if not PIL_AVAILABLE:
            return None
        try:
            target = source_path.with_name(f"{source_path.stem}_orion.ppm")
            if not target.exists():
                with Image.open(source_path) as image:
                    image.convert("RGB").save(target, format="PPM")
            return target
        except Exception:
            return None

    def _rewrite_mtl_textures(self, mtl_path: Path) -> None:
        try:
            lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return

        out_lines: List[str] = []
        changed = False
        imports_dir = self._obj_imports_dir()

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                out_lines.append(line)
                continue

            parts = stripped.split(None, 1)
            if len(parts) != 2:
                out_lines.append(line)
                continue

            raw_key = parts[0]
            key = raw_key.lower()
            if key not in {"map_kd", "map_ke"}:
                out_lines.append(line)
                continue

            value = parts[1].strip()
            if not value:
                out_lines.append(line)
                continue

            if '"' in value:
                q0 = value.find('"')
                q1 = value.rfind('"')
                tex_name = value[q0 + 1:q1].strip() if q0 >= 0 and q1 > q0 else ""
            else:
                tex_name = value.split()[-1] if value.split() else ""

            if not tex_name:
                out_lines.append(line)
                continue

            tex_path = Path(tex_name)
            if not tex_path.is_absolute():
                tex_path = (mtl_path.parent / tex_path).resolve()
            else:
                tex_path = tex_path.resolve()

            if imports_dir not in tex_path.parents or not tex_path.exists() or not tex_path.is_file():
                out_lines.append(line)
                continue

            converted_path = self._convert_texture_to_ppm(tex_path)
            if converted_path is None:
                out_lines.append(line)
                continue

            try:
                rel = converted_path.relative_to(mtl_path.parent).as_posix()
            except ValueError:
                rel = converted_path.name
            out_lines.append(f"{raw_key} {rel}")
            changed = True

        if changed:
            mtl_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    def _postprocess_obj_bundle(self, obj_path: Path) -> None:
        try:
            obj_lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return

        imports_dir = self._obj_imports_dir()
        mtl_paths: List[Path] = []
        for line in obj_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or not stripped.lower().startswith("mtllib "):
                continue
            for token in stripped[7:].split():
                name = Path(token).name
                if not name:
                    continue
                candidate = (obj_path.parent / name).resolve()
                if imports_dir in candidate.parents and candidate.exists() and candidate.is_file():
                    mtl_paths.append(candidate)

        seen: set[str] = set()
        for mtl_path in mtl_paths:
            key = str(mtl_path)
            if key in seen:
                continue
            seen.add(key)
            self._rewrite_mtl_textures(mtl_path)

    def _import_obj_from_path(self, target_obj: Path, display_name: str) -> Dict[str, object]:
        now = time.time()
        should_reset = False
        with self.lock:
            self.obj_path = str(target_obj)
            self.scene_revision += 1
            self.last_motion_ts = now
            if self.running and self.mode == MODE_LIVE:
                self.movement_event.set()
                should_reset = True
                self.logs.append("[live] OBJ mesh imported; accumulation reset.")

        if not should_reset:
            self._append_log(f"[scene] imported OBJ: {display_name}")

        return {
            "obj_path": str(target_obj),
            "obj_name": display_name,
            "status": self.status(),
        }

    def import_obj(self, filename: object, content: object, assets: Optional[object] = None) -> Dict[str, object]:
        source_name = str(filename if filename is not None else "import.obj").strip() or "import.obj"
        payload = str(content if content is not None else "")

        asset_list = assets if isinstance(assets, list) else None
        if asset_list:
            # Bundle mode: write all uploaded assets into a private import directory and
            # automatically wire OBJ->MTL->texture links to local files.
            stem = Path(source_name).stem or "import"
            safe_stem = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in stem).strip("_")
            if not safe_stem:
                safe_stem = "import"
            safe_stem = safe_stem[:48]

            bundle_dir = self._obj_imports_dir() / f"{int(time.time() * 1000)}_{safe_stem}"
            bundle_dir.mkdir(parents=True, exist_ok=True)

            used_names: set[str] = set()
            written_files: List[Path] = []
            total_bytes = 0
            max_bundle_bytes = 96 * 1024 * 1024
            for entry in asset_list:
                if not isinstance(entry, dict):
                    continue
                file_name = self._sanitize_asset_name(entry.get("name", "asset.bin"), used_names)
                encoded = str(entry.get("data_base64", "")).strip()
                if not encoded:
                    continue
                try:
                    blob = base64.b64decode(encoded, validate=True)
                except Exception as exc:
                    raise ValueError(f"Invalid base64 payload for `{file_name}`.") from exc
                if not blob:
                    continue
                total_bytes += len(blob)
                if total_bytes > max_bundle_bytes:
                    raise ValueError("OBJ asset bundle is too large (max 96 MB).")
                target = bundle_dir / file_name
                target.write_bytes(blob)
                written_files.append(target)

            desired_name = self._sanitize_asset_name(source_name)
            obj_files = [path for path in written_files if path.suffix.lower() == ".obj"]
            target_obj = next((path for path in obj_files if path.name.lower() == desired_name.lower()), None)

            if target_obj is None and payload.strip():
                if len(payload.encode("utf-8", errors="ignore")) > 48 * 1024 * 1024:
                    raise ValueError("OBJ file is too large (max 48 MB).")
                target_obj = bundle_dir / desired_name
                target_obj.write_text(payload, encoding="utf-8", errors="ignore")
                written_files.append(target_obj)

            if target_obj is None and obj_files:
                target_obj = obj_files[0]

            if target_obj is None:
                raise ValueError("No OBJ file found in imported asset bundle.")

            self._postprocess_obj_bundle(target_obj)
            return self._import_obj_from_path(target_obj, target_obj.name)

        if not payload.strip():
            raise ValueError("OBJ content is empty.")
        if len(payload.encode("utf-8", errors="ignore")) > 48 * 1024 * 1024:
            raise ValueError("OBJ file is too large (max 48 MB).")

        stem = Path(source_name).stem or "import"
        safe_stem = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in stem).strip("_")
        if not safe_stem:
            safe_stem = "import"
        safe_stem = safe_stem[:48]

        target = self._obj_imports_dir() / f"{int(time.time() * 1000)}_{safe_stem}.obj"
        target.write_text(payload, encoding="utf-8", errors="ignore")
        self._postprocess_obj_bundle(target)
        return self._import_obj_from_path(target, target.name)

    def _camera_snapshot(self) -> Tuple[Tuple[float, float, float], float, float, float]:
        with self.lock:
            return (
                (self.camera_pos_x, self.camera_pos_y, self.camera_pos_z),
                self.camera_yaw,
                self.camera_pitch,
                self.camera_fov,
            )

    def _parse_camera_payload(
        self,
        payload: Dict[str, object],
        *,
        fallback_pos: Tuple[float, float, float],
        fallback_yaw: float,
        fallback_pitch: float,
        fallback_fov: float,
    ) -> Dict[str, float]:
        def parse_number(name: str, raw: object, fallback: float, minimum: float, maximum: float) -> float:
            if raw is None:
                return fallback
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a number.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value < minimum or value > maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}.")
            return value

        pos_x, pos_y, pos_z = fallback_pos
        raw_pos = payload.get("camera_pos")
        if isinstance(raw_pos, (list, tuple)) and len(raw_pos) == 3:
            pos_x = parse_number("Camera X", raw_pos[0], pos_x, -2000.0, 2000.0)
            pos_y = parse_number("Camera Y", raw_pos[1], pos_y, -2000.0, 2000.0)
            pos_z = parse_number("Camera Z", raw_pos[2], pos_z, -2000.0, 2000.0)
        elif isinstance(raw_pos, dict):
            pos_x = parse_number("Camera X", raw_pos.get("x"), pos_x, -2000.0, 2000.0)
            pos_y = parse_number("Camera Y", raw_pos.get("y"), pos_y, -2000.0, 2000.0)
            pos_z = parse_number("Camera Z", raw_pos.get("z"), pos_z, -2000.0, 2000.0)
        elif isinstance(raw_pos, str) and raw_pos.strip():
            parts = [part.strip() for part in raw_pos.split(",")]
            if len(parts) != 3:
                raise ValueError("Camera position must be 3 comma-separated values.")
            pos_x = parse_number("Camera X", parts[0], pos_x, -2000.0, 2000.0)
            pos_y = parse_number("Camera Y", parts[1], pos_y, -2000.0, 2000.0)
            pos_z = parse_number("Camera Z", parts[2], pos_z, -2000.0, 2000.0)

        pos_x = parse_number("Camera X", payload.get("camera_x"), pos_x, -2000.0, 2000.0)
        pos_y = parse_number("Camera Y", payload.get("camera_y"), pos_y, -2000.0, 2000.0)
        pos_z = parse_number("Camera Z", payload.get("camera_z"), pos_z, -2000.0, 2000.0)

        yaw = parse_number("Camera yaw", payload.get("camera_yaw"), fallback_yaw, -36000.0, 36000.0)
        pitch = parse_number("Camera pitch", payload.get("camera_pitch"), fallback_pitch, -89.0, 89.0)
        fov = parse_number("Camera FOV", payload.get("camera_fov"), fallback_fov, 20.0, 100.0)

        return {
            "camera_pos_x": pos_x,
            "camera_pos_y": pos_y,
            "camera_pos_z": pos_z,
            "camera_yaw": yaw,
            "camera_pitch": pitch,
            "camera_fov": fov,
        }

    def _build_render_command(
        self,
        *,
        width: int,
        height: int,
        samples: int,
        depth: int,
        threads: int,
        seed: int,
        backend: str,
        demo: str,
        output_path: Path,
        scene_spec: str = "",
        obj_path: str = "",
        camera_pos: Optional[Tuple[float, float, float]] = None,
        camera_yaw: Optional[float] = None,
        camera_pitch: Optional[float] = None,
        camera_fov: Optional[float] = None,
        exposure: float = 1.0,
        quiet: bool = False,
    ) -> List[str]:
        cmd = [
            str(self._orion_binary()),
            "--backend", backend,
            "--demo", demo,
            "--width", str(width),
            "--height", str(height),
            "--samples", str(samples),
            "--depth", str(depth),
            "--threads", str(threads),
            "--seed", str(seed),
            "--exposure", str(exposure),
            "--output", str(output_path),
        ]
        if scene_spec:
            cmd.extend(["--scene-spec", scene_spec])
        if obj_path:
            cmd.extend(["--obj-path", obj_path])
        if camera_pos is not None:
            cmd.extend(["--camera-pos", f"{camera_pos[0]},{camera_pos[1]},{camera_pos[2]}"])
        if camera_yaw is not None:
            cmd.extend(["--camera-yaw", str(camera_yaw)])
        if camera_pitch is not None:
            cmd.extend(["--camera-pitch", str(camera_pitch)])
        if camera_fov is not None:
            cmd.extend(["--camera-fov", str(camera_fov)])
        if quiet:
            cmd.append("--quiet")
        return cmd

    def _parse_common(self, data: Dict[str, object]) -> Dict[str, object]:
        width = clamp_int("Width", data.get("width", 1280), 64, 16384)
        height = clamp_int("Height", data.get("height", 720), 64, 16384)
        threads = clamp_int("Threads", data.get("threads", 8), 1, MAX_THREADS)
        seed = clamp_int("Seed", data.get("seed", 7), 0, 2**31 - 1)

        backend = str(data.get("backend", "auto")).lower()
        if backend not in BACKENDS:
            raise ValueError("Backend must be auto, cpu, or gpu.")
        # Raycism Engine now runs only the editable environment.
        demo = DEFAULT_DEMO
        scene_spec = str(data.get("scene_spec", "")).strip()
        if len(scene_spec) > 120000:
            raise ValueError("Scene specification is too large.")
        with self.lock:
            current_obj_path = self.obj_path
        obj_path = self._resolve_obj_path(data.get("obj_path", current_obj_path))
        if backend == "gpu" and not is_gpu_compatible_demo(demo):
            gpu_demos = gpu_compatible_demo_ids()
            if gpu_demos:
                demo_list = ", ".join(f"`{name}`" for name in gpu_demos)
                raise ValueError(
                    f"Demo `{demo}` is CPU-only and cannot run with GPU backend. "
                    f"Choose one of {demo_list} for GPU rendering."
                )
            raise ValueError(
                f"Demo `{demo}` is CPU-only and GPU demos are disabled in this Raycism Engine build."
            )

        fallback_pos, fallback_yaw, fallback_pitch, fallback_fov = self._camera_snapshot()
        camera = self._parse_camera_payload(
            data,
            fallback_pos=fallback_pos,
            fallback_yaw=fallback_yaw,
            fallback_pitch=fallback_pitch,
            fallback_fov=fallback_fov,
        )

        return {
            "width": width,
            "height": height,
            "threads": threads,
            "seed": seed,
            "backend": backend,
            "demo": demo,
            "scene_spec": scene_spec,
            "obj_path": obj_path,
            **camera,
        }

    def _parse_render_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
        common = self._parse_common(payload)
        samples = clamp_int("Samples", payload.get("samples", 400), 1, 200000)
        depth = clamp_int("Depth", payload.get("depth", 18), 1, 128)
        output_path = self._resolve_output_path(payload.get("output"), f"out/render_{int(time.time())}.ppm")

        return {
            **common,
            "samples": samples,
            "depth": depth,
            "output_path": output_path,
        }

    def _parse_live_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
        common = self._parse_common(payload)
        live_profile = normalize_live_profile_ui(payload.get("live_profile", DEFAULT_LIVE_PROFILE))
        profile = LIVE_PROFILES[live_profile] if live_profile in LIVE_PROFILES else LIVE_PROFILES[DEFAULT_LIVE_PROFILE]

        live_samples = clamp_int(
            "Live samples",
            payload.get("live_samples", payload.get("samples", profile["live_samples"])),
            1,
            512,
        )
        live_depth = clamp_int(
            "Live depth",
            payload.get("live_depth", payload.get("depth", profile["live_depth"])),
            1,
            64,
        )
        interval_ms = clamp_int("Live interval", payload.get("live_interval_ms", profile["live_interval_ms"]), 0, 10000)
        preview_scale = clamp_float(
            "Live preview scale",
            payload.get("live_preview_scale", profile["live_preview_scale"]),
            0.8,
            1.0,
        )
        preview_samples = clamp_int(
            "Live preview samples",
            payload.get("live_preview_samples", profile["live_preview_samples"]),
            1,
            64,
        )
        preview_depth = clamp_int(
            "Live preview depth",
            payload.get("live_preview_depth", profile["live_preview_depth"]),
            1,
            64,
        )
        refine_every = clamp_int("Live refine every", payload.get("live_refine_every", profile["live_refine_every"]), 1, 120)
        live_add_to_recent = parse_bool(payload.get("live_add_to_recent", True), default=True)
        output_path = self._resolve_output_path(payload.get("live_output"), "out/live_preview.ppm")
        output_suffix = output_path.suffix.lower()
        if output_suffix not in LIVE_ACCUM_OUTPUT_EXTENSIONS:
            choices = ", ".join(sorted(LIVE_ACCUM_OUTPUT_EXTENSIONS))
            raise ValueError(
                f"Live output format `{output_suffix or '(none)'}` is not supported for stable accumulation. "
                f"Use one of: {choices}."
            )

        return {
            **common,
            "samples": live_samples,
            "depth": live_depth,
            "interval_ms": interval_ms,
            "preview_scale": preview_scale,
            "preview_samples": preview_samples,
            "preview_depth": preview_depth,
            "refine_every": refine_every,
            "live_profile": live_profile,
            "live_add_to_recent": live_add_to_recent,
            "output_path": output_path,
        }

    @staticmethod
    def _scaled_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
        scaled_w = max(64, int(round(width * scale)))
        scaled_h = max(64, int(round(height * scale)))
        return scaled_w, scaled_h

    def _finalize_single(self, proc: subprocess.Popen, before_outputs: List[str]) -> None:
        if proc.stdout is not None:
            for line in proc.stdout:
                self._update_backend_from_output_line(line.rstrip("\n"))
                self._append_log(line)

        return_code = proc.wait()

        outputs = self.list_outputs()
        new_outputs = [name for name in outputs if name not in before_outputs]
        detected_output = new_outputs[0] if new_outputs else (outputs[0] if outputs else None)

        with self.lock:
            self.running = False
            self.process = None
            self.end_time = time.time()
            self.last_return_code = return_code
            self.last_output = detected_output
            if return_code != 0:
                self.last_error = f"Process exited with code {return_code}. Check logs for details."

    def _run_live_loop(self, settings: Dict[str, object]) -> None:
        frame = 0

        seed = int(settings["seed"])
        backend = str(settings["backend"])
        demo = str(settings["demo"])
        frame_output_path = self.out_dir / ".live_preview.frame.ppm"
        motion_hold_sec = 0.45

        accumulated_rgb: Optional[bytearray] = None
        accumulated_width = int(settings["width"])
        accumulated_height = int(settings["height"])
        accumulated_sample_count = 0

        self.movement_event.clear()
        self.live_settings_event.clear()

        with self.lock:
            self.live_phase = "accumulating"
            self.live_preview_scale = 1.0
            self.live_preview_samples = 0
            self.live_quality_samples = 0

        # Live mode runs in two phases:
        # 1) interactive: low-cost frame updates while user moves camera/scene
        # 2) accumulating: higher-quality batches blended over time while still
        while not self.stop_event.is_set():
            if self.movement_event.is_set():
                self.movement_event.clear()
                accumulated_rgb = None
                accumulated_sample_count = 0
                with self.lock:
                    self.logs.append("[live] camera/scene motion detected; accumulation reset.")

            if self.live_settings_event.is_set():
                self.live_settings_event.clear()
                accumulated_rgb = None
                accumulated_sample_count = 0

            frame += 1
            frame_seed = seed + (frame - 1) * 104729
            with self.lock:
                live_scene_spec = self.scene_spec
                live_obj_path = self.obj_path
                width = self.live_target_width
                height = self.live_target_height
                samples = self.live_target_samples
                depth = self.live_target_depth
                threads = self.live_target_threads
                interval_ms = self.live_target_interval_ms
                preview_scale = self.live_target_preview_scale
                preview_samples = max(1, min(self.live_target_preview_samples, samples))
                preview_depth = max(1, min(self.live_target_preview_depth, depth))
                live_profile = self.live_profile
                output_path = Path(self.live_output_path)
                camera_pos = (self.camera_pos_x, self.camera_pos_y, self.camera_pos_z)
                camera_yaw = self.camera_yaw
                camera_pitch = self.camera_pitch
                camera_fov = self.camera_fov
                moving = (time.time() - self.last_motion_ts) < motion_hold_sec

            # Keep batch settings stable for accumulation while allowing runtime retuning.
            quality_frame_samples = max(1, preview_samples)
            quality_frame_depth = depth
            interactive_scale = max(0.12, min(0.50, preview_scale * 0.20))
            interactive_frame_depth = max(1, min(2, preview_depth, depth))

            if moving:
                frame_width, frame_height = self._scaled_dimensions(width, height, interactive_scale)
                frame_samples = 1
                frame_depth = interactive_frame_depth
            else:
                frame_width, frame_height = width, height
                frame_samples = quality_frame_samples
                frame_depth = quality_frame_depth

            command = self._build_render_command(
                width=frame_width,
                height=frame_height,
                samples=frame_samples,
                depth=frame_depth,
                threads=threads,
                seed=frame_seed,
                backend=backend,
                demo=demo,
                output_path=frame_output_path,
                scene_spec=live_scene_spec,
                obj_path=live_obj_path,
                camera_pos=camera_pos,
                camera_yaw=camera_yaw,
                camera_pitch=camera_pitch,
                camera_fov=camera_fov,
                exposure=1.0,
                quiet=True,
            )

            with self.lock:
                self.command = command
                self.live_phase = "interactive" if moving else "accumulating"
                self.live_preview_scale = frame_width / max(1, width)
                self.logs.append(
                    f"[live] frame {frame} {'interactive' if moving else 'accumulate'} "
                    f"{frame_width}x{frame_height} batch_spp={frame_samples} total_spp={accumulated_sample_count}"
                )

            try:
                frame_start = time.time()
                proc = subprocess.Popen(
                    command,
                    cwd=str(self.root_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                with self.lock:
                    self.last_error = f"Failed to start live frame {frame}: {exc}"
                    self.last_return_code = -1
                break

            with self.lock:
                self.process = proc

            if proc.stdout is not None:
                for line in proc.stdout:
                    text = line.rstrip("\n")
                    if text:
                        self._update_backend_from_output_line(text)
                        self._append_log(text)
                    if self.stop_event.is_set() and proc.poll() is None:
                        proc.terminate()

            if self.stop_event.is_set() and proc.poll() is None:
                proc.terminate()

            return_code = proc.wait()
            if return_code != 0:
                with self.lock:
                    self.process = None
                    self.last_return_code = return_code
                    if not self.stop_event.is_set():
                        self.last_error = f"Live frame {frame} failed with code {return_code}."
                if not self.stop_event.is_set():
                    break
                continue

            try:
                frame_width, frame_height, frame_rgb = load_rgb8_image(frame_output_path)
            except Exception as exc:
                with self.lock:
                    self.process = None
                    self.last_return_code = -1
                    self.last_error = f"Failed to load live frame {frame}: {exc}"
                break

            if moving:
                # During interaction we show only the latest frame (no temporal accumulation).
                accumulated_rgb = None
                accumulated_sample_count = 0
                try:
                    save_rgb8_image(output_path, frame_width, frame_height, frame_rgb)
                except Exception as exc:
                    with self.lock:
                        self.process = None
                        self.last_return_code = -1
                        self.last_error = f"Failed to write interactive frame {frame}: {exc}"
                    break
                shown_sample_count = frame_samples
            else:
                if (
                    accumulated_rgb is None
                    or frame_width != accumulated_width
                    or frame_height != accumulated_height
                ):
                    accumulated_width = frame_width
                    accumulated_height = frame_height
                    accumulated_rgb = bytearray(frame_rgb)
                    accumulated_sample_count = frame_samples
                else:
                    # Weighted blending preserves total effective sample count across batches.
                    try:
                        accumulated_sample_count = blend_accumulated_rgb8(
                            accumulated_rgb,
                            frame_rgb,
                            accumulated_sample_count,
                            frame_samples,
                        )
                    except Exception as exc:
                        with self.lock:
                            self.process = None
                            self.last_return_code = -1
                            self.last_error = f"Failed to blend live frame {frame}: {exc}"
                        break

                try:
                    save_rgb8_image(output_path, accumulated_width, accumulated_height, bytes(accumulated_rgb))
                except Exception as exc:
                    with self.lock:
                        self.process = None
                        self.last_return_code = -1
                        self.last_error = f"Failed to write accumulated frame {frame}: {exc}"
                    break
                shown_sample_count = accumulated_sample_count

            elapsed_ms = max(0.1, (time.time() - frame_start) * 1000.0)
            live_fps = 1000.0 / elapsed_ms

            with self.lock:
                self.process = None
                self.last_return_code = 0
                self.last_output = output_path.name
                self.live_frame = frame
                self.live_last_frame_ms = elapsed_ms
                self.live_fps = live_fps
                self.live_phase = "interactive" if moving else "accumulating"
                self.live_preview_scale = frame_width / max(1, width)
                self.live_preview_samples = shown_sample_count
                self.live_quality_samples = shown_sample_count

            target_interval_ms = 0 if moving else interval_ms
            if target_interval_ms > 0:
                wait_sec = max(0.0, (target_interval_ms - elapsed_ms) / 1000.0)
                if wait_sec > 0.0 and self.stop_event.wait(wait_sec):
                    break

        with self.lock:
            self.running = False
            self.process = None
            self.end_time = time.time()
            self.worker_thread = None
            self.live_phase = "idle"
            self.live_fps = 0.0

        if self.stop_event.is_set():
            self._append_log("[live] stopped")
        try:
            frame_output_path.unlink(missing_ok=True)
        except OSError:
            pass

    def start(self, payload: Dict[str, object]) -> Dict[str, object]:
        mode = str(payload.get("mode", MODE_LIVE)).lower()
        if mode in {"studio", "web", "live_web"}:
            mode = MODE_LIVE

        if mode not in {MODE_RENDER, MODE_LIVE}:
            raise ValueError("Mode must be render or live.")

        binary = self._orion_binary()
        if not binary.exists() or not binary.is_file():
            raise ValueError(
                f"Binary missing: {binary}. Build first with `cmake --build {BUILD_DIR} -j`."
            )

        with self.lock:
            if self.running:
                raise ValueError("Another Raycism Engine task is already running.")

        self.stop_event.clear()
        self.movement_event.clear()
        self.live_settings_event.clear()

        if mode == MODE_RENDER:
            # One-shot render mode: spawn exactly one renderer process and watch for completion.
            parsed = self._parse_render_payload(payload)
            command = self._build_render_command(
                width=int(parsed["width"]),
                height=int(parsed["height"]),
                samples=int(parsed["samples"]),
                depth=int(parsed["depth"]),
                threads=int(parsed["threads"]),
                seed=int(parsed["seed"]),
                backend=str(parsed["backend"]),
                demo=str(parsed["demo"]),
                output_path=Path(parsed["output_path"]),
                scene_spec=str(parsed.get("scene_spec", "")),
                obj_path=str(parsed.get("obj_path", "")),
                camera_pos=(
                    float(parsed["camera_pos_x"]),
                    float(parsed["camera_pos_y"]),
                    float(parsed["camera_pos_z"]),
                ),
                camera_yaw=float(parsed["camera_yaw"]),
                camera_pitch=float(parsed["camera_pitch"]),
                camera_fov=float(parsed["camera_fov"]),
            )

            before_outputs = self.list_outputs()

            try:
                proc = subprocess.Popen(
                    command,
                    cwd=str(self.root_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                raise ValueError(f"Failed to start process: {exc}") from exc

            with self.lock:
                self.process = proc
                self.running = True
                self.mode = MODE_RENDER
                self.demo = str(parsed["demo"])
                self.backend_requested = str(parsed["backend"])
                self.backend_active = self._backend_guess_for_request(
                    str(parsed["backend"]),
                    str(parsed["demo"]),
                )
                self.scene_spec = str(parsed.get("scene_spec", ""))
                self.obj_path = str(parsed.get("obj_path", ""))
                self.camera_pos_x = float(parsed["camera_pos_x"])
                self.camera_pos_y = float(parsed["camera_pos_y"])
                self.camera_pos_z = float(parsed["camera_pos_z"])
                self.camera_yaw = float(parsed["camera_yaw"])
                self.camera_pitch = float(parsed["camera_pitch"])
                self.camera_fov = float(parsed["camera_fov"])
                self.live_frame = 0
                self.live_phase = "idle"
                self.live_profile = DEFAULT_LIVE_PROFILE
                self.live_fps = 0.0
                self.live_last_frame_ms = 0.0
                self.live_preview_scale = 1.0
                self.live_preview_samples = 1
                self.live_quality_samples = int(parsed["samples"])
                self.live_output_name = ""
                self.live_output_path = ""
                self.live_add_to_recent = True
                self.live_target_width = int(parsed["width"])
                self.live_target_height = int(parsed["height"])
                self.live_target_samples = int(parsed["samples"])
                self.live_target_depth = int(parsed["depth"])
                self.live_target_threads = int(parsed["threads"])
                self.live_target_interval_ms = 24
                self.live_target_preview_scale = 1.0
                self.live_target_preview_samples = 1
                self.live_target_preview_depth = int(parsed["depth"])
                self.live_target_refine_every = 1
                self.command = command
                self.logs.clear()
                self.logs.append("[render] starting...")
                self.last_return_code = None
                self.last_error = ""
                self.last_output = None
                self.start_time = time.time()
                self.end_time = None
                self.last_motion_ts = 0.0

            watcher = threading.Thread(target=self._finalize_single, args=(proc, before_outputs), daemon=True)
            watcher.start()
            with self.lock:
                self.worker_thread = watcher
            return self.status()

        parsed = self._parse_live_payload(payload)

        # Live mode keeps a background loop alive until explicit stop.
        with self.lock:
            self.running = True
            self.mode = MODE_LIVE
            self.demo = str(parsed["demo"])
            self.backend_requested = str(parsed["backend"])
            self.backend_active = self._backend_guess_for_request(
                str(parsed["backend"]),
                str(parsed["demo"]),
            )
            self.scene_spec = str(parsed.get("scene_spec", ""))
            self.obj_path = str(parsed.get("obj_path", ""))
            self.camera_pos_x = float(parsed["camera_pos_x"])
            self.camera_pos_y = float(parsed["camera_pos_y"])
            self.camera_pos_z = float(parsed["camera_pos_z"])
            self.camera_yaw = float(parsed["camera_yaw"])
            self.camera_pitch = float(parsed["camera_pitch"])
            self.camera_fov = float(parsed["camera_fov"])
            self.live_frame = 0
            self.live_phase = "accumulating"
            self.live_profile = str(parsed["live_profile"])
            self.live_fps = 0.0
            self.live_last_frame_ms = 0.0
            self.live_preview_scale = 1.0
            self.live_preview_samples = 0
            self.live_quality_samples = 0
            self.live_output_name = Path(parsed["output_path"]).name
            self.live_output_path = str(Path(parsed["output_path"]))
            self.live_add_to_recent = bool(parsed.get("live_add_to_recent", True))
            self.live_target_width = int(parsed["width"])
            self.live_target_height = int(parsed["height"])
            self.live_target_samples = int(parsed["samples"])
            self.live_target_depth = int(parsed["depth"])
            self.live_target_threads = int(parsed["threads"])
            self.live_target_interval_ms = int(parsed["interval_ms"])
            self.live_target_preview_scale = float(parsed["preview_scale"])
            self.live_target_preview_samples = int(parsed["preview_samples"])
            self.live_target_preview_depth = int(parsed["preview_depth"])
            self.live_target_refine_every = int(parsed["refine_every"])
            self.command = []
            self.logs.clear()
            self.logs.append(
                "[live] starting progressive accumulation "
                f"(interval={int(parsed['interval_ms'])}ms, batch_spp={int(parsed['preview_samples'])})..."
            )
            self.last_return_code = None
            self.last_error = ""
            self.last_output = None
            self.start_time = time.time()
            self.end_time = None
            self.last_motion_ts = 0.0

        worker = threading.Thread(target=self._run_live_loop, args=(parsed,), daemon=True)
        worker.start()

        with self.lock:
            self.worker_thread = worker

        return self.status()

    def stop(self) -> Dict[str, object]:
        with self.lock:
            proc = self.process
            thread = self.worker_thread
            is_running = self.running

        if not is_running:
            return self.status()

        self.stop_event.set()
        self.movement_event.clear()
        self.live_settings_event.clear()

        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=6)

        with self.lock:
            if self.running:
                self.running = False
                self.process = None
                self.end_time = time.time()
            self.live_phase = "idle"
            self.live_fps = 0.0

        return self.status()

    def latest_ppm_output(self) -> Optional[Path]:
        with self.lock:
            candidate_name = self.last_output

        if candidate_name:
            candidate = self.out_dir / candidate_name
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".ppm":
                return candidate

        for name in self.list_outputs():
            path = self.out_dir / name
            if path.suffix.lower() == ".ppm":
                return path

        return None


class OrionFrontendHandler(BaseHTTPRequestHandler):
    state: OrionFrontendState = None  # type: ignore

    def log_message(self, fmt: str, *args) -> None:
        return

    @staticmethod
    def _is_disconnect_error(exc: BaseException) -> bool:
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)):
            return True
        if isinstance(exc, OSError):
            return exc.errno in {
                errno.EPIPE,
                errno.ECONNRESET,
                errno.ECONNABORTED,
                errno.ETIMEDOUT,
            }
        return False

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except Exception as exc:  # broad: keep request thread stable on abrupt client disconnect
            if self._is_disconnect_error(exc):
                self.close_connection = True
                return
            raise

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:  # broad: stream/client may close between header and body write
            if self._is_disconnect_error(exc):
                self.close_connection = True
                return
            raise

    def _send_json(self, status: int, payload: Dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(status, "application/json", body)

    @staticmethod
    def _prune_preview_cache(cache: Dict[str, Tuple[int, int, bytes]]) -> None:
        while len(cache) > PREVIEW_CACHE_MAX_ENTRIES:
            cache.pop(next(iter(cache)))

    def _send_html(self) -> None:
        if HTML_PATH.exists():
            body = HTML_PATH.read_bytes()
        else:
            body = (
                b"<!doctype html><html><body><h1>Missing frontend/index.html</h1>"
                b"<p>Create frontend/index.html and refresh.</p></body></html>"
            )
        self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, json_error("File not found."))
            return

        mime_type, _ = mimetypes.guess_type(str(path))
        content_type = mime_type or "application/octet-stream"
        try:
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.end_headers()

            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except Exception as exc:
            if self._is_disconnect_error(exc):
                self.close_connection = True
                return
            raise

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        # Read-only API surface for status, previews, outputs, and capability probes.
        if parsed.path in {"/", "/index.html"}:
            self._send_html()
            return

        if parsed.path == "/api/status":
            self._send_json(HTTPStatus.OK, {"status": self.state.status()})
            return

        if parsed.path == "/api/outputs":
            self._send_json(HTTPStatus.OK, {"outputs": self.state.list_outputs()})
            return

        if parsed.path == "/api/demos":
            self._send_json(
                HTTPStatus.OK,
                {
                    "default_demo": DEFAULT_DEMO,
                    "demos": DEMOS,
                    "default_live_profile": DEFAULT_LIVE_PROFILE,
                    "live_profiles": LIVE_PROFILES,
                },
            )
            return

        if parsed.path in {"/api/gpu_probe", "/api/gpu_probe/"}:
            force = query.get("force", ["0"])[0] in {"1", "true", "yes", "on"}
            probe = self.state.gpu_probe(force=force)
            self._send_json(HTTPStatus.OK, {"gpu": probe})
            return

        if parsed.path == "/api/preview.bmp":
            requested_name = str(query.get("output", [""])[0]).strip()
            ppm_path: Optional[Path] = None
            if requested_name:
                candidate = (self.state.out_dir / requested_name).resolve()
                if self.state.out_dir not in candidate.parents:
                    self._send_json(HTTPStatus.BAD_REQUEST, json_error("Invalid output path."))
                    return
                if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".ppm":
                    ppm_path = candidate
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, json_error("Requested PPM output not found."))
                    return
            else:
                ppm_path = self.state.latest_ppm_output()
            if ppm_path is None:
                self._send_json(HTTPStatus.NOT_FOUND, json_error("No PPM output available for preview."))
                return

            cache_key = ppm_path.name
            try:
                ppm_stat = ppm_path.stat()
                ppm_mtime_ns = int(ppm_stat.st_mtime_ns)
                ppm_size = int(ppm_stat.st_size)
            except OSError:
                self._send_json(HTTPStatus.NOT_FOUND, json_error("Preview output is unavailable."))
                return

            with self.state.lock:
                cached = self.state.preview_bmp_cache.get(cache_key)
            if cached is not None and cached[0] == ppm_mtime_ns and cached[1] == ppm_size:
                self._send_bytes(HTTPStatus.OK, "image/bmp", cached[2])
                return

            try:
                bmp_bytes = convert_with_retries(parse_ppm_to_bmp_bytes, ppm_path)
                with self.state.lock:
                    self.state.preview_bmp_cache[cache_key] = (ppm_mtime_ns, ppm_size, bmp_bytes)
                    self._prune_preview_cache(self.state.preview_bmp_cache)
            except Exception as exc:  # broad: malformed image should not crash server
                with self.state.lock:
                    cached = self.state.preview_bmp_cache.get(cache_key)
                if cached is None:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, json_error(f"Preview conversion failed: {exc}"))
                    return
                bmp_bytes = cached[2]

            self._send_bytes(HTTPStatus.OK, "image/bmp", bmp_bytes)
            return

        if parsed.path.startswith("/output/"):
            rel_name = unquote(parsed.path.removeprefix("/output/"))
            requested = (self.state.out_dir / rel_name).resolve()
            if self.state.out_dir not in requested.parents:
                self._send_json(HTTPStatus.BAD_REQUEST, json_error("Invalid output path."))
                return
            if not requested.exists() or not requested.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, json_error("Output file not found."))
                return
            
            # Convert PPM to PNG for browser compatibility
            if requested.suffix.lower() == ".ppm":
                cache_key = requested.name
                try:
                    ppm_stat = requested.stat()
                    ppm_mtime_ns = int(ppm_stat.st_mtime_ns)
                    ppm_size = int(ppm_stat.st_size)
                except OSError:
                    self._send_json(HTTPStatus.NOT_FOUND, json_error("Output file is unavailable."))
                    return
                with self.state.lock:
                    cached = self.state.preview_png_cache.get(cache_key)
                if cached is not None and cached[0] == ppm_mtime_ns and cached[1] == ppm_size:
                    self._send_bytes(HTTPStatus.OK, "image/png", cached[2])
                    return
                try:
                    png_bytes = convert_with_retries(parse_ppm_to_png_bytes, requested)
                    with self.state.lock:
                        self.state.preview_png_cache[cache_key] = (ppm_mtime_ns, ppm_size, png_bytes)
                        self._prune_preview_cache(self.state.preview_png_cache)
                    self._send_bytes(HTTPStatus.OK, "image/png", png_bytes)
                    return
                except Exception as exc:
                    with self.state.lock:
                        cached = self.state.preview_png_cache.get(cache_key)
                    if cached is not None:
                        self._send_bytes(HTTPStatus.OK, "image/png", cached[2])
                        return
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, json_error(f"Failed to convert PPM: {exc}"))
                    return
            
            self._send_file(requested)
            return

        self._send_json(HTTPStatus.NOT_FOUND, json_error("Endpoint not found."))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, json_error("Invalid JSON payload."))
            return

        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, json_error("JSON body must be an object."))
            return

        # Mutating API surface for run control, scene/camera updates, and live tuning.
        if parsed.path == "/api/run":
            try:
                status = self.state.start(payload)
                self._send_json(HTTPStatus.ACCEPTED, {"status": status})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, json_error(str(exc)))
            return

        if parsed.path == "/api/stop":
            status = self.state.stop()
            self._send_json(HTTPStatus.OK, {"status": status})
            return

        if parsed.path == "/api/camera":
            try:
                self.state.update_camera(payload)
                self._send_json(HTTPStatus.OK, {"ok": True})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, json_error(str(exc)))
            return

        if parsed.path == "/api/move":
            # Trigger render on camera movement
            with self.state.lock:
                self.state.last_motion_ts = time.time()
                self.state.movement_event.set()
            self._send_json(HTTPStatus.OK, {"triggered": "movement"})
            return

        if parsed.path == "/api/scene":
            try:
                scene_spec = str(payload.get("scene_spec", ""))
                obj_path = payload.get("obj_path", None)
                status = self.state.update_scene(scene_spec, obj_path=obj_path)
                self._send_json(HTTPStatus.OK, {"status": status})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, json_error(str(exc)))
            return

        if parsed.path == "/api/live_settings":
            try:
                result = self.state.update_live_settings(payload)
                self._send_json(HTTPStatus.OK, result)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, json_error(str(exc)))
            return

        if parsed.path == "/api/import_obj":
            try:
                result = self.state.import_obj(
                    payload.get("filename", "import.obj"),
                    payload.get("content", ""),
                    assets=payload.get("assets", None),
                )
                self._send_json(HTTPStatus.OK, result)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, json_error(str(exc)))
            return

        self._send_json(HTTPStatus.NOT_FOUND, json_error("Endpoint not found."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Raycism Engine local frontend")
    parser.add_argument("--host", default="127.0.0.1", help="Host bind address")
    parser.add_argument("--port", type=int, default=8092, help="Port (default: 8092)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OBJ_IMPORTS_DIR.mkdir(parents=True, exist_ok=True)

    state = OrionFrontendState(root_dir=PROJECT_ROOT, out_dir=OUT_DIR)
    OrionFrontendHandler.state = state

    server = ThreadingHTTPServer((args.host, args.port), OrionFrontendHandler)
    print(f"[raycism-frontend] URL: http://{args.host}:{args.port}")
    print(f"[raycism-frontend] Project: {PROJECT_ROOT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
