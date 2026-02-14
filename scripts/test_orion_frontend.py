#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    module_path = Path(__file__).resolve().with_name("orion_frontend.py")
    spec = importlib.util.spec_from_file_location("orion_frontend", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load orion_frontend module for tests.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FRONTEND = load_module()


class OrionFrontendTests(unittest.TestCase):
    def make_state(self) -> object:
        root = Path(tempfile.mkdtemp(prefix="orion_test_root_"))
        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        return FRONTEND.OrionFrontendState(root_dir=root, out_dir=out_dir)

    def test_blend_accumulated_rgb8_weighted_average(self) -> None:
        accumulated = bytearray([120, 60, 30])
        frame = bytes([20, 140, 210])
        total = FRONTEND.blend_accumulated_rgb8(
            accumulated,
            frame,
            previous_sample_count=4,
            frame_samples=2,
        )
        self.assertEqual(total, 6)
        self.assertEqual(list(accumulated), [87, 87, 90])

    def test_blend_accumulated_rgb8_initial_frame(self) -> None:
        accumulated = bytearray([0, 0, 0, 0, 0, 0])
        frame = bytes([5, 10, 15, 20, 25, 30])
        total = FRONTEND.blend_accumulated_rgb8(
            accumulated,
            frame,
            previous_sample_count=0,
            frame_samples=3,
        )
        self.assertEqual(total, 3)
        self.assertEqual(bytes(accumulated), frame)

    def test_clamp_float_rejects_non_finite(self) -> None:
        with self.assertRaises(ValueError):
            FRONTEND.clamp_float("Live preview scale", "nan", 0.8, 1.0)
        with self.assertRaises(ValueError):
            FRONTEND.clamp_float("Live preview scale", "inf", 0.8, 1.0)

    def test_parse_live_payload_parses_live_add_to_recent(self) -> None:
        state = self.make_state()
        payload = {
            "mode": "live",
            "backend": "cpu",
            "width": 96,
            "height": 64,
            "threads": 1,
            "seed": 7,
            "live_samples": 1,
            "live_depth": 2,
            "live_interval_ms": 24,
            "live_output": "out/live_preview.ppm",
            "live_add_to_recent": "false",
        }
        parsed = state._parse_live_payload(payload)
        self.assertFalse(parsed["live_add_to_recent"])

    def test_parse_camera_payload_rejects_non_finite_values(self) -> None:
        state = self.make_state()
        with self.assertRaises(ValueError):
            state._parse_camera_payload(
                {"camera_yaw": "nan"},
                fallback_pos=(0.0, 0.0, 0.0),
                fallback_yaw=0.0,
                fallback_pitch=0.0,
                fallback_fov=34.0,
            )

    def test_parse_common_rejects_unreasonable_thread_counts(self) -> None:
        state = self.make_state()
        with self.assertRaises(ValueError):
            state._parse_common(
                {
                    "backend": "cpu",
                    "width": 96,
                    "height": 64,
                    "seed": 7,
                    "threads": 100,
                }
            )

    def test_list_outputs_ignores_internal_files_and_live_preview_when_hidden(self) -> None:
        state = self.make_state()
        out_dir = state.out_dir

        (out_dir / "render_a.ppm").write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        (out_dir / ".live_preview.frame.ppm").write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
        (out_dir / "render_b.png.tmp").write_bytes(b"temp")

        with state.lock:
            state.running = True
            state.mode = FRONTEND.MODE_LIVE
            state.live_add_to_recent = False
            state.live_output_name = "render_a.ppm"

        outputs = state.list_outputs()
        self.assertEqual(outputs, [])

    def test_status_includes_telemetry_payload(self) -> None:
        state = self.make_state()
        status = state.status()
        telemetry = status.get("telemetry")
        self.assertIsInstance(telemetry, dict)
        if not isinstance(telemetry, dict):
            return
        for key in [
            "cpu_percent",
            "mem_percent",
            "rss_mb",
            "gpu_available",
            "gpu_util_percent",
            "gpu_mem_used_mb",
            "gpu_mem_total_mb",
            "gpu_temp_c",
            "updated_at",
        ]:
            self.assertIn(key, telemetry)

    def test_update_live_settings_requires_running_live_mode(self) -> None:
        state = self.make_state()
        with self.assertRaises(ValueError):
            state.update_live_settings({"live_samples": 32})

    def test_update_live_settings_applies_changes_and_profile_defaults(self) -> None:
        state = self.make_state()
        with state.lock:
            state.running = True
            state.mode = FRONTEND.MODE_LIVE
            state.live_profile = "speed"
            state.live_target_width = 1280
            state.live_target_height = 720
            state.live_target_samples = 24
            state.live_target_depth = 10
            state.live_target_threads = 4
            state.live_target_interval_ms = 24
            state.live_target_preview_scale = 0.84
            state.live_target_preview_samples = 8
            state.live_target_preview_depth = 8
            state.live_target_refine_every = 7
            state.live_add_to_recent = True
            state.live_output_path = str(state.out_dir / "live_preview.ppm")
            state.live_output_name = "live_preview.ppm"
            state.live_settings_event.clear()

        result = state.update_live_settings({
            "width": 960,
            "live_profile": "ultra",
            "threads": 6,
        })
        applied = result.get("applied", {})
        self.assertTrue(result.get("reset"))
        self.assertEqual(applied.get("width"), 960)
        self.assertEqual(applied.get("threads"), 6)
        self.assertEqual(applied.get("live_profile"), "ultra")
        self.assertEqual(applied.get("live_samples"), 128)
        self.assertEqual(applied.get("live_depth"), 18)
        self.assertEqual(applied.get("live_interval_ms"), 90)
        with state.lock:
            self.assertEqual(state.live_target_width, 960)
            self.assertEqual(state.live_target_threads, 6)
            self.assertEqual(state.live_profile, "ultra")
            self.assertTrue(state.live_settings_event.is_set())


if __name__ == "__main__":
    unittest.main()
