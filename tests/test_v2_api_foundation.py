"""Foundation and feature-flag tests for RoadVision AI 2.0 Flask API."""

import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml
from flask import Flask

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roadvision_api_v2 import load_api_config, register_v2_api
from roadvision_api_v2.configuration import APIConfig


class TestV2APIFoundation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = APIConfig(
            enabled=True,
            host="127.0.0.1",
            port=5000,
            database_path=os.path.join(self._tmp.name, "jobs.db"),
            upload_directory=os.path.join(self._tmp.name, "uploads"),
            artifact_directory=os.path.join(self._tmp.name, "artifacts"),
            project_root=self._tmp.name,
        )
        self.cfg.validate()
        self.cfg.ensure_runtime_dirs()

        self.app = Flask(__name__)
        register_v2_api(self.app, self.cfg)
        self.client = self.app.test_client()

    def tearDown(self):
        self._tmp.cleanup()

    def test_committed_config_disabled_by_default(self):
        cfg = load_api_config(ensure_dirs=False, apply_env_overrides=False)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(cfg.max_gpu_concurrency, 1)
        self.assertEqual(cfg.max_queued_jobs, 5)

    def test_env_override_enables_flag(self):
        with mock.patch.dict(os.environ, {"ROADVISION_V2_ENABLED": "true"}):
            cfg = load_api_config(ensure_dirs=False, apply_env_overrides=True)
            self.assertTrue(cfg.enabled)

    def test_config_validation_rejects_public_bind_and_multi_gpu(self):
        bad_host = APIConfig(host="0.0.0.0")
        with self.assertRaises(ValueError):
            bad_host.validate()

        bad_gpu = APIConfig(max_gpu_concurrency=2)
        with self.assertRaises(ValueError):
            bad_gpu.validate()

        bad_cors = APIConfig(allowed_cors_origins=["*"])
        with self.assertRaises(ValueError):
            bad_cors.validate()

    def test_v2_health_endpoint(self):
        resp = self.client.get("/api/v2/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["api_enabled"])
        self.assertFalse(data["worker_running"])  # Never auto-started on registration

    def test_v2_modes_endpoint(self):
        resp = self.client.get("/api/v2/modes")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("modes", data)
        for m in ("live", "balanced", "survey"):
            self.assertIn(m, data["modes"])

    def test_v2_capabilities_endpoint(self):
        resp = self.client.get("/api/v2/capabilities")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("gpu_available", data)
        self.assertIn("models", data)
        self.assertEqual(data["limits"]["max_gpu_concurrency"], 1)

    def test_server_module_feature_flag_and_legacy_routes(self):
        # Verify server.py mounts legacy routes and disables V2 by default without heavy reload
        fake_pipeline = mock.MagicMock()
        with mock.patch("roadvision_engine.RoadVisionPipeline", return_value=fake_pipeline):
            if "server" in sys.modules:
                srv = importlib.reload(sys.modules["server"])
            else:
                import server as srv
            client = srv.app.test_client()
            self.assertEqual(client.get("/api/health").status_code, 200)
            self.assertEqual(client.get("/api/v2/health").status_code, 404)

            with mock.patch.dict(os.environ, {"ROADVISION_V2_ENABLED": "true"}):
                importlib.reload(srv)
                v2_client = srv.app.test_client()
                self.assertEqual(v2_client.get("/api/v2/health").status_code, 200)
                self.assertEqual(v2_client.get("/api/health").status_code, 200)

        # Reload back to default disabled state cleanly
        with mock.patch("roadvision_engine.RoadVisionPipeline", return_value=fake_pipeline):
            importlib.reload(sys.modules["server"])

    def test_standalone_v2_cli_help(self):
        import subprocess
        cli_path = os.path.join(os.path.dirname(__file__), "..", "run_roadvision_v2.py")
        res = subprocess.run([sys.executable, cli_path, "--help"], capture_output=True, text=True, timeout=15)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--enable-depth", res.stdout)
        self.assertIn("--mode", res.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
