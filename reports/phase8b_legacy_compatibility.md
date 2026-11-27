# Phase 8B — Legacy V1 Compatibility Report

## 1. Preserved Legacy Files and Constants

- `roadvision_engine.py`: **Unmodified** (retains exact Phase 8A SHA256 `3805808750134A38B3FA9BCFC42FB6583A843DDA575E70AEF28FF205C0E2A51E`).
- `config/camera_params.json`: **Unmodified** (retains exact Phase 8A SHA256 `0C7633F330A8B5C15A1696B85D81E9CA6D32E3CAC1C1C237CFFF8E576E789AF1`).
- Calibration constants `k_disp=0.082` and `depth_exponent=1.35` remain untouched.
- Production YOLOv8 checkpoints and frozen dataset manifests remain untouched.
- React dashboard (`roadvision-dashboard/`) remains untouched.

## 2. Minimal `server.py` Integration

Modifications to `server.py` are limited to:
1. Importing V2 helpers (`load_api_config`, `register_v2_api`, `start_v2_worker`, `stop_v2_worker`, `gpu_inference_lock`).
2. Conditionally registering `/api/v2` when `api_cfg.enabled` is `True` (committed default is `enabled: false`).
3. Wrapping legacy `pipeline.process_video(...)` inside `/api/detect` with `with gpu_inference_lock():` so legacy V1 and V2 GPU jobs serialize safely without changing any legacy JSON response structure or status codes.
4. Explicitly starting/stopping the V2 worker inside `if __name__ == "__main__":` only when `api_cfg.enabled` is `True`.

## 3. Legacy Routes Verified

- `GET /api/health`
- `GET /api/benchmark`
- `GET /api/camera-config`
- `POST /api/detect`
- `GET /outputs/<path:filename>`

## 4. Standalone V2 CLI

`run_roadvision_v2.py` preserves all existing CLI flags (`--source`, `--mode`, `--enable-depth`, `--no-depth`, `--max-frames`, `--save-video`, `--save-json`, `--save-csv`, `--diagnostics`, `--output-dir`, `--device`) and runs independently of the Flask server.
