"""
========================================================================================
🏷️ RoadVision AI — Human-in-the-Loop Dataset V2 Annotation Platform
========================================================================================
Dedicated local web application for precise manual verification of candidate frames:
  1. Interactive canvas for drawing, adjusting, and deleting pothole bounding boxes.
  2. One-click "Mark as Clean Road (0 Potholes)" for negative training samples.
  3. Real-time YOLOv8 format label generation into dataset_v2_candidates/labels/
  4. Keeps original candidate images untouched.
  5. Manifest logger tracking verified status, timestamps, and box counts.
========================================================================================
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, Any, List
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CANDIDATES_DIR = os.path.join(BASE_DIR, "dataset_v2_candidates")
LABELS_DIR = os.path.join(CANDIDATES_DIR, "labels")
MANIFEST_PATH = os.path.join(CANDIDATES_DIR, "review_manifest.json")

os.makedirs(LABELS_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

def load_manifest() -> Dict[str, Any]:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"reviews": {}, "updated_at": 0}

def save_manifest(manifest: Dict[str, Any]):
    manifest["updated_at"] = time.time()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def get_candidate_files() -> List[str]:
    files = [
        f for f in sorted(os.listdir(CANDIDATES_DIR))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return files

# ======================================================================================
# API ROUTES
# ======================================================================================

@app.route("/api/frames", methods=["GET"])
def get_frames():
    manifest = load_manifest()
    reviews = manifest.get("reviews", {})
    files = get_candidate_files()

    frames = []
    for idx, fname in enumerate(files):
        stem = os.path.splitext(fname)[0]
        label_file = os.path.join(LABELS_DIR, f"{stem}.txt")

        # Load existing boxes if label file exists
        existing_boxes = []
        if os.path.exists(label_file):
            try:
                with open(label_file, "r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # cls, xc, yc, w, h (normalized)
                            cls_id = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])
                            existing_boxes.append({
                                "cls": cls_id,
                                "xc": xc,
                                "yc": yc,
                                "w": w,
                                "h": h
                            })
            except Exception:
                pass

        rev_info = reviews.get(fname, {
            "reviewed": os.path.exists(label_file),
            "is_clean": os.path.exists(label_file) and len(existing_boxes) == 0,
            "num_boxes": len(existing_boxes),
            "timestamp": None
        })

        frames.append({
            "index": idx,
            "filename": fname,
            "reviewed": rev_info.get("reviewed", False),
            "is_clean": rev_info.get("is_clean", False),
            "num_boxes": len(existing_boxes),
            "boxes": existing_boxes
        })

    return jsonify({
        "frames": frames,
        "total": len(frames),
        "reviewed_count": sum(1 for f in frames if f["reviewed"]),
        "clean_count": sum(1 for f in frames if f["reviewed"] and f["is_clean"]),
        "pothole_frames_count": sum(1 for f in frames if f["reviewed"] and not f["is_clean"])
    })

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(CANDIDATES_DIR, filename)

@app.route("/api/save", methods=["POST"])
def save_annotation():
    data = request.json or {}
    fname = data.get("filename")
    boxes = data.get("boxes", [])  # list of {xc, yc, w, h} (normalized)
    is_clean = data.get("is_clean", False) or len(boxes) == 0

    if not fname:
        return jsonify({"error": "Filename required"}), 400

    stem = os.path.splitext(fname)[0]
    label_path = os.path.join(LABELS_DIR, f"{stem}.txt")

    # Write YOLO format label
    with open(label_path, "w", encoding="utf-8") as f:
        if not is_clean:
            for b in boxes:
                xc = float(np.clip(b.get("xc", 0.5), 0.0, 1.0))
                yc = float(np.clip(b.get("yc", 0.5), 0.0, 1.0))
                w = float(np.clip(b.get("w", 0.1), 0.001, 1.0))
                h = float(np.clip(b.get("h", 0.1), 0.001, 1.0))
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        else:
            # Clean frame: empty file
            f.write("")

    # Update Manifest
    manifest = load_manifest()
    manifest["reviews"][fname] = {
        "reviewed": True,
        "is_clean": is_clean,
        "num_boxes": 0 if is_clean else len(boxes),
        "boxes": [] if is_clean else boxes,
        "timestamp": time.time(),
        "date_str": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_manifest(manifest)

    return jsonify({
        "status": "success",
        "filename": fname,
        "label_path": label_path,
        "is_clean": is_clean,
        "num_boxes": 0 if is_clean else len(boxes)
    })

@app.route("/api/summary", methods=["GET"])
def get_summary():
    manifest = load_manifest()
    files = get_candidate_files()
    reviews = manifest.get("reviews", {})

    total = len(files)
    reviewed = 0
    clean = 0
    pothole_frames = 0
    total_boxes = 0

    details = []
    for f in files:
        rev = reviews.get(f)
        if rev and rev.get("reviewed"):
            reviewed += 1
            if rev.get("is_clean"):
                clean += 1
            else:
                pothole_frames += 1
                total_boxes += rev.get("num_boxes", 0)
            details.append({
                "filename": f,
                "status": "Clean Road (0 Potholes)" if rev.get("is_clean") else f"Potholes ({rev.get('num_boxes', 0)} boxes)",
                "boxes": rev.get("num_boxes", 0),
                "timestamp": rev.get("date_str")
            })
        else:
            details.append({
                "filename": f,
                "status": "Pending Review",
                "boxes": 0,
                "timestamp": None
            })

    return jsonify({
        "total_candidate_frames": total,
        "reviewed_count": reviewed,
        "pending_count": total - reviewed,
        "clean_frames_count": clean,
        "pothole_frames_count": pothole_frames,
        "total_pothole_boxes": total_boxes,
        "percent_complete": round((reviewed / max(1, total)) * 100, 1),
        "details": details
    })


# ======================================================================================
# FRONTEND UI
# ======================================================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RoadVision AI — Dataset V2 Human-in-the-Loop Annotator</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #090d16;
      --bg-card: #111827;
      --bg-surface: #1e293b;
      --border: #334155;
      --primary: #10b981;
      --primary-hover: #059669;
      --accent-blue: #06b6d4;
      --accent-amber: #f59e0b;
      --danger: #ef4444;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* TOP HEADER */
    header {
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 10px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 10;
    }

    .brand-title {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-title h1 {
      font-family: 'Outfit', sans-serif;
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #34d399, #06b6d4);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .badge-v2 {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.35);
      color: #34d399;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 9999px;
    }

    .stats-ribbon {
      display: flex;
      align-items: center;
      gap: 18px;
    }

    .stat-pill {
      font-size: 0.8rem;
      color: var(--text-muted);
      background: rgba(30, 41, 59, 0.6);
      padding: 5px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      display: flex;
      gap: 6px;
      align-items: center;
    }

    .stat-pill strong {
      color: var(--text-main);
      font-family: 'Fira Code', monospace;
    }

    .progress-bar-container {
      width: 140px;
      height: 8px;
      background: #1e293b;
      border-radius: 4px;
      overflow: hidden;
      border: 1px solid var(--border);
    }

    .progress-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #10b981, #06b6d4);
      width: 0%;
      transition: width 0.3s ease;
    }

    /* MAIN CONTAINER */
    .app-body {
      display: flex;
      flex: 1;
      height: calc(100vh - 60px);
      overflow: hidden;
    }

    /* LEFT FILMSTRIP SIDEBAR */
    .sidebar {
      width: 290px;
      background: var(--bg-card);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .sidebar-header {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .filmstrip {
      flex: 1;
      overflow-y: auto;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .frame-card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      cursor: pointer;
      display: flex;
      gap: 10px;
      align-items: center;
      transition: all 0.15s ease;
    }

    .frame-card:hover {
      border-color: #475569;
      transform: translateX(2px);
    }

    .frame-card.active {
      border-color: var(--primary);
      background: rgba(16, 185, 129, 0.08);
      box-shadow: 0 0 0 1px var(--primary);
    }

    .frame-thumb {
      width: 64px;
      height: 38px;
      border-radius: 4px;
      object-fit: cover;
      background: #000;
      border: 1px solid #334155;
    }

    .frame-info {
      flex: 1;
      font-size: 0.76rem;
      line-height: 1.3;
    }

    .frame-title {
      font-weight: 600;
      color: var(--text-main);
      display: flex;
      justify-content: space-between;
    }

    .status-tag {
      font-size: 0.68rem;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
      margin-top: 3px;
      display: inline-block;
    }

    .status-tag.pending {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .status-tag.potholes {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-tag.clean {
      background: rgba(6, 182, 212, 0.15);
      color: #22d3ee;
      border: 1px solid rgba(6, 182, 212, 0.3);
    }

    /* CENTER CANVAS WORKSPACE */
    .workspace {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: #060910;
      position: relative;
    }

    .workspace-toolbar {
      padding: 10px 20px;
      background: var(--bg-card);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .nav-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .btn {
      background: var(--bg-surface);
      color: var(--text-main);
      border: 1px solid var(--border);
      padding: 7px 14px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }

    .btn:hover {
      background: #334155;
      border-color: #64748b;
    }

    .btn-clean {
      background: rgba(6, 182, 212, 0.15);
      border-color: rgba(6, 182, 212, 0.4);
      color: #22d3ee;
      font-weight: 600;
    }

    .btn-clean:hover {
      background: rgba(6, 182, 212, 0.25);
      border-color: #22d3ee;
    }

    .btn-save {
      background: #10b981;
      border-color: #059669;
      color: #fff;
      font-weight: 600;
      box-shadow: 0 2px 10px rgba(16, 185, 129, 0.25);
    }

    .btn-save:hover {
      background: #059669;
    }

    .btn-danger {
      background: rgba(239, 68, 68, 0.15);
      border-color: rgba(239, 68, 68, 0.35);
      color: #f87171;
    }

    .btn-danger:hover {
      background: rgba(239, 68, 68, 0.25);
      border-color: #ef4444;
    }

    .canvas-container {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      overflow: hidden;
      padding: 16px;
    }

    .viewport-wrapper {
      position: relative;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.7);
      border-radius: 6px;
      overflow: hidden;
      border: 2px solid #334155;
    }

    #targetImage {
      display: block;
      max-height: calc(100vh - 200px);
      max-width: calc(100vw - 360px);
      user-select: none;
      -webkit-user-drag: none;
    }

    #drawCanvas {
      position: absolute;
      top: 0;
      left: 0;
      cursor: crosshair;
    }

    /* INSTRUCTIONS FOOTER */
    .instructions-footer {
      background: var(--bg-card);
      border-top: 1px solid var(--border);
      padding: 8px 20px;
      font-size: 0.75rem;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .shortcut-keys {
      display: flex;
      gap: 14px;
    }

    .shortcut-keys span kbd {
      background: #1e293b;
      border: 1px solid #475569;
      border-radius: 4px;
      padding: 2px 6px;
      color: #e2e8f0;
      font-family: 'Fira Code', monospace;
    }

    /* SUMMARY MODAL */
    .modal-overlay {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 100;
    }

    .modal-content {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      width: 780px;
      max-height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
    }

    .modal-header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .modal-body {
      padding: 20px 24px;
      overflow-y: auto;
      flex: 1;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-bottom: 24px;
    }

    .summary-card {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      text-align: center;
    }

    .summary-card h4 {
      font-size: 0.72rem;
      color: var(--text-muted);
      text-transform: uppercase;
      margin-bottom: 6px;
    }

    .summary-card .num {
      font-size: 1.5rem;
      font-weight: 700;
      font-family: 'Fira Code', monospace;
      color: var(--primary);
    }

    .manifest-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
    }

    .manifest-table th, .manifest-table td {
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }

    .manifest-table th {
      color: var(--text-muted);
      background: rgba(30, 41, 59, 0.5);
    }
  </style>
</head>
<body>

  <!-- TOP HEADER -->
  <header>
    <div class="brand-title">
      <h1>RoadVision AI</h1>
      <span class="badge-v2">Dataset V2 Human-in-the-Loop Annotator</span>
    </div>

    <div class="stats-ribbon">
      <div class="progress-bar-container" title="Verification Progress">
        <div class="progress-bar-fill" id="progressBar"></div>
      </div>
      <div class="stat-pill">
        Verified: <strong id="statReviewed">0 / 35</strong>
      </div>
      <div class="stat-pill">
        Pothole Boxes: <strong id="statBoxes" style="color: #34d399;">0</strong>
      </div>
      <div class="stat-pill">
        Clean Road Frames: <strong id="statClean" style="color: #22d3ee;">0</strong>
      </div>
      <button class="btn" id="btnOpenSummary" onclick="openSummaryModal()">📋 View Summary</button>
    </div>
  </header>

  <!-- APP BODY -->
  <div class="app-body">
    <!-- SIDEBAR FILMSTRIP -->
    <div class="sidebar">
      <div class="sidebar-header">
        <span>CANDIDATE FRAMES</span>
        <span id="sidebarCount">35 Frames</span>
      </div>
      <div class="filmstrip" id="filmstripList">
        <!-- Frame cards dynamically injected -->
      </div>
    </div>

    <!-- MAIN WORKSPACE -->
    <div class="workspace">
      <!-- WORKSPACE TOOLBAR -->
      <div class="workspace-toolbar">
        <div class="nav-group">
          <button class="btn" id="btnPrev" onclick="navigateFrame(-1)">◀ Prev (A)</button>
          <span style="font-size: 0.85rem; font-family: 'Fira Code', monospace; color: #94a3b8;" id="currentFrameLabel">Frame 1 of 35</span>
          <button class="btn" id="btnNext" onclick="navigateFrame(1)">Next (D) ▶</button>
        </div>

        <div class="nav-group">
          <button class="btn btn-clean" id="btnMarkClean" onclick="markAsClean()">
            ✅ Mark as Clean Road (0 Potholes) [C]
          </button>
          <button class="btn btn-danger" id="btnDeleteBox" onclick="deleteSelectedBox()">
            🗑️ Delete Box [Del]
          </button>
          <button class="btn" id="btnClearAll" onclick="clearAllBoxes()">
            Clear All
          </button>
          <button class="btn btn-save" id="btnSaveNext" onclick="saveAndNext()">
            💾 Save & Next (Enter)
          </button>
        </div>
      </div>

      <!-- CANVAS VIEWPORT -->
      <div class="canvas-container">
        <div class="viewport-wrapper" id="viewportWrapper">
          <img id="targetImage" src="" alt="Candidate Frame" onload="onImageLoaded()" />
          <canvas id="drawCanvas"></canvas>
        </div>
      </div>

      <!-- INSTRUCTIONS FOOTER -->
      <div class="instructions-footer">
        <div class="shortcut-keys">
          <span><kbd>Click & Drag</kbd> Draw Pothole Box</span>
          <span><kbd>Click Box</kbd> Select Box</span>
          <span><kbd>Del / Backspace</kbd> Delete Box</span>
          <span><kbd>C</kbd> Mark Clean (0 Potholes)</span>
          <span><kbd>Enter / Ctrl+S</kbd> Save & Advance</span>
          <span><kbd>A / D</kbd> Prev / Next Frame</span>
        </div>
        <div style="font-family: 'Fira Code', monospace;" id="activeStatusLabel">
          Ready
        </div>
      </div>
    </div>
  </div>

  <!-- SUMMARY MODAL -->
  <div class="modal-overlay" id="summaryModal">
    <div class="modal-content">
      <div class="modal-header">
        <h3 style="font-family: 'Outfit', sans-serif;">Dataset V2 Review & Annotation Summary</h3>
        <button class="btn" onclick="closeSummaryModal()">✕ Close</button>
      </div>
      <div class="modal-body">
        <div class="summary-grid">
          <div class="summary-card">
            <h4>Total Frames</h4>
            <div class="num" id="modalTotal">35</div>
          </div>
          <div class="summary-card">
            <h4>Verified Complete</h4>
            <div class="num" id="modalReviewed">0</div>
          </div>
          <div class="summary-card">
            <h4>Total Potholes</h4>
            <div class="num" id="modalBoxes" style="color: #34d399;">0</div>
          </div>
          <div class="summary-card">
            <h4>Clean Road (0 Potholes)</h4>
            <div class="num" id="modalClean" style="color: #22d3ee;">0</div>
          </div>
        </div>

        <table class="manifest-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Filename</th>
              <th>Verification Status</th>
              <th>Boxes Count</th>
              <th>Review Timestamp</th>
            </tr>
          </thead>
          <tbody id="modalTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let framesList = [];
    let currentIndex = 0;
    let currentBoxes = []; // array of {xc, yc, w, h} (normalized)
    let selectedBoxIndex = -1;
    let isDrawing = false;
    let startX = 0, startY = 0;
    let isDragging = false;
    let dragStartX = 0, dragStartY = 0;

    const img = document.getElementById('targetImage');
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');

    // Fetch initial frame list
    async function loadFrames() {
      try {
        const res = await fetch('/api/frames');
        const data = await res.json();
        framesList = data.frames || [];
        updateHeaderStats(data);
        renderSidebar();
        if (framesList.length > 0) {
          selectFrame(0);
        }
      } catch (err) {
        console.error("Failed to load frames:", err);
      }
    }

    function updateHeaderStats(data) {
      const reviewed = data.reviewed_count || 0;
      const total = data.total || 35;
      const pct = Math.round((reviewed / Math.max(1, total)) * 100);
      document.getElementById('statReviewed').innerText = `${reviewed} / ${total} (${pct}%)`;
      document.getElementById('progressBar').style.width = `${pct}%`;
      document.getElementById('statClean').innerText = data.clean_count || 0;
      
      let totalBoxes = 0;
      framesList.forEach(f => {
        if (f.reviewed && !f.is_clean) totalBoxes += (f.num_boxes || 0);
      });
      document.getElementById('statBoxes').innerText = totalBoxes;
    }

    function renderSidebar() {
      const list = document.getElementById('filmstripList');
      list.innerHTML = '';
      framesList.forEach((f, idx) => {
        const card = document.createElement('div');
        card.className = `frame-card ${idx === currentIndex ? 'active' : ''}`;
        card.id = `card_${idx}`;
        card.onclick = () => selectFrame(idx);

        let tagHtml = '<span class="status-tag pending">Pending</span>';
        if (f.reviewed) {
          if (f.is_clean) {
            tagHtml = '<span class="status-tag clean">Clean (0)</span>';
          } else {
            tagHtml = `<span class="status-tag potholes">${f.num_boxes} Pothole${f.num_boxes > 1 ? 's' : ''}</span>`;
          }
        }

        card.innerHTML = `
          <img class="frame-thumb" src="/images/${f.filename}" loading="lazy" />
          <div class="frame-info">
            <div class="frame-title">
              <span>#${String(idx + 1).padStart(2, '0')}</span>
              <span style="font-family: 'Fira Code', monospace; color: #64748b;">f${f.filename.match(/f(\\d+)/)?.[1] || '0'}</span>
            </div>
            ${tagHtml}
          </div>
        `;
        list.appendChild(card);
      });
    }

    function selectFrame(index) {
      if (index < 0 || index >= framesList.length) return;
      currentIndex = index;
      selectedBoxIndex = -1;

      // Update sidebar active class
      document.querySelectorAll('.frame-card').forEach((c, idx) => {
        c.classList.toggle('active', idx === currentIndex);
      });

      const frame = framesList[currentIndex];
      document.getElementById('currentFrameLabel').innerText = `Frame ${currentIndex + 1} of ${framesList.length}`;
      document.getElementById('activeStatusLabel').innerText = `Loading ${frame.filename}...`;

      currentBoxes = (frame.boxes || []).map(b => ({ ...b }));
      img.src = `/images/${frame.filename}`;
    }

    function onImageLoaded() {
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      document.getElementById('activeStatusLabel').innerText = `${framesList[currentIndex].filename} (${img.naturalWidth}x${img.naturalHeight}px)`;
      redrawCanvas();
    }

    window.addEventListener('resize', () => {
      if (img.complete && img.naturalWidth) {
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        redrawCanvas();
      }
    });

    // DRAWING LOGIC
    function redrawCanvas() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const W = canvas.width;
      const H = canvas.height;

      currentBoxes.forEach((b, idx) => {
        // b has normalized xc, yc, w, h
        const bx = (b.xc - b.w / 2) * W;
        const by = (b.yc - b.h / 2) * H;
        const bw = b.w * W;
        const bh = b.h * H;

        const isSelected = (idx === selectedBoxIndex);

        // Bounding Box
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeStyle = isSelected ? '#3b82f6' : '#10b981';
        ctx.fillStyle = isSelected ? 'rgba(59, 130, 246, 0.20)' : 'rgba(16, 185, 129, 0.15)';

        ctx.fillRect(bx, by, bw, bh);
        ctx.strokeRect(bx, by, bw, bh);

        // Label Badge
        ctx.fillStyle = isSelected ? '#3b82f6' : '#10b981';
        const labelText = `Pothole #${idx + 1}`;
        ctx.font = '600 11px Inter, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillRect(bx, Math.max(0, by - 18), textWidth + 10, 18);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, bx + 5, Math.max(13, by - 5));

        // Corner resize handles if selected
        if (isSelected) {
          drawHandle(bx, by);
          drawHandle(bx + bw, by);
          drawHandle(bx, by + bh);
          drawHandle(bx + bw, by + bh);
        }
      });
    }

    function drawHandle(x, y) {
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.fillRect(x - 4, y - 4, 8, 8);
      ctx.strokeRect(x - 4, y - 4, 8, 8);
    }

    // Canvas Mouse Events
    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const W = canvas.width;
      const H = canvas.height;

      // Check if clicked inside existing box
      let clickedIdx = -1;
      for (let i = currentBoxes.length - 1; i >= 0; i--) {
        const b = currentBoxes[i];
        const bx = (b.xc - b.w / 2) * W;
        const by = (b.yc - b.h / 2) * H;
        const bw = b.w * W;
        const bh = b.h * H;
        if (mx >= bx && mx <= bx + bw && my >= by && my <= by + bh) {
          clickedIdx = i;
          break;
        }
      }

      if (clickedIdx !== -1) {
        selectedBoxIndex = clickedIdx;
        isDragging = true;
        dragStartX = mx;
        dragStartY = my;
        redrawCanvas();
      } else {
        // Start drawing new box
        selectedBoxIndex = -1;
        isDrawing = true;
        startX = mx;
        startY = my;
        redrawCanvas();
      }
    });

    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const W = canvas.width;
      const H = canvas.height;

      if (isDrawing) {
        redrawCanvas();
        const curW = mx - startX;
        const curH = my - startY;
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#06b6d4';
        ctx.fillStyle = 'rgba(6, 182, 212, 0.20)';
        ctx.fillRect(startX, startY, curW, curH);
        ctx.strokeRect(startX, startY, curW, curH);
      } else if (isDragging && selectedBoxIndex !== -1) {
        const dx = (mx - dragStartX) / W;
        const dy = (my - dragStartY) / H;
        const b = currentBoxes[selectedBoxIndex];
        b.xc = Math.min(1.0 - b.w/2, Math.max(b.w/2, b.xc + dx));
        b.yc = Math.min(1.0 - b.h/2, Math.max(b.h/2, b.yc + dy));
        dragStartX = mx;
        dragStartY = my;
        redrawCanvas();
      }
    });

    canvas.addEventListener('mouseup', (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const W = canvas.width;
      const H = canvas.height;

      if (isDrawing) {
        isDrawing = false;
        const x1 = Math.min(startX, mx);
        const y1 = Math.min(startY, my);
        const x2 = Math.max(startX, mx);
        const y2 = Math.max(startY, my);
        const bw = x2 - x1;
        const bh = y2 - y1;

        // Minimum box size threshold (12px)
        if (bw > 12 && bh > 12) {
          const xc = (x1 + bw / 2) / W;
          const yc = (y1 + bh / 2) / H;
          const normW = bw / W;
          const normH = bh / H;

          currentBoxes.push({ cls: 0, xc, yc, w: normW, h: normH });
          selectedBoxIndex = currentBoxes.length - 1;
        }
        redrawCanvas();
      }
      isDragging = false;
    });

    // ACTION HANDLERS
    function deleteSelectedBox() {
      if (selectedBoxIndex >= 0 && selectedBoxIndex < currentBoxes.length) {
        currentBoxes.splice(selectedBoxIndex, 1);
        selectedBoxIndex = -1;
        redrawCanvas();
      }
    }

    function clearAllBoxes() {
      currentBoxes = [];
      selectedBoxIndex = -1;
      redrawCanvas();
    }

    async function markAsClean() {
      currentBoxes = [];
      selectedBoxIndex = -1;
      redrawCanvas();
      await saveCurrentFrame(true);
      navigateFrame(1);
    }

    async function saveAndNext() {
      await saveCurrentFrame(currentBoxes.length === 0);
      navigateFrame(1);
    }

    async function saveCurrentFrame(isClean = false) {
      const frame = framesList[currentIndex];
      const payload = {
        filename: frame.filename,
        boxes: isClean ? [] : currentBoxes,
        is_clean: isClean
      };

      try {
        const res = await fetch('/api/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const result = await res.json();

        // Update local state
        frame.reviewed = true;
        frame.is_clean = isClean;
        frame.num_boxes = isClean ? 0 : currentBoxes.length;
        frame.boxes = isClean ? [] : [...currentBoxes];

        // Refresh stats & sidebar
        renderSidebar();
        const statRes = await fetch('/api/frames');
        const statData = await statRes.json();
        updateHeaderStats(statData);

        document.getElementById('activeStatusLabel').innerText = `Saved ${frame.filename} (${isClean ? 'Clean Road' : currentBoxes.length + ' Potholes'})`;
      } catch (err) {
        console.error("Save failed:", err);
      }
    }

    function navigateFrame(delta) {
      const nextIdx = currentIndex + delta;
      if (nextIdx >= 0 && nextIdx < framesList.length) {
        selectFrame(nextIdx);
      }
    }

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        deleteSelectedBox();
      } else if (e.key === 'c' || e.key === 'C') {
        markAsClean();
      } else if (e.key === 'Enter' || (e.ctrlKey && e.key === 's')) {
        e.preventDefault();
        saveAndNext();
      } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
        navigateFrame(1);
      } else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
        navigateFrame(-1);
      }
    });

    // SUMMARY MODAL
    async function openSummaryModal() {
      try {
        const res = await fetch('/api/summary');
        const data = await res.json();

        document.getElementById('modalTotal').innerText = data.total_candidate_frames;
        document.getElementById('modalReviewed').innerText = `${data.reviewed_count} (${data.percent_complete}%)`;
        document.getElementById('modalBoxes').innerText = data.total_pothole_boxes;
        document.getElementById('modalClean').innerText = data.clean_frames_count;

        const tbody = document.getElementById('modalTableBody');
        tbody.innerHTML = '';
        data.details.forEach((item, idx) => {
          const row = document.createElement('tr');
          const isDone = item.status !== 'Pending Review';
          const isClean = item.status.includes('Clean');
          const badgeClass = !isDone ? 'pending' : (isClean ? 'clean' : 'potholes');

          row.innerHTML = `
            <td>${idx + 1}</td>
            <td style="font-family: 'Fira Code', monospace;">${item.filename}</td>
            <td><span class="status-tag ${badgeClass}">${item.status}</span></td>
            <td><strong>${item.boxes}</strong></td>
            <td style="color: #64748b;">${item.timestamp || '—'}</td>
          `;
          tbody.appendChild(row);
        });

        document.getElementById('summaryModal').style.display = 'flex';
      } catch (err) {
        console.error("Failed to load summary:", err);
      }
    }

    function closeSummaryModal() {
      document.getElementById('summaryModal').style.display = 'none';
    }

    // Initialize
    loadFrames();
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RoadVision AI Human-in-the-Loop Annotator Server")
    parser.add_argument("--port", type=int, default=5055, help="Port to run annotator web app")
    args = parser.parse_args()

    print("=" * 80)
    print("ROADVISION AI: HUMAN-IN-THE-LOOP DATASET V2 ANNOTATOR")
    print(f"Candidates Path : {CANDIDATES_DIR}")
    print(f"Labels Path     : {LABELS_DIR}")
    print(f"Review Manifest : {MANIFEST_PATH}")
    print(f"Local Server URL: http://localhost:{args.port}")
    print("=" * 80)

    app.run(host="127.0.0.1", port=args.port, debug=False)

if __name__ == "__main__":
    main()
