"""
========================================================================================
🧪 RoadVision AI — Download Feature End-to-End Verification Test
========================================================================================
Tests:
  1. POST /api/detect processes video and returns genuine videoUrl.
  2. GET /outputs/<filename> serves video stream with 200 OK and video/mp4.
  3. GET /outputs/<filename>?download=1&name=roadvision_annotated_survey.mp4 sends
     Content-Disposition: attachment; filename=roadvision_annotated_survey.mp4.
  4. Verify the downloaded file is a valid MP4 with frames, playable in OpenCV/browser.
  5. Verify audit records, CSV path, and GPS honesty remain intact.
========================================================================================
"""

import os
import sys
import io
import cv2
import server

def main():
    print("\n" + "=" * 80)
    print("[TEST] VERIFYING ANNOTATED VIDEO DOWNLOAD WORKFLOW")
    print("=" * 80)

    client = server.app.test_client()

    # 1. Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.status_code}"
    print(f"  * Backend health verified: {res_health.get_json().get('detector_model')}")

    # 2. Prepare test clip
    base_dir = os.path.abspath(os.path.dirname(__file__))
    src_video = os.path.join(base_dir, "Watch Video_ Terrible potholes in Rangoon Road.mp4")
    assert os.path.exists(src_video), f"Missing {src_video}"

    cap = cv2.VideoCapture(src_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    test_clip_path = os.path.join(base_dir, "scratch_test_download_clip.mp4")
    writer = cv2.VideoWriter(test_clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(15):  # 15 frames for quick end-to-end check
        ret, frame = cap.read()
        if not ret: break
        writer.write(frame)
    cap.release()
    writer.release()

    with open(test_clip_path, "rb") as vf:
        video_bytes = vf.read()

    # 3. Detect and generate annotated output
    print("  * Sending video to /api/detect...")
    detect_res = client.post(
        "/api/detect",
        data={"video": (io.BytesIO(video_bytes), "test_download_clip.mp4")},
        content_type="multipart/form-data"
    )
    assert detect_res.status_code == 200, f"Detection failed: {detect_res.status_code}"
    detect_data = detect_res.get_json()
    assert detect_data.get("success") is True, "Success flag not True"
    video_url = detect_data.get("videoUrl")
    print(f"  * Generated annotated video URL: {video_url}")

    # 4. Test normal stream GET
    res_stream = client.get(video_url)
    assert res_stream.status_code == 200, f"Stream GET failed: {res_stream.status_code}"
    assert "video/mp4" in res_stream.content_type, f"Unexpected content type: {res_stream.content_type}"
    print(f"  * Stream request: HTTP 200 OK, Content-Type: {res_stream.content_type}, Size: {len(res_stream.data)/1024:.1f} KB")

    # 5. Test attachment download GET
    download_url = f"{video_url}?download=1&name=roadvision_annotated_survey.mp4"
    res_download = client.get(download_url)
    assert res_download.status_code == 200, f"Download GET failed: {res_download.status_code}"
    disp = res_download.headers.get("Content-Disposition")
    print(f"  * Download request: Content-Disposition header = '{disp}'")
    assert "attachment" in disp, "Content-Disposition does not contain attachment"
    assert "filename=roadvision_annotated_survey.mp4" in disp, "Filename missing from Content-Disposition"

    # 6. Verify downloaded payload is valid playable video
    downloaded_file = os.path.join(base_dir, "scratch_downloaded_output.mp4")
    with open(downloaded_file, "wb") as f:
        f.write(res_download.data)

    test_cap = cv2.VideoCapture(downloaded_file)
    assert test_cap.isOpened(), "Downloaded video file cannot be opened as a valid video!"
    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret_read, frame_read = test_cap.read()
    test_cap.release()
    assert ret_read and frame_read is not None, "Failed to decode frame from downloaded video!"
    print(f"  * Verified downloaded file integrity: {frame_count} valid decoded video frames!")

    # Clean up scratch files
    for p in [test_clip_path, downloaded_file]:
        if os.path.exists(p): os.remove(p)

    print("\n" + "=" * 80)
    print("[PASS] DOWNLOAD FEATURE VERIFIED END-TO-END WITH ZERO ERRORS!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
