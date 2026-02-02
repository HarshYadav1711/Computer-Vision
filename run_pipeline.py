#!/usr/bin/env python3
"""
Cricket biomechanics pipeline: pose estimation + three movement metrics.
No UI, no cloud; video in, keypoints and metrics out.
Uses MediaPipe 0.10+ Tasks API (PoseLandmarker).
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# MediaPipe 0.10+ uses Tasks API
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# BlazePose 33-landmark connections (for skeleton overlay when solutions.pose not available)
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
])

# Landmark indices for metrics
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def get_pose_model_path():
    """Return path to pose_landmarker.task; download from Google if missing."""
    cache_dir = Path(__file__).resolve().parent / "models"
    cache_dir.mkdir(exist_ok=True)
    model_path = cache_dir / "pose_landmarker_lite.task"
    if model_path.is_file():
        return str(model_path)
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    print(f"Downloading pose model to {model_path} (one-time)...")
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as e:
        print(f"Download failed: {e}. Place pose_landmarker_lite.task in {cache_dir}.", file=sys.stderr)
        sys.exit(1)
    return str(model_path)


def angle_at_point(p_hip, p_knee, p_ankle, deg=True):
    """Angle at knee (hip–knee–ankle). Returns angle in [0, 180] degrees."""
    v1 = np.array([p_hip[0] - p_knee[0], p_hip[1] - p_knee[1]])
    v2 = np.array([p_ankle[0] - p_knee[0], p_ankle[1] - p_knee[1]])
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    rad = np.arccos(cos_a)
    return np.degrees(rad) if deg else rad


def torso_angle_deg(shoulder_mid, hip_mid, vertical_down=True):
    """Angle of shoulder–hip line relative to image vertical."""
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    rad = np.arctan2(dx, dy) if vertical_down else np.arctan2(dx, -dy)
    return np.degrees(rad)


def smooth_sequence(values, window=5):
    """Simple moving average; odd window. NaNs preserved as NaN."""
    if window <= 1 or len(values) == 0:
        return values
    out = np.full_like(values, np.nan, dtype=float)
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = values[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        out[i] = np.mean(valid) if len(valid) > 0 else np.nan
    return out


def extract_keypoints_from_landmarks(landmarks_list, h, w):
    """From PoseLandmarker result: one pose's landmarks -> list of dicts (pixel x,y)."""
    out = []
    for idx, lm in enumerate(landmarks_list):
        x = getattr(lm, "x", 0.0)
        y = getattr(lm, "y", 0.0)
        z = getattr(lm, "z", 0.0)
        vis = getattr(lm, "visibility", getattr(lm, "presence", 1.0))
        out.append({
            "landmark_id": idx,
            "x": x * w,
            "y": y * h,
            "z": z,
            "visibility": float(vis),
        })
    return out


def run_pose_on_video(video_path, smooth_window=5, front_leg="left"):
    """
    Run MediaPipe PoseLandmarker on video; optional temporal smoothing.
    front_leg: "left" or "right" — which knee to use as "front" for metrics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model_path = get_pose_model_path()
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    frames = []
    keypoints_per_frame = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(1000 * frame_idx / fps)
        try:
            result = detector.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            result = None
        frames.append(frame)
        if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
            kp = extract_keypoints_from_landmarks(result.pose_landmarks[0], h, w)
            keypoints_per_frame.append(kp)
        else:
            keypoints_per_frame.append([])
        frame_idx += 1

    cap.release()
    detector.close()

    n_frames = len(frames)
    if front_leg == "left":
        hip_idx, knee_idx, ankle_idx = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
    else:
        hip_idx, knee_idx, ankle_idx = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE

    hip_x = np.full(n_frames, np.nan)
    hip_y = np.full(n_frames, np.nan)
    knee_x = np.full(n_frames, np.nan)
    knee_y = np.full(n_frames, np.nan)
    ankle_x = np.full(n_frames, np.nan)
    ankle_y = np.full(n_frames, np.nan)
    shoulder_mid_x = np.full(n_frames, np.nan)
    shoulder_mid_y = np.full(n_frames, np.nan)
    hip_mid_x = np.full(n_frames, np.nan)
    hip_mid_y = np.full(n_frames, np.nan)

    for i, kp_list in enumerate(keypoints_per_frame):
        if len(kp_list) <= max(hip_idx, knee_idx, ankle_idx, RIGHT_SHOULDER, RIGHT_HIP):
            continue
        hip_x[i] = kp_list[hip_idx]["x"] / w
        hip_y[i] = kp_list[hip_idx]["y"] / h
        knee_x[i] = kp_list[knee_idx]["x"] / w
        knee_y[i] = kp_list[knee_idx]["y"] / h
        ankle_x[i] = kp_list[ankle_idx]["x"] / w
        ankle_y[i] = kp_list[ankle_idx]["y"] / h
        shoulder_mid_x[i] = (kp_list[LEFT_SHOULDER]["x"] + kp_list[RIGHT_SHOULDER]["x"]) / 2 / w
        shoulder_mid_y[i] = (kp_list[LEFT_SHOULDER]["y"] + kp_list[RIGHT_SHOULDER]["y"]) / 2 / h
        hip_mid_x[i] = (kp_list[LEFT_HIP]["x"] + kp_list[RIGHT_HIP]["x"]) / 2 / w
        hip_mid_y[i] = (kp_list[LEFT_HIP]["y"] + kp_list[RIGHT_HIP]["y"]) / 2 / h

    if smooth_window > 1:
        hip_x = smooth_sequence(hip_x, smooth_window)
        hip_y = smooth_sequence(hip_y, smooth_window)
        knee_x = smooth_sequence(knee_x, smooth_window)
        knee_y = smooth_sequence(knee_y, smooth_window)
        ankle_x = smooth_sequence(ankle_x, smooth_window)
        ankle_y = smooth_sequence(ankle_y, smooth_window)
        shoulder_mid_x = smooth_sequence(shoulder_mid_x, smooth_window)
        shoulder_mid_y = smooth_sequence(shoulder_mid_y, smooth_window)
        hip_mid_x = smooth_sequence(hip_mid_x, smooth_window)
        hip_mid_y = smooth_sequence(hip_mid_y, smooth_window)

    return {
        "frames": frames,
        "keypoints_per_frame": keypoints_per_frame,
        "fps": fps,
        "width": w,
        "height": h,
        "hip_x": hip_x, "hip_y": hip_y,
        "knee_x": knee_x, "knee_y": knee_y,
        "ankle_x": ankle_x, "ankle_y": ankle_y,
        "shoulder_mid_x": shoulder_mid_x, "shoulder_mid_y": shoulder_mid_y,
        "hip_mid_x": hip_mid_x, "hip_mid_y": hip_mid_y,
        "front_leg": front_leg,
    }


def compute_metrics(data, action_start=None, action_end=None):
    """
    Metric 1: Front knee flexion angle per frame.
    Metric 2: ROM of front knee (max - min) in action phase.
    Metric 3: Torso stability = variance of torso angle in action phase.
    """
    n = len(data["frames"])
    if action_start is None:
        action_start = 0
    if action_end is None:
        action_end = n
    action_start = max(0, min(action_start, n))
    action_end = max(action_start, min(action_end, n))

    h, w = data["height"], data["width"]
    hip_x = data["hip_x"] * w
    hip_y = data["hip_y"] * h
    knee_x = data["knee_x"] * w
    knee_y = data["knee_y"] * h
    ankle_x = data["ankle_x"] * w
    ankle_y = data["ankle_y"] * h
    sh_x = data["shoulder_mid_x"] * w
    sh_y = data["shoulder_mid_y"] * h
    hp_x = data["hip_mid_x"] * w
    hp_y = data["hip_mid_y"] * h

    knee_angles = np.array([
        angle_at_point(
            (hip_x[i], hip_y[i]),
            (knee_x[i], knee_y[i]),
            (ankle_x[i], ankle_y[i]),
        )
        for i in range(n)
    ])

    action_angles = knee_angles[action_start:action_end]
    valid = action_angles[~np.isnan(action_angles)]
    rom = float(np.ptp(valid)) if len(valid) > 0 else np.nan

    torso_angles = np.array([
        torso_angle_deg((sh_x[i], sh_y[i]), (hp_x[i], hp_y[i]))
        for i in range(n)
    ])
    action_torso = torso_angles[action_start:action_end]
    valid_torso = action_torso[~np.isnan(action_torso)]
    torso_var = float(np.var(valid_torso)) if len(valid_torso) > 0 else np.nan

    return {
        "knee_angle_per_frame": knee_angles,
        "rom_degrees": rom,
        "torso_angle_variance": torso_var,
        "action_start": action_start,
        "action_end": action_end,
        "torso_angle_per_frame": torso_angles,
    }


def write_keypoints_csv(keypoints_per_frame, out_path, width, height):
    """Frame-wise keypoints: one row per (frame_id, landmark_id)."""
    rows = []
    for frame_id, kp_list in enumerate(keypoints_per_frame):
        for kp in kp_list:
            rows.append({
                "frame_id": frame_id,
                "landmark_id": kp["landmark_id"],
                "x": kp["x"],
                "y": kp["y"],
                "z": kp["z"],
                "visibility": kp["visibility"],
            })
    if not rows:
        return
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "landmark_id", "x", "y", "z", "visibility"])
        writer.writeheader()
        writer.writerows(rows)


def write_keypoints_json(keypoints_per_frame, out_path):
    """Frame-wise keypoints as list of lists (one list per frame)."""
    def to_serializable(kp_list):
        return [
            {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in kp.items()}
            for kp in kp_list
        ]
    payload = [to_serializable(kp_list) for kp_list in keypoints_per_frame]
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=0)


def draw_skeleton_opencv(frame, kp_list, w, h):
    """Draw skeleton on frame using OpenCV and POSE_CONNECTIONS."""
    if not kp_list or len(kp_list) < 33:
        return
    pts = [(int(kp["x"]), int(kp["y"])) for kp in kp_list]
    for (i, j) in POSE_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)


def write_skeleton_video(frames, keypoints_per_frame, out_path, fps, w, h):
    """Draw skeleton on each frame and write video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for frame, kp_list in zip(frames, keypoints_per_frame):
        vis = frame.copy()
        draw_skeleton_opencv(vis, kp_list, w, h)
        out.write(vis)
    out.release()


def print_interpretation(metrics, front_leg):
    """Plain-language interpretation of the three metrics."""
    rom = metrics["rom_degrees"]
    torso_var = metrics["torso_angle_variance"]
    knee_angles = metrics["knee_angle_per_frame"]
    valid_knee = knee_angles[~np.isnan(knee_angles)]

    print("\n--- Metric interpretation ---")
    print("1) Front knee flexion over time: computed per frame (see keypoints CSV + metrics_summary.json).")
    if len(valid_knee) > 0:
        print(f"   Mean knee angle (full clip): {np.mean(valid_knee):.1f}°; range [{np.nanmin(knee_angles):.1f}°, {np.nanmax(knee_angles):.1f}°].")
    print("   Knee bend affects load absorption, balance, and power transfer; tracking over time shows how the front leg is used.")

    rom_str = f"{rom:.1f}°" if not np.isnan(rom) else "N/A (no valid pose in action phase)"
    print(f"\n2) ROM of front knee (action phase): {rom_str}.")
    if not np.isnan(rom):
        if rom < 20:
            print("   Low ROM: movement may be stiff or restricted; could limit drive or flexibility.")
        elif rom > 70:
            print("   High ROM: large range; ensure it's controlled and not from instability or technique issues.")
        else:
            print("   ROM in a typical band; interpret in context of skill level and intent.")

    torso_str = f"{torso_var:.2f} (deg^2)" if not np.isnan(torso_var) else "N/A (no valid pose in action phase)"
    print(f"\n3) Torso stability (angle variance): {torso_str}.")
    if not np.isnan(torso_var):
        if torso_var < 10:
            print("   Low variance: relatively stable torso; good for repeatability and balance.")
        elif torso_var > 50:
            print("   High variance: more sway or inconsistency; may affect repeatability.")
        else:
            print("   Moderate variance; compare across sessions or players for context.")
    print("---\n")


# Default folder for input videos (relative to script location)
VIDEOS_DIR = "videos"


def resolve_video_path(video_arg):
    """Resolve --video: if path does not exist, try VIDEOS_DIR / basename."""
    path = Path(video_arg)
    if path.is_file():
        return str(path)
    # Try videos/<basename> relative to project root (script's parent)
    project_root = Path(__file__).resolve().parent
    fallback = project_root / VIDEOS_DIR / path.name
    if fallback.is_file():
        return str(fallback)
    return str(path)  # Return original so error message shows what user passed


def main():
    parser = argparse.ArgumentParser(description="Cricket biomechanics: pose + 3 metrics")
    parser.add_argument("--video", required=True,
                        help='Path to video (e.g. videos/batting.mp4). Quote paths with spaces: "videos/My clip.mp4"')
    parser.add_argument("--out-dir", default="./output", help="Output directory")
    parser.add_argument("--front-leg", choices=["left", "right"], default="left",
                        help="Which knee is 'front' (left = typical right-handed batsman side-on)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Temporal smoothing window (0 = off)")
    parser.add_argument("--action-phase", nargs=2, type=int, default=None,
                        help="Start and end frame index for ROM and torso variance (default: full clip)")
    parser.add_argument("--no-json", action="store_true", help="Skip writing keypoints JSON")
    args = parser.parse_args()

    video_path = resolve_video_path(args.video)
    if not os.path.isfile(video_path):
        print(f"Error: video not found: {args.video}")
        print("Place your video in the 'videos/' folder and run e.g.: python run_pipeline.py --video videos/your_clip.mp4")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(video_path).stem
    smooth_window = max(0, args.smooth)
    action_phase = args.action_phase

    print("Running pose estimation (MediaPipe PoseLandmarker)...")
    data = run_pose_on_video(video_path, smooth_window=smooth_window, front_leg=args.front_leg)
    n_frames = len(data["frames"])
    n_detected = sum(1 for kp in data["keypoints_per_frame"] if kp)
    print(f"Frames: {n_frames}, with pose: {n_detected}. Smoothing: {smooth_window} frames.")

    csv_path = out_dir / "keypoints.csv"
    write_keypoints_csv(data["keypoints_per_frame"], str(csv_path), data["width"], data["height"])
    print(f"Keypoints CSV: {csv_path}")

    if not args.no_json:
        json_path = out_dir / "keypoints.json"
        write_keypoints_json(data["keypoints_per_frame"], str(json_path))
        print(f"Keypoints JSON: {json_path}")

    start_f, end_f = (action_phase[0], action_phase[1]) if action_phase else (None, None)
    metrics = compute_metrics(data, action_start=start_f, action_end=end_f)

    metrics_summary = {
        "rom_degrees": metrics["rom_degrees"],
        "torso_angle_variance": metrics["torso_angle_variance"],
        "action_start": metrics["action_start"],
        "action_end": metrics["action_end"],
        "front_leg": data["front_leg"],
    }
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics summary: {out_dir / 'metrics_summary.json'}")

    overlay_path = out_dir / f"{base_name}_skeleton.mp4"
    write_skeleton_video(
        data["frames"],
        data["keypoints_per_frame"],
        str(overlay_path),
        data["fps"],
        data["width"],
        data["height"],
    )
    print(f"Skeleton overlay: {overlay_path}")

    print_interpretation(metrics, data["front_leg"])
    return 0


if __name__ == "__main__":
    exit(main())
