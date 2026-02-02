# Cricket Biomechanics: Pose-Based Movement Analysis

A minimal pipeline for pose estimation and biomechanics metrics from side-on cricket batting or bowling video. No cloud, no UI, no billing—just video in, keypoints and metrics out.

---

## What This Is

I built this to show how an early-stage sports analytics product might reason about movement from a single phone-recorded clip: pose → keypoints → a few interpretable metrics. The goal is clear thinking and interpretability, not research-level accuracy or polish.

---

## Project Approach

1. **Video in**: One side-on or near side-on cricket video (batting or bowling). Phone-recorded is fine; I’m not assuming broadcast quality.
2. **Pose**: One pretrained model (MediaPipe Pose) runs per frame. No training or fine-tuning.
3. **Outputs**: Skeleton overlay video for sanity-check, frame-wise keypoints (CSV/JSON), and three movement metrics with short interpretation.
4. **Honest limits**: Jitter, occlusion, and depth ambiguity are called out; improvement is discussed in terms of data and evaluation, not magic.

---

## Model Choice: MediaPipe Pose

I use **MediaPipe Pose** (BlazePose via the PoseLandmarker task in MediaPipe 0.10+) because:

- Free, open-source, actively maintained; no API keys or billing.
- Runs on CPU; no GPU required. The script downloads the lite pose model once (from Google’s CDN) into a `models/` folder; no credit card or account.
- Good enough for a prototype: 33 landmarks, real-time capable.
- Well-documented; I use the Tasks API (PoseLandmarker) with VIDEO running mode.

I did not use RTMPose or YOLO Pose here to keep the stack small and the narrative focused. For a production path, I’d compare them on cricket-specific footage (occlusion, clothing, angles) before committing.

---

## Video Selection (Step 1)

**What I assume:**

- **Side-on or semi side-on** so that the front leg (and torso) are visible for most of the action.
- **Single player** in frame; camera reasonably stable.
- **Batting or bowling** action so “front” knee and “action phase” are meaningful.

**Why this is suitable:** Side-on gives a clear view of knee bend and torso lean without heavy perspective distortion. It’s how coaches often film for technique review.

**Where it falls short:**

- **Occlusion**: Bat, ball, or other players can hide joints; the model will guess or drop keypoints.
- **Depth**: Everything is 2D from the camera; “vertical” is image vertical, not gravity. Fine for relative angles and stability, not for true 3D loads.
- **Jitter**: Phone footage and per-frame inference cause small frame-to-frame jumps; I apply light temporal smoothing and report that it’s there.

If your clip has heavy occlusion or is fully front-on, I’d treat the metrics as indicative only and stress the need for better viewing angle or multi-camera setup.

---

## Pipeline (Steps 2 & 3)

- **Pose**: MediaPipe Pose per frame → 33 landmarks (image and world).
- **Temporal smoothing**: Optional 5-frame moving average on keypoint coordinates to reduce jitter. I use it by default and document it so reviewers know we’re not claiming raw keypoints are perfectly stable.
- **Outputs**:
  - Skeleton overlay video.
  - Frame-wise keypoints: CSV (and optionally JSON) with columns like `frame_id`, `landmark_id`, `x`, `y`, `z`, `visibility`.
- **Metrics** (see below): Computed from smoothed keypoints; action phase can be full clip or a specified frame range.

**Pose evaluation (brief):**

- **Keypoint stability**: MediaPipe’s built-in temporal smoothing plus an optional 5-frame moving average reduce frame-to-frame jitter. I use smoothing by default so angle and variance metrics aren’t dominated by noise; the trade-off is a slight delay on fast movements.
- **Missed detections**: The script reports “Frames with pose: X / Y”. If X is much smaller than Y, occlusion or framing is likely; metrics in those clips should be treated with caution.
- **Jitter**: Without smoothing, phone footage and per-frame inference often show small jumps. Smoothing is applied and documented so reviewers know we’re not claiming raw keypoints are perfectly stable.

---

## The Three Metrics

### 1. Front knee flexion angle over time

- **Definition**: Angle at the front knee (hip–knee–ankle). I use the knee as the vertex; the angle is in the leg plane (from 2D image coordinates).
- **Why it matters**: Knee bend affects load absorption, balance, and power transfer. Too straight and you lose shock absorption and flexibility; too bent and you can restrict drive or balance. Tracking it over time shows how the player uses the front leg through the action.

### 2. Range of motion (ROM) of the front knee

- **Definition**: Max knee angle minus min knee angle during the main action phase (e.g. backlift to follow-through).
- **Why it matters**: Low ROM can mean stiff or restricted movement; very high ROM might indicate instability or technique issues. In context, ROM helps compare consistency and “freedom” of the front leg across sessions or players.

### 3. Torso stability (angle variance)

- **Definition**: Variance of the torso angle over time. Torso angle = angle of the shoulder–hip line relative to image vertical (0° = upright).
- **Why it matters**: Lower variance suggests a more stable, repeatable posture; high variance can mean excess sway or inconsistency. It’s a simple proxy for balance and repeatability, not a replacement for full 3D analysis.

All three are reported with numbers plus a short plain-language interpretation in the script output.

---

## Observations and Limitations

- **Jitter**: Frame-to-frame keypoint noise; smoothing reduces it but can slightly delay rapid changes.
- **Occlusion**: Hidden joints produce low visibility or odd angles; I don’t mask these in the metrics (you could add visibility thresholds).
- **Incorrect keypoints**: Occasional mis-association or drift, especially with loose clothing or unusual poses; visual check via skeleton video is important.
- **2D**: Angles are in the image plane; left/right lean and true 3D angles would need multi-view or depth.

---

## Improvement Plan (If Given More Time and Data)

**Problems observed:** Jitter, occlusion, occasional wrong keypoints, 2D-only interpretation.

**Improving accuracy:**

- **Temporal models**: Use a short temporal window (e.g. 1D Conv or small RNN) on keypoint sequences to smooth and correct.
- **Visibility-aware metrics**: Down-weight or exclude frames where keypoint visibility is below a threshold.
- **Cricket-specific adaptation**: Fine-tune or train a pose model on cricket footage (batting/bowling, side-on and semi side-on) so that bat, pads, and typical poses are better handled.

**Data I’d collect:**

- **Players**: Multiple skill levels and body types.
- **Angles**: Side-on, semi side-on, and maybe one other fixed angle per session.
- **Conditions**: Indoor/outdoor, different clothing, different cameras/phones.

**Splits (avoid leakage):**

- **Train**: One set of players (and their clips).
- **Validation**: Different players, same protocol (angle, task).
- **Test**: Hold-out players, never seen during development.

So: player-wise split, not frame-wise or clip-wise random, so we measure generalization to new players.

**Evaluation:**

- **Keypoint consistency**: Variance of repeated annotations or model predictions on same frame (or very similar frames).
- **Metric stability**: Same player, same action type, multiple clips — metrics should be similar.
- **Interpretability**: Coaches or analysts can use the metrics to discriminate “good” vs “needs work” or to track change over time; that’s the practical test.

---

## How to Run

**Setup (no GPU required):**

```bash
pip install -r requirements.txt
```

**Input:** Put your side-on cricket video in the **`videos/`** folder (e.g. `videos/batting.mp4`).

**Run:**

```bash
python run_pipeline.py --video videos/batting.mp4
```

If the file is inside `videos/`, you can pass just the filename and the script will look there:

```bash
python run_pipeline.py --video batting.mp4
```

If the filename has spaces, quote the path: `--video "videos/Volleyball spike.mp4"`

Optional:

- `--front-leg left` or `--front-leg right`: which knee to treat as “front” (default: `left` for typical right-handed batsman side-on).
- `--smooth 5`: window size for temporal smoothing (default 5); use 0 to disable.
- `--action-phase 30 90`: start and end frame index for ROM and stability (default: full clip).
- `--out-dir ./output`: where to write overlay video and keypoints.

**Outputs:**

- `skeleton_overlay.mp4` (or name derived from input).
- `keypoints.csv` (and optionally `keypoints.json`).
- Printed metrics and a short interpretation.

---

## Deliverables Checklist

- [x] README: approach, model choice, metrics, limitations, improvement plan.
- [x] Skeleton overlay video (generated when you run the script).
- [x] Keypoints CSV/JSON.
- [x] Code: single script, no UI, readable and commented where it matters.

Clarity over perfection; this is the kind of thing I’d hand to a reviewer and say: “This is how I’d start; here’s what I’d do next with more time and data.”
