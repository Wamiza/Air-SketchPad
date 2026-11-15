# Hand-Gesture Air Sketchpad

Draw on a virtual canvas using only hand gestures in front of a webcam. A trained gesture
classifier (chosen from KNN, SVM, Random Forest and MLP) turns MediaPipe hand landmarks into
commands: **Draw, Pause, Colour, Erase, Clear** (plus **Idle**).

This code implements the problem formulation in `ML_Problem_Formulation_Wamiza_Nomn.pdf`:

| Formulation section | Where it lives |
|---|---|
| 2.1 Pipeline (frame -> landmarks -> features -> classifier -> filter -> action) | `air_sketchpad.py` |
| 2.2 Input: 21 landmarks -> 63-d wrist-centred, scale-normalised vector | `features.py`, `hand_detector.py` |
| 2.3 Output: 6 gestures + smoothed cursor | `features.py`, `engine.py` |
| 2.4 Dataset (self-collected) and candidate models | `collect_data.py`, `train_compare.py` |
| 2.5 Temporal decoding, confidence gating, hold times, cursor EMA | `engine.py` |
| 2.6 Metrics, participant-wise split | `train_compare.py` |
| 2.7 Model comparison protocol, selection rule, ablation | `train_compare.py` |
| Rule-based baseline | `rule_based.py` |

## 1. Setup

Python **3.9 - 3.12** is needed for the pinned MediaPipe version.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Allow camera access for your terminal/editor if your OS asks (macOS: System Settings -> Privacy -> Camera).

## 2. Collect your own data (do this first)

```bash
python collect_data.py --participant P01
```

Press **1-6** to record Idle / Draw / Pause / Colour / Erase / Clear. After a 3-second countdown it
records 300 frames of that gesture and stops by itself. Press `q` to save and quit. Only landmarks are
saved (`data/landmarks.csv`), never images.

| Gesture | Hand pose |
|---|---|
| 1 Idle | anything else: relaxed hand, thumbs-up, pointing at the screen, half-closed hand... |
| 2 Draw | index finger only |
| 3 Pause | index + middle fingers |
| 4 Colour | index + middle + ring fingers |
| 5 Erase | closed fist |
| 6 Clear | open palm |

To get data that generalises: use **at least 5 different people** (friends, family, classmates), and
let each one move closer/farther, rotate the hand a little, and use different lighting. Record **Idle**
with many different poses. Rule of thumb: about 300 frames per gesture per person, so around 9,000 rows.
Run the command once per person with a new `--participant` id (P02, P03, ...).

## 3. Train and compare the models

```bash
python train_compare.py --data data/landmarks.csv --ablation
```

What it does:
* holds out some participants as a **test set** (default about 20%; or choose with `--test-participants P05`),
* tunes KNN, SVM (RBF), Random Forest and MLP with participant-grouped cross-validation (macro-F1),
* evaluates every model and the rule-based baseline on the held-out participants,
* applies the selection rule (best validation macro-F1; if models are within one standard deviation,
  the faster one) and saves the winner to `models/best_model.joblib`,
* with `--ablation`, compares raw landmarks against the normalised features.

Outputs in `results/`: `comparison.csv`, `report.txt` (includes a per-class report), `confusion_matrices.png`,
`model_comparison.png`. **These are the real results to quote in your video and report.**
Add `--quick` for smaller grids if it is slow.

## 4. Run the app

```bash
python air_sketchpad.py
```

Keys: `q`/Esc quit, `c` clear, `s` save drawing as PNG, `r` switch between the trained model and the
rule-based baseline (nice for a live comparison in the video). Hold the open palm for one second to clear.
Without a trained model the app uses the rule-based baseline automatically. Other options:
`--camera 1` (another camera), `--video file.mp4`, `--rule-based`.

Tuning constants (smoothing window, confidence threshold, hold times, active region) are in `EngineConfig`
in `engine.py`.

## 5. Dry run without a webcam or real data

```bash
python make_synthetic_data.py          # writes data/synthetic_landmarks.csv
python train_compare.py --data data/synthetic_landmarks.csv --quick
pytest -q                              # unit + end-to-end tests
```

**Warning:** the synthetic hands are artificial and almost perfectly separable, so scores on them mean
nothing. Never report them. They only prove the pipeline runs.

## What has and has not been tested

Tested automatically (`pytest`): feature normalisation (translation, scale and left/right invariance),
the gesture engine (vote smoothing, confidence gating, hold times, Clear fires once after 1 s and never
on a brief pose, edge-triggered Colour), the canvas, the whole app loop with a fake detector, and the
training script end to end. The real MediaPipe detector was also loaded and run on a test video.

**Not tested here:** a live webcam, real human hands, and real-data accuracy. Expect to adjust things after
your first real session, for example the rule-based thresholds in `rule_based.py` or `EngineConfig`
values. If you use Python 3.13+ with a newer MediaPipe, `hand_detector.py` falls back to the Tasks API and
downloads `hand_landmarker.task` on first run (this fallback is untested).

## Notes and small differences from the written formulation

* The MLP is scikit-learn's `MLPClassifier` (softmax output, cross-entropy loss, L2 `alpha`, Adam). It has no
  per-class weights; SVM and Random Forest use `class_weight="balanced"`. Recording equal frames per gesture
  keeps the classes roughly balanced anyway.
* Left hands are mirrored into a right-hand frame, so one classifier handles both hands.
* The Erase gesture moves the cursor with the index fingertip (curled in a fist), so place the fist so the
  curled index finger is over what you want to erase.

## Files

```
air_sketchpad.py     real-time app            collect_data.py   record labelled landmarks
train_compare.py     train + compare models   engine.py         temporal filter + cursor
features.py          constants + features     canvas.py         drawing canvas
rule_based.py        baseline                 hand_detector.py  MediaPipe wrapper
make_synthetic_data.py  dev-only test data    tests/            pytest suite
```
