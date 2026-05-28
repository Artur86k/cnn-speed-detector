# cnn_speed_detector

CNN that estimates vehicle speed from smartphone IMU (accelerometer + gyroscope)
for GNSS-denied dead reckoning. The repo holds two halves of the system:

1. **A Python training pipeline** (`scripts/`) — preprocesses raw Android
   IMU/GNSS recordings, builds windowed training tensors, trains a small 1D CNN,
   exports a TFLite model.
2. **An Android test app** (`app/`) — runs the trained TFLite on-device and
   displays GPS speed alongside the CNN prediction, plus a stats page with
   live-recorded MAE / RMSE / bias / per-bin error.

The Python pipeline mirrors the quality criteria from the offline MATLAB EKF
at `E:\Project\offline_navigation_system\matlab\ekf_car_nav.m` so on-device
inference and offline evaluation are consistent.

---

## Performance (held-out session 26, `navlog_20260526_224145`)

Latest model: 28 sessions, ~88k windows, trained with `--rebalance`.

| Speed bin | MAE | Notes |
|---|---|---|
| 0–18 km/h | 1.2 km/h | stops + low urban |
| 18–54 km/h | 3.4 km/h | urban |
| 54–90 km/h | 6.7 km/h | suburban / rural |
| **>90 km/h** | **18 km/h, bias −18** | one-drive variety bottleneck |
| Overall | 5.9 km/h | |

The high-speed gap is a **data-variety** problem (above 126 km/h the corpus is
backed by a single drive) — collecting 1–2 more highway sessions on different
roads/cars/phone mounts is expected to close it.

---

## Python pipeline

Required: Python 3.10+, `tensorflow ~= 2.21`, `numpy`, `pandas`, `matplotlib`.

Raw IMU/GNSS recordings (one folder per drive) are produced by the companion
Android NavRecorder (sister project, not in this repo) and live under
`samples/` or `E:\Project\offline_navigation_system\android_recorder\samples\`.
Each session folder contains `accel.csv`, `gyro.csv`, `linacc.csv`,
`gamerot.csv`, `mag.csv`, `gnss.csv` on a shared `elapsed_ns` clock.

### 1. Preprocess

```powershell
python scripts/preprocess.py `
    --samples "E:\Project\offline_navigation_system\android_recorder\samples" `
    --out     "E:\Project\offline_navigation_system\android_recorder\samples\preprocessed" `
    --no-merge
```

Per session, the preprocessor:

- detects stationary segments and estimates phone-body gravity from them,
- rotates phone → ENU world frame via `gamerot` (game rotation, mag-free),
- rotates ENU → vehicle frame (forward / lateral / vertical) using GPS bearing
  while moving, gamerot azimuth as fallback,
- runs the same spoof detector as the MATLAB EKF (hAcc ramp, position jump,
  IMU/GPS heading divergence),
- resamples everything to 200 Hz,
- writes `<session>.npz` with per-sample `valid_label` boolean and `speed_ms`.

### 2. Build training windows

```powershell
python scripts/dataset.py `
    --dir       "E:\Project\offline_navigation_system\android_recorder\samples\preprocessed" `
    --blocklist navlog_20260515_013747 `
    --min-valid-frac         0.95 `
    --min-session-valid-frac 0.10
```

Produces `windows.npz` with `(N, T=400, C=6)` features (`accel_veh` + `gyro`,
200 Hz × 2 s) and `(N,)` speed labels. Sessions whose own `valid_label`
fraction is below 10 % are skipped automatically.

> ⚠ Always blocklist `navlog_20260515_013747` — it is a 24-hour accidental
> recording of daily life (walking, scrolling, sleeping), not driving.

### 3. Train

```powershell
python scripts/train.py `
    --windows     "E:\Project\offline_navigation_system\android_recorder\samples\preprocessed\windows.npz" `
    --out         "E:\Project\offline_navigation_system\android_recorder\samples\preprocessed" `
    --val-session 26 `
    --epochs 30 --batch 128 `
    --rebalance
```

`--rebalance` applies per-bin inverse-frequency sample weighting — essential
for highway-speed coverage; without it the model under-predicts above 90 km/h
by 10–20 km/h.

Outputs (written to `--out`):

- `speed_cnn.keras` — Keras model
- `speed_cnn_smoketest.tflite` — TFLite (float32; do **not** enable
  `Optimize.DEFAULT` — see _Known constraints_ below)
- `train_history.npz` — loss/MAE curves + the per-channel mean/std required
  at inference time

### 4. Evaluate

```powershell
python scripts/evaluate.py `
    --windows     "...\windows.npz" `
    --model       "...\speed_cnn.keras" `
    --hist        "...\train_history.npz" `
    --val-session 26
```

Renders a 4-panel PNG: pred-vs-truth time series, scatter, residual histogram,
per-bin MAE bars.

---

## Android test app

Open `E:\Project\cnn_speed_detector\` in Android Studio (Hedgehog/Iguana,
AGP 8.2, Gradle 8.5). Build & run on a real Android 8.0+ device with GNSS.

```
app/src/main/
├── assets/
│   ├── speed_cnn_full.tflite     ← bundled trained model (118 KB)
│   └── cnn_norm.json             ← per-channel mean/std for normalization
└── java/com/cnnspeed/app/
    ├── MainActivity.kt            ← 2-screen Compose nav
    ├── sensors/GnssManager.kt
    ├── test/
    │   ├── SpeedCnnFull.kt        ← TFLite wrapper + normalization
    │   ├── WindowSampler.kt       ← sensor listener → 2 s vehicle-frame ring
    │   ├── TestEngine.kt          ← coordinator + StateFlow
    │   └── TestSession.kt         ← stat accumulator
    └── ui/TestScreens.kt          ← LiveScreen + StatsScreen + TestViewModel
```

The on-device pipeline mirrors `preprocess.py`:

- `TYPE_LINEAR_ACCELERATION` (gravity pre-removed by OS) for the accelerometer
  channels.
- `TYPE_GAME_ROTATION_VECTOR` (magnetometer-free) for orientation — same
  source as `gamerot.csv` used in training.
- Vehicle-frame rotation: GPS bearing where moving, gamerot azimuth fallback.
- Window: 400 samples × 6 channels, refreshed every 250 ms.

### Live screen

Two large text fields: GPS speed (cyan) and CNN prediction (orange).
A `START REC` button captures `(timestamp, gps_speed, cnn_pred)` triples;
`VIEW STATS` opens the second screen.

### Stats screen

Live MAE/RMSE/bias plus three charts: time-series overlay, scatter pred-vs-
truth with diagonal, and per-bin MAE bars (0–1 / 1–5 / 5–15 / 15–25 / 25+ m/s).

---

## Updating the bundled model after retraining

```powershell
cp "...\preprocessed\speed_cnn_smoketest.tflite" `
   "app\src\main\assets\speed_cnn_full.tflite"
# then update mean/std in app\src\main\assets\cnn_norm.json from
# train_history.npz (norm_mean / norm_std)
```

Build → Clean Project in Android Studio, then Run. Both files must always be
updated together — normalization stats are training-set-specific.

---

## Known constraints

- **`LABEL_SPDACC_MAX` in `preprocess.py` must stay ≥ 1.0 m/s.** The original
  0.5 threshold silently rejected every GNSS fix above ~25 m/s because Doppler
  speed accuracy degrades with velocity.
- **Never re-enable `converter.optimizations = [tf.lite.Optimize.DEFAULT]`.**
  TF 2.21's dynamic-range quantizer emits `FULLY_CONNECTED v12`, which the
  app's bundled `org.tensorflow:tensorflow-lite:2.16.1` runtime does not
  support and refuses to load. The trained TFLite must stay float32.
- **High-speed (>90 km/h) accuracy depends on collecting more highway drives
  on different roads/cars/phone mounts.** Above 126 km/h the current corpus
  has only one drive's worth of spectrum.

---

## Repo layout

```
.
├── README.md                  this file
├── app/                       Android app (Kotlin / Compose)
├── scripts/                   Python training pipeline
│   ├── preprocess.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
├── samples/                   gitignored — raw recordings + preprocessed NPZ
├── build.gradle.kts
├── settings.gradle.kts
└── .gitignore
```
