# IMU + GNSS + CNN Speed Estimator

Android app implementing the spec in `IMU_EKF_CNN_Speed_App.md`. Runs three
parallel speed estimators in real time:

1. **GPS** — raw GNSS speed.
2. **EKF** — CTRV Extended Kalman Filter fusing IMU + GNSS with Sage-Husa
   adaptive Q/R.
3. **EKF + CNN** — feeds a CNN-derived speed (from sliding FFT of accelerometer
   data) as an extra measurement into the EKF. The CNN head trains online from
   GNSS whenever the signal is good.

## Build

1. Open the project in Android Studio (Hedgehog / Iguana). Gradle 8.5, AGP 8.2.
2. Generate the TFLite asset (one-time):
   ```
   pip install tensorflow==2.14
   python scripts/generate_tflite.py
   cp speed_cnn.tflite app/src/main/assets/
   ```
   Skipping this step is OK — the app falls back to an average-pool embedding.
3. Build & run on a physical Android 8.0+ device (an emulator without real
   IMU/GNSS won't produce meaningful output).
4. Grant the location permission when prompted.

## Architecture

```
sensors/      ImuManager (200 Hz), GnssManager (1 Hz), SensorFusion
ekf/          FloatMatrix, EkfCore (CTRV), EkfImuGnss, SageHusa
cnn/          FftExtractor (sliding 512-sample Hann FFT),
              SpeedCnn (frozen TFLite backbone 786→64),
              OnlineTrainer (Kotlin SGD head 64→1),
              ReplayBuffer (diversity-sampled)
calibration/  CalibrationEngine (master HandlerThread),
              GnssQualityGate, BiasEstimator
ui/           SpeedViewModel, SpeedCard, MainActivity (Compose)
```

Threading:

- IMU callbacks → `ImuThread` HandlerThread (200 Hz)
- GNSS callbacks → main thread → posted to `CalibThread`
- CNN training → coroutine on `Dispatchers.Default`
- UI ticks at 10 Hz via the ViewModel

## Notes

- Option B from the spec is used for online training: TFLite backbone is
  frozen, the trainable head is a single dense layer (64→1) implemented in
  pure Kotlin. This avoids the experimental TFLite training APIs.
- HDOP is approximated from `Location.speedAccuracyMetersPerSecond` since the
  Android `Location` API doesn't expose it directly. For a real deployment,
  wire up `addNmeaListener` and parse `$GPGSA` instead.
- The CNN card shows `TRAINING…` until the replay buffer holds ≥ 100 samples.
- The trained head weights are persisted to `filesDir/speed_cnn_head.bin`
  after every training batch.
