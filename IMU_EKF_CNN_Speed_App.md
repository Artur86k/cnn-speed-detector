# Android IMU Speed Estimator — Claude Code Spec

## Project Overview

Build an Android app (Kotlin, minSdk 26, targetSdk 34) that runs **three parallel speed estimators** in real time and displays them as live text fields. The system continuously self-calibrates using GNSS when signal quality is good.

---

## UI Layout

Single-screen `MainActivity` with a dark-themed dashboard aesthetic. Three card panels stacked vertically, each showing a speed value and status indicator:

```
┌─────────────────────────────────────────┐
│  📡  GPS Speed (Raw GNSS)               │
│      [ 72.4 km/h ]    HDOP: 1.2  9 sat  │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│  🔵  EKF Speed (IMU + GNSS calibrated)  │
│      [ 71.8 km/h ]    Q: 0.012  R: 0.34 │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│  🧠  EKF+CNN Speed (Neural Network)     │
│      [ 72.1 km/h ]    CNN loss: 0.018   │
└─────────────────────────────────────────┘

[● CALIBRATING]  GNSS good · 9 sats · HDOP 1.2
```

- Update rate: 10 Hz display refresh
- All speed values in km/h with 1 decimal place
- Status row at bottom: GNSS quality indicator, calibration mode (CALIBRATING / COASTING)
- A small scrolling log panel at the bottom (last 5 lines) for debug output

---

## Architecture

```
app/
├── MainActivity.kt
├── sensors/
│   ├── GnssManager.kt          # GNSS speed + quality metrics
│   ├── ImuManager.kt           # Accelerometer + gyroscope + rotation vector
│   └── SensorFusion.kt         # Coordinates all sensor streams
├── ekf/
│   ├── EkfCore.kt              # Base EKF (CTRV model)
│   ├── EkfImuGnss.kt           # EKF #2: IMU + GNSS, adaptive Q/R
│   └── SageHusa.kt             # Online Q/R estimation
├── cnn/
│   ├── FftExtractor.kt         # Sliding FFT on IMU windows
│   ├── SpeedCnn.kt             # TFLite model wrapper
│   ├── OnlineTrainer.kt        # Online fine-tuning via TFLite GPU delegate
│   └── ReplayBuffer.kt         # Circular buffer of (features, gnss_speed)
├── calibration/
│   ├── CalibrationEngine.kt    # Master calibration controller
│   ├── GnssQualityGate.kt      # HDOP / sat / speed thresholds
│   └── BiasEstimator.kt        # Accel/gyro bias tracking
└── ui/
    ├── SpeedViewModel.kt        # LiveData for UI updates
    └── DashboardFragment.kt
```

---

## 1. GNSS Manager (`GnssManager.kt`)

Use `LocationManager` with `GPS_PROVIDER` and register a `GnssStatus.Callback`.

**Outputs per fix:**
- `speedMs: Float` — from `Location.speed` (m/s), convert to km/h for display
- `bearing: Float` — `Location.bearing` (degrees, 0=North)
- `hdop: Float` — parse from `GnssStatus` or NMEA `$GPGSA`
- `satCount: Int` — satellites used in fix
- `speedAccuracyMs: Float` — `Location.speedAccuracyMetersPerSecond` (API 26+)
- `fixTimestamp: Long` — `Location.elapsedRealtimeNanos`

**GNSS quality gate logic (used by CalibrationEngine):**
```kotlin
fun isGoodForCalibration(): Boolean =
    hdop < 2.0f &&
    satCount >= 6 &&
    speedMs > 15f / 3.6f &&        // > 15 km/h
    speedAccuracyMs < 1.0f &&      // reported accuracy < 1 m/s
    !isHardBraking                 // |ax_longitudinal| < 2 m/s²
```

Request location updates at 1 Hz (GNSS fix rate). For speed estimation between fixes, interpolate using IMU.

---

## 2. IMU Manager (`ImuManager.kt`)

Register three sensors at `SENSOR_DELAY_FASTEST` (~200 Hz target):
- `TYPE_ACCELEROMETER` → `ax, ay, az` in m/s²
- `TYPE_GYROSCOPE` → `gx, gy, gz` in rad/s
- `TYPE_ROTATION_VECTOR` → quaternion → convert to yaw (azimuth) in radians

**Rotation vector → heading:**
```kotlin
val rotMatrix = FloatArray(9)
SensorManager.getRotationMatrixFromVector(rotMatrix, rotationVector)
val orientation = FloatArray(3)
SensorManager.getOrientation(rotMatrix, orientation)
val azimuthRad = orientation[0]  // yaw, radians, -π to π
```

Store raw samples in a thread-safe ring buffer (2048 samples × 6 channels) for FFT consumption.

Maintain running bias estimates (from `BiasEstimator`). Apply before feeding EKF:
```
ax_corrected = ax_raw - bias_ax
```

---

## 3. EKF Core (`EkfCore.kt`)

**State vector:** `x = [px, py, vx, vy, ψ, ψ̇]ᵀ` (position x/y in metres, velocity x/y, heading, yaw rate)

CTRV (Constant Turn Rate and Velocity) process model.

**Predict step:**
```kotlin
// If yaw rate ψ̇ ≈ 0, use straight-line model to avoid division by zero
fun predict(dt: Float, ax: Float, az: Float) {
    // propagate state through CTRV equations
    // apply process noise Q
}
```

**Update step:**
```kotlin
// Generic measurement update: z, H (Jacobian), R
fun update(z: FloatMatrix, H: FloatMatrix, R: FloatMatrix)
```

Jacobians computed analytically (no numerical differentiation).

---

## 4. EKF #2 — IMU + GNSS with Adaptive Q/R (`EkfImuGnss.kt`)

Extends `EkfCore`. Measurement vector when GNSS available:
```
z = [speed_gnss, heading_gnss]ᵀ
```

Uses Android `TYPE_ROTATION_VECTOR` as heading measurement (always available, low noise).

Speed measurement uses `speedMs` from GNSS fix.

**Sage-Husa adaptive estimator (`SageHusa.kt`):**

Updates Q and R online from innovation sequence when GNSS is good:

```kotlin
// Innovation
val nu = z - H * x_pred
val S  = H * P_pred * H.T + R

// Sage-Husa Q update
val d = (1 - b) / (1 - b.pow(k))           // b = 0.97, k = step count
Q = (1 - d) * Q + d * (K * nu.outer(nu) * K.T)

// R update from innovation covariance
R = (1 - d) * R + d * (nu.outer(nu) - H * P_pred * H.T)
R = R.clampDiagonal(min = 0.01f)            // prevent negative variance
```

Display current diagonal values of Q and R in the UI card.

---

## 5. FFT Extractor (`FftExtractor.kt`)

Processes IMU ring buffer continuously.

**Parameters:**
- Window size: 512 samples (~2.56 s at 200 Hz)
- Overlap: 50% (new window every 256 samples ≈ 1.28 s)
- Window function: Hann
- Output: magnitude spectrum 0–100 Hz (257 bins)

**Feature vector (per window):**
```
[ 257 FFT magnitude bins (ax),
  257 FFT magnitude bins (ay),
  257 FFT magnitude bins (az),
  spectral_centroid_x,
  spectral_centroid_y,
  spectral_centroid_z,
  rms_ax, rms_ay, rms_az,
  kurtosis_ax, kurtosis_ay, kurtosis_az,
  dominant_freq_ax, dominant_freq_ay, dominant_freq_az ]
  
Total: ~786 floats
```

Use `org.jtransforms:jtransforms:3.1` for FFT (add to `build.gradle`).

---

## 6. CNN Speed Estimator (`SpeedCnn.kt`)

**Model architecture (define in Python/Keras, convert to TFLite, embed as asset):**

```python
Input(shape=(786,))
Dense(256, activation='relu')
BatchNormalization()
Dense(128, activation='relu')
Dropout(0.2)
Dense(64, activation='relu')
Dense(1, activation='linear')   # output: speed in m/s
```

**Initial pretrained weights:** initialize with random weights — the app learns from scratch via online calibration. Embed a `speed_cnn.tflite` asset even if weights are near-random at first install; calibration will train it within the first ~20 minutes of driving.

**TFLite inference:**
```kotlin
val interpreter = Interpreter(loadModelFile("speed_cnn.tflite"))
val input  = Array(1) { fftFeatures }       // FloatArray(786)
val output = Array(1) { FloatArray(1) }
interpreter.run(input, output)
val speedMs = output[0][0]
```

**Model persistence:** save/load TFLite FlatBuffer weights to `filesDir/speed_cnn_trained.tflite` after each calibration batch.

---

## 7. Online Trainer (`OnlineTrainer.kt`)

Runs on a background coroutine (`Dispatchers.Default`).

**Replay buffer (`ReplayBuffer.kt`):**
- Capacity: 500 samples
- Structure: circular buffer of `Pair<FloatArray, Float>` (features, gnss_speed_ms)
- Diversity sampling: reject sample if nearest-neighbor in feature space is within distance ε (avoid motorway cruise redundancy)

**Training trigger:**
```kotlin
// Every 30 new calibration samples collected:
if (newSamplesCount >= 30 && gnssGoodDuration > 10.seconds) {
    trainOnBatch(replayBuffer.sample(64))
    newSamplesCount = 0
    saveModelWeights()
}
```

**Training loop (mini-batch gradient descent on TFLite):**

Since TFLite does not natively support training on Android, implement one of:

**Option A (preferred):** Use **TFLite Model Personalization / on-device training API** (TensorFlow Lite Task Library or custom training op). Freeze first 2 Dense layers, train only last Dense(64) → Dense(1).

**Option B (fallback):** Maintain a small `NeuralNet` implemented in pure Kotlin (no TFLite training needed) mirroring the last 2 layers only. The frozen backbone (first 3 layers) runs in TFLite; the trainable head is a simple matrix multiply updated by SGD in Kotlin:

```kotlin
// Head: 64 → 1, SGD
val W = FloatArray(64)   // weights
var b = 0f               // bias
val lr = 1e-4f

fun trainStep(features64: FloatArray, target: Float) {
    val pred = dot(W, features64) + b
    val err  = pred - target
    for (i in W.indices) W[i] -= lr * err * features64[i]
    b -= lr * err
    loss = err * err   // track for UI
}
```

Display running MSE loss on last 32 samples in the CNN card.

---

## 8. Calibration Engine (`CalibrationEngine.kt`)

Master controller. Runs on a `HandlerThread`.

```kotlin
fun onImuSample(imu: ImuSample) {
    biasEstimator.update(imu)
    ekfImuGnss.predict(dt, imu.ax_corrected, imu.gz_corrected)
    fftExtractor.addSample(imu)
}

fun onGnssFix(fix: GnssFix) {
    val good = gnssQualityGate.evaluate(fix)

    // Always update EKF #2 with GNSS measurement
    ekfImuGnss.update(fix)

    if (good) {
        // Sage-Husa: adapt Q and R
        sageHusa.update(ekfImuGnss.innovation, ekfImuGnss.P)

        // CNN: collect training sample if FFT window is fresh
        fftExtractor.latestFeatures()?.let { features ->
            replayBuffer.add(features, fix.speedMs)
            onlineTrainer.notifyNewSample()
        }

        calibrationState = CALIBRATING
    } else {
        calibrationState = COASTING
    }
}

fun onFftWindowReady(features: FloatArray) {
    val cnnSpeed = speedCnn.infer(features)
    // Feed CNN speed as measurement into EKF (with learned R_cnn)
    ekfImuGnss.updateWithCnn(cnnSpeed, R_cnn = onlineTrainer.currentLoss + 0.5f)
}
```

---

## 9. Speed ViewModel (`SpeedViewModel.kt`)

```kotlin
data class SpeedState(
    val gnssSpeedKmh:   Float,
    val ekfSpeedKmh:    Float,
    val cnnEkfSpeedKmh: Float,
    val hdop:           Float,
    val satCount:       Int,
    val ekfQ:           Float,   // diagonal mean of Q
    val ekfR:           Float,   // diagonal mean of R
    val cnnLoss:        Float,
    val calibrating:    Boolean,
    val logLine:        String
)

val state: StateFlow<SpeedState>
```

Update at 10 Hz from a `CoroutineScope` coroutine using `conflatedChannel` so fast sensor data never blocks UI.

---

## 10. UI (`MainActivity.kt` + layout)

Use **Jetpack Compose** for the UI.

Three `SpeedCard` composables:

```kotlin
@Composable
fun SpeedCard(
    icon: String,
    title: String,
    speedKmh: Float,
    subtitle: String,
    color: Color
)
```

Color coding:
- GPS card: `Color(0xFF00E5FF)` (cyan)
- EKF card: `Color(0xFF69FF47)` (green)
- CNN+EKF card: `Color(0xFFFF6D00)` (amber)

Speed value font: large monospace (`FontFamily.Monospace`, 48sp), bold.

Bottom status bar: flashes green when `calibrating == true`.

Log panel: `LazyColumn` of last 5 log strings, auto-scrolls, monospace 10sp, dark background.

---

## 11. Permissions (`AndroidManifest.xml`)

```xml
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
<uses-feature android:name="android.hardware.sensor.accelerometer" android:required="true" />
<uses-feature android:name="android.hardware.sensor.gyroscope" android:required="true" />
```

Request `ACCESS_FINE_LOCATION` at runtime in `MainActivity` using `ActivityResultContracts.RequestPermission`.

---

## 12. Dependencies (`build.gradle.kts`)

```kotlin
dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("androidx.activity:activity-compose:1.8.2")

    // Compose BOM
    implementation(platform("androidx.compose:compose-bom:2024.02.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")

    // TFLite
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")

    // FFT
    implementation("com.github.wendykierp:JTransforms:3.1")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

---

## 13. Initial TFLite Model Asset

Place a pre-initialized (random weights) `speed_cnn.tflite` in `app/src/main/assets/`.

Generate it with this Python snippet (run once, commit the output):

```python
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(786,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('speed_cnn.tflite', 'wb') as f:
    f.write(tflite_model)
print("Saved speed_cnn.tflite —", len(tflite_model), "bytes")
```

---

## 14. Known Constraints & Notes for Claude Code

1. **TFLite on-device training** is experimental. If `org.tensorflow:tensorflow-lite-select-tf-ops` causes build issues, fall back to the Kotlin head-only SGD approach described in Section 7 Option B.

2. **JTransforms** may require adding JitPack to `settings.gradle.kts`:
   ```kotlin
   maven { url = uri("https://jitpack.io") }
   ```

3. **Matrix math**: implement a minimal `FloatMatrix` class (6×6 max) with `+`, `*`, `.T`, `.inv()` using Cramer's rule or LU decomposition — avoid pulling in a full linear algebra library.

4. **Threading model**:
   - IMU callbacks → `HandlerThread("ImuThread")`
   - GNSS callbacks → main thread → dispatch to `ImuThread`
   - TFLite inference → `Dispatchers.Default`
   - UI updates → `Dispatchers.Main` at 10 Hz

5. **EKF numerical stability**: add a small `ε = 1e-6` to diagonal of P after each predict step to prevent covariance collapse.

6. **First-run behavior**: CNN speed will be garbage until ~500 calibration samples are collected. Show `"TRAINING..."` instead of a speed value when `replayBuffer.size < 100`.

7. **Location mock detection**: check `Location.isFromMockProvider` and gate calibration if true.

---

## Deliverable

A fully buildable Android Studio project with the above structure. The app should compile and run on any Android 8.0+ device with GPS, accelerometer, and gyroscope. All three speed fields must update live. Calibration must run silently in the background with no user interaction required.
