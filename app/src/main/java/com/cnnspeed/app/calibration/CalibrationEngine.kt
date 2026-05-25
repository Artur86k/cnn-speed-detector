package com.cnnspeed.app.calibration

import android.content.Context
import android.os.HandlerThread
import com.cnnspeed.app.cnn.FftExtractor
import com.cnnspeed.app.cnn.OnlineTrainer
import com.cnnspeed.app.cnn.ReplayBuffer
import com.cnnspeed.app.cnn.SpeedCnn
import com.cnnspeed.app.ekf.EkfImuGnss
import com.cnnspeed.app.sensors.GnssFix
import com.cnnspeed.app.sensors.ImuSample
import com.cnnspeed.app.sensors.SensorFusion
import kotlin.math.cos
import kotlin.math.sin

/** Aggregated runtime state suitable for ViewModel consumption. */
data class CalibState(
    val gnssSpeedMs: Float = 0f,
    val ekfSpeedMs: Float = 0f,           // EKF fused with IMU + GNSS only
    val cnnEkfSpeedMs: Float = 0f,        // EKF fused with IMU + GNSS + CNN
    val hdop: Float = 99f,
    val satCount: Int = 0,
    val ekfQMean: Float = 0f,
    val ekfRMean: Float = 0f,
    val cnnLoss: Float = 1f,
    val calibrating: Boolean = false,
    val cnnReady: Boolean = false,
    val log: String = ""
)

/**
 * Master controller. Owns two EKFs running in parallel:
 *   - [ekfNoCnn]: fused only with IMU and GNSS  → "EKF Speed" card
 *   - [ekfWithCnn]: also fused with CNN updates → "EKF + CNN Speed" card
 *
 * Both EKFs receive identical predict steps and GNSS measurements so the only
 * difference between them is whether the CNN's noisy speed estimate is folded
 * in. This lets the UI compare the value the CNN adds.
 */
class CalibrationEngine(context: Context) {

    val fusion = SensorFusion(context)
    val ekfNoCnn = EkfImuGnss()
    val ekfWithCnn = EkfImuGnss()
    val biasEstimator = BiasEstimator()
    val qualityGate = GnssQualityGate()
    val replayBuffer = ReplayBuffer()
    val fftExtractor = FftExtractor()
    val speedCnn = SpeedCnn(context)
    val onlineTrainer = OnlineTrainer(context, speedCnn, replayBuffer)

    private val workerThread = HandlerThread("CalibThread").apply { start() }
    private val handler = android.os.Handler(workerThread.looper)

    private var lastImuTNanos: Long = 0L
    private var lastGnssSpeedMs: Float = 0f
    private var gnssGoodSinceMs: Long = 0L

    @Volatile var state: CalibState = CalibState()
        private set

    /**
     * Debug toggle: when false, incoming GNSS fixes are ignored by the EKF
     * update path and no new replay-buffer samples are collected. Raw GNSS
     * values still flow to the UI so they can be compared against the EKF.
     */
    @Volatile var gnssEnabled: Boolean = true

    fun start() {
        fusion.imu.onSample = { sample -> handler.post { onImuSample(sample) } }
        fusion.gnss.onFix = { fix -> handler.post { onGnssFix(fix) } }
        fftExtractor.onFeatures = { features -> handler.post { onFftWindow(features) } }
        fusion.start()
    }

    fun stop() {
        fusion.stop()
        speedCnn.close()
        workerThread.quitSafely()
    }

    private fun onImuSample(s: ImuSample) {
        val dt = if (lastImuTNanos == 0L) 0f else (s.tNanos - lastImuTNanos) / 1e9f
        lastImuTNanos = s.tNanos

        // Forward acceleration = world-frame horizontal accel projected onto
        // the EKF's current heading. Using world-frame quantities means
        // gravity has already been cancelled out (it stays in world-Z).
        val psi = ekfNoCnn.headingRad()
        val axForward = s.linAxWorld * cos(psi) + s.linAyWorld * sin(psi)
        biasEstimator.update(axForward, s.yawRateWorld, lastGnssSpeedMs)
        val axCorr = biasEstimator.correctAx(axForward)
        val gzCorr = biasEstimator.correctGz(s.yawRateWorld)
        qualityGate.lastAxLongitudinal = axCorr

        if (dt in 0f..0.1f) {
            ekfNoCnn.predict(dt, axCorr, gzCorr)
            ekfWithCnn.predict(dt, axCorr, gzCorr)
        }
        fftExtractor.addSample(s)
        publishState()
    }

    private fun onGnssFix(fix: GnssFix) {
        lastGnssSpeedMs = fix.speedMs

        if (!gnssEnabled) {
            gnssGoodSinceMs = 0L
            qualityGate.lastFix = fix
            qualityGate.lastGood = false
            publishState()
            return
        }

        val good = qualityGate.isGood(fix)
        ekfNoCnn.updateGnss(fix.speedMs, fix.bearingRad, adapt = good)
        ekfWithCnn.updateGnss(fix.speedMs, fix.bearingRad, adapt = good)

        if (good) {
            if (gnssGoodSinceMs == 0L) gnssGoodSinceMs = System.currentTimeMillis()
            fftExtractor.latestFeatures?.let { features ->
                val kept = replayBuffer.add(features, fix.speedMs)
                if (kept) onlineTrainer.notifyNewSample()
            }
        } else {
            gnssGoodSinceMs = 0L
        }
        publishState()
    }

    private fun onFftWindow(features: FloatArray) {
        val cnnSpeedMs = onlineTrainer.infer(features) ?: return
        if (cnnSpeedMs.isFinite() && cnnSpeedMs in -2f..120f) {
            ekfWithCnn.updateWithCnn(cnnSpeedMs.coerceAtLeast(0f), onlineTrainer.currentLoss + 0.5f)
        }
        publishState()
    }

    private fun publishState() {
        val cnnReady = replayBuffer.size() >= 100
        state = CalibState(
            gnssSpeedMs = lastGnssSpeedMs,
            ekfSpeedMs = ekfNoCnn.speedMs(),
            cnnEkfSpeedMs = ekfWithCnn.speedMs(),
            hdop = qualityGate.lastFix?.hdop ?: 99f,
            satCount = qualityGate.lastFix?.satCount ?: 0,
            ekfQMean = ekfNoCnn.Q.diagMean(),
            ekfRMean = ekfNoCnn.R.diagMean(),
            cnnLoss = onlineTrainer.currentLoss,
            calibrating = qualityGate.lastGood,
            cnnReady = cnnReady,
            log = "buf=${replayBuffer.size()} sats=${qualityGate.lastFix?.satCount ?: 0}"
        )
    }
}
