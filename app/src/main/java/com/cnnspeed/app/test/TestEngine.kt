package com.cnnspeed.app.test

import android.content.Context
import com.cnnspeed.app.sensors.GnssManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Coordinator for the CNN test harness:
 *   GnssManager  -> latest GPS speed + heading
 *   WindowSampler -> 2 s ring of vehicle-frame IMU samples at 200 Hz
 *   SpeedCnnFull -> runs every INFER_PERIOD_MS, produces predicted speed
 *
 * Outputs a [TestEngineState] StateFlow for the UI and accumulates samples
 * into [session] for the Stats screen.
 */
class TestEngine(context: Context) {

    private val gnss     = GnssManager(context)
    private val sampler  = WindowSampler(context)
    private val cnn      = SpeedCnnFull(context)
    val session          = TestSession()

    private val scope    = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var loopJob: Job? = null
    private val winBuf   = FloatArray(cnn.tSteps * cnn.nChan)

    // GPS state — updated by GnssManager callback, read by inference loop.
    @Volatile private var gpsSpeedMs:   Float = 0f
    @Volatile private var gpsBearingRad:Float = 0f
    @Volatile private var hAccProxyM:   Float = 99f
    @Volatile private var satCount:     Int   = 0
    @Volatile private var lastFixTs:    Long  = 0L
    @Volatile private var hasFreshFix:  Boolean = false

    private val _state = MutableStateFlow(TestEngineState())
    val state: StateFlow<TestEngineState> = _state.asStateFlow()

    @Volatile private var recording: Boolean = false

    fun start() {
        gnss.onFix = { fix ->
            gpsSpeedMs    = fix.speedMs
            gpsBearingRad = fix.bearingRad
            hAccProxyM    = fix.hdop * 5f   // rough — GnssManager approximates hdop from speed_acc
            satCount      = fix.satCount
            lastFixTs     = fix.fixTimestampNanos
            hasFreshFix   = true
            // Update sampler heading from GPS when moving, else fall back to game-azimuth.
            sampler.headingRad =
                if (fix.speedMs > 1f) fix.bearingRad
                else Math.toRadians(sampler.azimuthDeg().toDouble()).toFloat()
        }
        gnss.start()
        sampler.start()

        loopJob = scope.launch {
            while (isActive) {
                runOnceSafely()
                delay(INFER_PERIOD_MS)
            }
        }
    }

    fun stop() {
        loopJob?.cancel()
        gnss.stop()
        sampler.stop()
        cnn.close()
        scope.cancel()
    }

    private fun runOnceSafely() {
        try {
            // Heading fallback: when GPS is cold or the car is stopped, use the
            // game-rotation azimuth (mag-free yaw). Matches what preprocess.py
            // does offline before the first moving GPS fix.
            val now = android.os.SystemClock.elapsedRealtimeNanos()
            val freshGps = hasFreshFix && (now - lastFixTs) < 3_000_000_000L
            if (!freshGps || gpsSpeedMs <= 1f) {
                sampler.headingRad =
                    Math.toRadians(sampler.azimuthDeg().toDouble()).toFloat()
            }

            val ready = sampler.snapshot(winBuf)
            if (!ready) {
                _state.value = _state.value.copy(
                    samplesReady = false,
                    samplesInRing = sampler.nSamplesIngested,
                )
                return
            }
            val pred = cnn.predict(winBuf).coerceAtLeast(0f)

            _state.value = TestEngineState(
                samplesReady   = true,
                samplesInRing  = sampler.nSamplesIngested,
                gpsSpeedMs     = gpsSpeedMs,
                cnnSpeedMs     = pred,
                hasFreshGps    = freshGps,
                satCount       = satCount,
                recording      = recording,
                nRecorded      = session.snapshot().size,
                headingDeg     = sampler.azimuthDeg(),
            )

            if (recording) {
                session.add(TestSample(
                    tNanos = now,
                    gpsMs  = if (freshGps) gpsSpeedMs else Float.NaN,
                    cnnMs  = pred,
                    hasGps = freshGps,
                ))
            }
        } catch (t: Throwable) {
            _state.value = _state.value.copy(lastError = t.message ?: t::class.java.simpleName)
        }
    }

    fun setRecording(on: Boolean) {
        recording = on
        _state.value = _state.value.copy(recording = on)
    }

    fun clearSession() {
        session.clear()
        _state.value = _state.value.copy(nRecorded = 0)
    }

    companion object {
        const val INFER_PERIOD_MS: Long = 250L   // 4 Hz inference cadence
    }
}

/** UI-facing state, refreshed by the inference loop. */
data class TestEngineState(
    val samplesReady:  Boolean = false,
    val samplesInRing: Long    = 0L,
    val gpsSpeedMs:    Float   = 0f,
    val cnnSpeedMs:    Float   = 0f,
    val hasFreshGps:   Boolean = false,
    val satCount:      Int     = 0,
    val recording:     Boolean = false,
    val nRecorded:     Int     = 0,
    val headingDeg:    Float   = 0f,
    val lastError:     String? = null,
)
