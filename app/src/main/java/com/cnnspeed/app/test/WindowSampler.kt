package com.cnnspeed.app.test

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.HandlerThread
import kotlin.math.sin
import kotlin.math.cos

/**
 * Self-contained sensor listener that produces vehicle-frame IMU windows
 * matching what scripts/preprocess.py emits for training.
 *
 *   Channels: [accel_fwd, accel_lat, accel_vrt, gyro_x, gyro_y, gyro_z]
 *
 * Pipeline per sample:
 *   1. Use TYPE_LINEAR_ACCELERATION (gravity already removed by the OS).
 *   2. Use TYPE_GAME_ROTATION_VECTOR (magnetometer-free, matches gamerot.csv
 *      used during training).
 *   3. Rotate linear accel into ENU world frame via game-rotation matrix.
 *   4. Rotate ENU -> vehicle frame using current heading. Heading is provided
 *      by [setHeadingDeg]; the TestEngine sets it from GPS bearing when moving
 *      and falls back to the game-rotation azimuth when stopped.
 *   5. Resample everything onto a fixed 200 Hz grid by stepping forward whole
 *      DT (5 ms) intervals; each grid slot uses the most recent samples.
 */
class WindowSampler(
    private val context: Context,
    private val tSteps: Int = 400,
    private val nChan: Int = 6,
    private val hz: Float = 200f,
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val thread = HandlerThread("WindowSamplerThread").apply { start() }
    private val handler = android.os.Handler(thread.looper)

    private val dtNanos: Long = (1_000_000_000L / hz).toLong()
    // Ring buffer of T*C floats. Layout: t-major then channel.
    private val ring = FloatArray(tSteps * nChan)
    private var ringHead = 0       // write index into ring (in samples)
    private var ringSize = 0       // 0..tSteps
    private var lastGridT: Long = 0
    private val lock = Any()

    private val rotMatrix = FloatArray(9).also { it[0]=1f; it[4]=1f; it[8]=1f }
    @Volatile private var lastLinAx = 0f
    @Volatile private var lastLinAy = 0f
    @Volatile private var lastLinAz = 0f
    @Volatile private var lastGx = 0f
    @Volatile private var lastGy = 0f
    @Volatile private var lastGz = 0f
    @Volatile private var azimuthRadGame: Float = 0f   // mag-free yaw

    @Volatile var headingRad: Float = 0f               // vehicle heading (ENU course)
    var nSamplesIngested: Long = 0L; private set

    fun start() {
        register(Sensor.TYPE_LINEAR_ACCELERATION)
        register(Sensor.TYPE_GYROSCOPE)
        register(Sensor.TYPE_GAME_ROTATION_VECTOR)
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        thread.quitSafely()
    }

    fun azimuthDeg(): Float = Math.toDegrees(azimuthRadGame.toDouble()).toFloat()

    /** Copy the current window into [out] (length T*C). Returns true if full. */
    fun snapshot(out: FloatArray): Boolean {
        synchronized(lock) {
            if (ringSize < tSteps) return false
            require(out.size == ring.size) { "out size mismatch" }
            // Tail-first read into out, starting at oldest sample.
            val start = (ringHead) % tSteps         // oldest position
            val tailCount = tSteps - start
            System.arraycopy(ring, start * nChan, out, 0, tailCount * nChan)
            System.arraycopy(ring, 0, out, tailCount * nChan, start * nChan)
            return true
        }
    }

    private fun register(type: Int) {
        val s = sensorManager.getDefaultSensor(type) ?: return
        try {
            sensorManager.registerListener(this, s, SensorManager.SENSOR_DELAY_FASTEST, handler)
        } catch (_: SecurityException) {
            sensorManager.registerListener(this, s, SensorManager.SENSOR_DELAY_GAME, handler)
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                lastLinAx = event.values[0]
                lastLinAy = event.values[1]
                lastLinAz = event.values[2]
                pushGridIfDue(event.timestamp)
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGx = event.values[0]; lastGy = event.values[1]; lastGz = event.values[2]
            }
            Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                SensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
                val ori = FloatArray(3)
                SensorManager.getOrientation(rotMatrix, ori)
                azimuthRadGame = ori[0]
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    /**
     * Push one resampled sample onto the 200 Hz grid whenever the time gap
     * since the last grid point reaches [dtNanos]. If multiple grid steps have
     * elapsed (sensor slowed down briefly), fill them with the most recent
     * values — a held-sample approximation that's fine for spectrograms but
     * also matches what linear interp at the boundary would do.
     */
    private fun pushGridIfDue(tNow: Long) {
        if (lastGridT == 0L) { lastGridT = tNow; return }
        while (tNow - lastGridT >= dtNanos) {
            lastGridT += dtNanos
            // Rotate linear accel into ENU using game-rotation matrix.
            val ax = lastLinAx; val ay = lastLinAy; val az = lastLinAz
            val aE = rotMatrix[0]*ax + rotMatrix[1]*ay + rotMatrix[2]*az
            val aN = rotMatrix[3]*ax + rotMatrix[4]*ay + rotMatrix[5]*az
            val aU = rotMatrix[6]*ax + rotMatrix[7]*ay + rotMatrix[8]*az
            // ENU -> vehicle (forward / lateral / vertical) using current heading.
            val h = headingRad
            val s = sin(h); val c = cos(h)
            val aFwd =  aE * s + aN * c
            val aLat =  aE * c + aN * (-s)
            val aVrt =  aU

            synchronized(lock) {
                val base = ringHead * nChan
                ring[base + 0] = aFwd
                ring[base + 1] = aLat
                ring[base + 2] = aVrt
                ring[base + 3] = lastGx
                ring[base + 4] = lastGy
                ring[base + 5] = lastGz
                ringHead = (ringHead + 1) % tSteps
                if (ringSize < tSteps) ringSize++
            }
            nSamplesIngested++
        }
    }
}
