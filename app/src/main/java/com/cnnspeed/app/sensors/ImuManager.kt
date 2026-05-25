package com.cnnspeed.app.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.HandlerThread

/** Single IMU sample (raw, with rotation-derived heading). */
data class ImuSample(
    val tNanos: Long,
    val ax: Float, val ay: Float, val az: Float,
    val gx: Float, val gy: Float, val gz: Float,
    val azimuthRad: Float
)

/** Thread-safe circular ring buffer of IMU samples. */
class ImuRingBuffer(val capacity: Int) {
    private val buf = arrayOfNulls<ImuSample>(capacity)
    private var head = 0
    private var sz = 0
    private val lock = Any()

    fun add(s: ImuSample) {
        synchronized(lock) {
            buf[head] = s
            head = (head + 1) % capacity
            if (sz < capacity) sz++
        }
    }

    /** Snapshot the most recent [n] samples in chronological order. */
    fun snapshot(n: Int = capacity): List<ImuSample> {
        synchronized(lock) {
            val count = minOf(n, sz)
            if (count == 0) return emptyList()
            val out = ArrayList<ImuSample>(count)
            val start = (head - count + capacity) % capacity
            for (i in 0 until count) {
                val idx = (start + i) % capacity
                out.add(buf[idx] ?: continue)
            }
            return out
        }
    }

    fun size(): Int = synchronized(lock) { sz }
}

/**
 * Manages accelerometer, gyroscope, rotation vector. All callbacks land on a
 * dedicated HandlerThread "ImuThread" to keep heavy work off the main thread.
 */
class ImuManager(private val context: Context) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val handlerThread = HandlerThread("ImuThread").apply { start() }
    private val handler = android.os.Handler(handlerThread.looper)

    val ringBuffer = ImuRingBuffer(2048)

    private var lastAx = 0f; private var lastAy = 0f; private var lastAz = 0f
    private var lastGx = 0f; private var lastGy = 0f; private var lastGz = 0f
    private var lastAzimuth = 0f
    private var lastSampleT = 0L

    /** Listener invoked on every produced ImuSample (on ImuThread). */
    var onSample: ((ImuSample) -> Unit)? = null

    fun start() {
        registerSensor(Sensor.TYPE_ACCELEROMETER)
        registerSensor(Sensor.TYPE_GYROSCOPE)
        registerSensor(Sensor.TYPE_ROTATION_VECTOR)
    }

    /**
     * Register at SENSOR_DELAY_FASTEST (0 µs → ~200 Hz on most devices). Android 12+
     * requires HIGH_SAMPLING_RATE_SENSORS for rates > 200 Hz; if the permission is
     * missing the system throws SecurityException. Fall back to SENSOR_DELAY_GAME
     * (~50 Hz) so the app stays alive even without that permission.
     */
    private fun registerSensor(type: Int) {
        val sensor = sensorManager.getDefaultSensor(type) ?: return
        try {
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST, handler)
        } catch (_: SecurityException) {
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME, handler)
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        handlerThread.quitSafely()
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAx = event.values[0]; lastAy = event.values[1]; lastAz = event.values[2]
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGx = event.values[0]; lastGy = event.values[1]; lastGz = event.values[2]
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                val rot = FloatArray(9)
                SensorManager.getRotationMatrixFromVector(rot, event.values)
                val ori = FloatArray(3)
                SensorManager.getOrientation(rot, ori)
                lastAzimuth = ori[0]
            }
            else -> return
        }

        // Emit a sample on every accelerometer event (highest-rate sensor).
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            val sample = ImuSample(
                tNanos = event.timestamp,
                ax = lastAx, ay = lastAy, az = lastAz,
                gx = lastGx, gy = lastGy, gz = lastGz,
                azimuthRad = lastAzimuth
            )
            ringBuffer.add(sample)
            lastSampleT = event.timestamp
            onSample?.invoke(sample)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* not used */ }
}
