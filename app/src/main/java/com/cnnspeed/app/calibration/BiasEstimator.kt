package com.cnnspeed.app.calibration

import com.cnnspeed.app.sensors.ImuSample
import kotlin.math.abs

/**
 * Tracks accelerometer and gyroscope bias using a slow exponential moving
 * average when the vehicle is essentially stationary (very low yaw rate AND
 * very low acceleration magnitude variation). The longitudinal bias on ax is
 * the most important — without correction it leaks straight into EKF velocity.
 */
class BiasEstimator(private val alpha: Float = 0.001f) {

    @Volatile var biasAx: Float = 0f; private set
    @Volatile var biasAy: Float = 0f; private set
    @Volatile var biasAz: Float = 0f; private set
    @Volatile var biasGx: Float = 0f; private set
    @Volatile var biasGy: Float = 0f; private set
    @Volatile var biasGz: Float = 0f; private set

    fun update(s: ImuSample, gnssSpeedMs: Float) {
        val stationary = gnssSpeedMs < 0.3f &&
            abs(s.gx) < 0.05f && abs(s.gy) < 0.05f && abs(s.gz) < 0.05f
        if (!stationary) return
        biasAx = (1f - alpha) * biasAx + alpha * s.ax
        biasAy = (1f - alpha) * biasAy + alpha * s.ay
        // For az we expect ~9.81 (gravity), so subtract it.
        biasAz = (1f - alpha) * biasAz + alpha * (s.az - 9.81f)
        biasGx = (1f - alpha) * biasGx + alpha * s.gx
        biasGy = (1f - alpha) * biasGy + alpha * s.gy
        biasGz = (1f - alpha) * biasGz + alpha * s.gz
    }

    fun correctAx(raw: Float): Float = raw - biasAx
    fun correctGz(raw: Float): Float = raw - biasGz
}
