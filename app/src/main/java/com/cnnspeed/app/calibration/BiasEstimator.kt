package com.cnnspeed.app.calibration

import kotlin.math.abs

/**
 * Tracks bias on the world-frame forward acceleration and yaw rate using a
 * slow exponential moving average when the vehicle is stationary (GNSS speed
 * effectively zero). Even with gravity already removed by the device→world
 * rotation, the IMU has a small intrinsic offset that, integrated over time,
 * leaks into the EKF velocity. This estimator subtracts it out.
 */
class BiasEstimator(private val alpha: Float = 0.005f) {

    @Volatile var biasAxForward: Float = 0f; private set
    @Volatile var biasYawRate: Float = 0f; private set

    /**
     * Update the running bias estimates. Called once per IMU sample. Only the
     * GNSS speed determines whether we're stationary — using IMU readings to
     * decide would create a feedback loop with the bias itself.
     */
    fun update(axForward: Float, yawRate: Float, gnssSpeedMs: Float) {
        val stationary = gnssSpeedMs < 0.3f
        if (!stationary) return
        if (!axForward.isFinite() || !yawRate.isFinite()) return
        // Reject extreme outliers (a vehicle door slam etc.) so bias stays sane.
        if (abs(axForward) > 5f || abs(yawRate) > 0.5f) return
        biasAxForward = (1f - alpha) * biasAxForward + alpha * axForward
        biasYawRate = (1f - alpha) * biasYawRate + alpha * yawRate
    }

    fun correctAx(rawForward: Float): Float = rawForward - biasAxForward
    fun correctGz(rawYawRate: Float): Float = rawYawRate - biasYawRate
}
