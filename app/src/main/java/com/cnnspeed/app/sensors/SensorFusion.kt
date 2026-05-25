package com.cnnspeed.app.sensors

import android.content.Context

/**
 * Thin facade that owns the IMU + GNSS managers and forwards their data to a
 * single set of listener callbacks. Avoids the calling site needing to wire
 * up two managers separately.
 */
class SensorFusion(context: Context) {
    val imu = ImuManager(context)
    val gnss = GnssManager(context)

    fun start() { imu.start(); gnss.start() }
    fun stop()  { imu.stop();  gnss.stop()  }
}
