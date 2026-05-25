package com.cnnspeed.app.calibration

import com.cnnspeed.app.sensors.GnssFix
import kotlin.math.abs

/** Decides whether a GNSS fix is trustworthy enough to drive online calibration. */
class GnssQualityGate(
    val maxHdop: Float = 2.0f,
    val minSats: Int = 6,
    val minSpeedMs: Float = 15f / 3.6f,
    val maxSpeedAccMs: Float = 1.0f,
    val maxBrakingMs2: Float = 2.0f
) {
    @Volatile var lastAxLongitudinal: Float = 0f
    @Volatile var lastFix: GnssFix? = null
    @Volatile var lastGood: Boolean = false

    fun isGood(fix: GnssFix): Boolean {
        lastFix = fix
        val good = !fix.isFromMock &&
            fix.hdop < maxHdop &&
            fix.satCount >= minSats &&
            fix.speedMs > minSpeedMs &&
            fix.speedAccuracyMs < maxSpeedAccMs &&
            abs(lastAxLongitudinal) < maxBrakingMs2
        lastGood = good
        return good
    }
}
