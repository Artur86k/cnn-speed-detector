package com.cnnspeed.app.sensors

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Looper
import androidx.core.content.ContextCompat

/** Snapshot of a single GNSS fix plus quality metrics. */
data class GnssFix(
    val speedMs: Float,
    val bearingRad: Float,
    val hdop: Float,
    val satCount: Int,
    val speedAccuracyMs: Float,
    val fixTimestampNanos: Long,
    val isFromMock: Boolean
)

class GnssManager(private val context: Context) {

    private val locationManager =
        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    private var satCount: Int = 0
    private var lastHdop: Float = 99f   // unknown until proven good

    var onFix: ((GnssFix) -> Unit)? = null
        @Synchronized set

    private val statusCallback = object : GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(status: GnssStatus) {
            var used = 0
            for (i in 0 until status.satelliteCount) {
                if (status.usedInFix(i)) used++
            }
            satCount = used
        }
    }

    private val locationListener = LocationListener { location -> emitFix(location) }

    @SuppressLint("MissingPermission")
    fun start() {
        if (!hasPermission()) return
        try {
            locationManager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                1000L,    // 1 Hz
                0f,
                locationListener,
                Looper.getMainLooper()
            )
            locationManager.registerGnssStatusCallback(
                statusCallback, android.os.Handler(Looper.getMainLooper())
            )
        } catch (_: SecurityException) { /* ignore */ }
    }

    fun stop() {
        try {
            locationManager.removeUpdates(locationListener)
            locationManager.unregisterGnssStatusCallback(statusCallback)
        } catch (_: SecurityException) {}
    }

    private fun emitFix(loc: Location) {
        // HDOP is not directly exposed via Location; approximate from speed accuracy.
        // A real implementation would parse NMEA $GPGSA from addNmeaListener.
        val acc = if (Build.VERSION.SDK_INT >= 26 && loc.hasSpeedAccuracy()) {
            loc.speedAccuracyMetersPerSecond
        } else 5f
        lastHdop = (acc * 2f).coerceIn(0.5f, 50f)

        val fix = GnssFix(
            speedMs = if (loc.hasSpeed()) loc.speed else 0f,
            bearingRad = if (loc.hasBearing()) Math.toRadians(loc.bearing.toDouble()).toFloat() else 0f,
            hdop = lastHdop,
            satCount = satCount,
            speedAccuracyMs = acc,
            fixTimestampNanos = loc.elapsedRealtimeNanos,
            isFromMock = if (Build.VERSION.SDK_INT >= 31) loc.isMock else loc.isFromMockProvider
        )
        onFix?.invoke(fix)
    }

    private fun hasPermission(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
}
