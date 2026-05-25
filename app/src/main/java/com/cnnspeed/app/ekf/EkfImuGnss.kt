package com.cnnspeed.app.ekf

import kotlin.math.cos
import kotlin.math.sin

/**
 * EKF #2 — IMU + GNSS with adaptive Q/R via Sage-Husa.
 *
 * Measurement model when GNSS fix is available:
 *   z = [vx_gnss, vy_gnss, ψ_gnss]ᵀ
 *
 * GNSS speed and bearing are converted to a 2D velocity vector to keep H linear.
 */
class EkfImuGnss : EkfCore() {

    // Measurement covariance
    val R: FloatMatrix = FloatMatrix.diag(0.5f, 0.5f, 0.05f)

    private val sageHusa = SageHusa()

    /** Most recent innovation, exposed for diagnostics. */
    var lastInnovation: FloatMatrix = FloatMatrix(3, 1)
        private set

    /** Apply a GNSS fix as a measurement. If [adapt] is true, run Sage-Husa. */
    fun updateGnss(speedMs: Float, bearingRad: Float, adapt: Boolean) {
        val vx = speedMs * cos(bearingRad)
        val vy = speedMs * sin(bearingRad)
        val z = FloatMatrix.column(vx, vy, bearingRad)

        val H = FloatMatrix(3, 6)
        H[0, 2] = 1f   // vx
        H[1, 3] = 1f   // vy
        H[2, 4] = 1f   // heading

        val Pbefore = P.copy()
        val nu = update(z, H, R)
        lastInnovation = nu

        if (adapt) {
            val Ht = H.transpose()
            val S = (H * Pbefore * Ht) + R
            val K = Pbefore * Ht * (S.inverse() ?: return)
            sageHusa.step(Q, R, nu, K, Pbefore, H)
        }
    }

    /** Apply a CNN-derived speed as a scalar speed measurement. */
    fun updateWithCnn(cnnSpeedMs: Float, rCnn: Float) {
        // Use current heading estimate to project scalar speed onto vx,vy.
        val psi = x[4, 0]
        val vx = cnnSpeedMs * cos(psi)
        val vy = cnnSpeedMs * sin(psi)
        val z = FloatMatrix.column(vx, vy)
        val H = FloatMatrix(2, 6)
        H[0, 2] = 1f
        H[1, 3] = 1f
        val Rcnn = FloatMatrix.diag(rCnn, rCnn)
        update(z, H, Rcnn)
    }
}
