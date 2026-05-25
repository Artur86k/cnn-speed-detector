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
        if (!speedMs.isFinite() || !bearingRad.isFinite()) return
        // Bearing is unreliable at very low speed (often reported as 0 or
        // garbage). Below 2 m/s, only fuse the velocity components and skip
        // the heading row entirely.
        val useHeading = speedMs > 2f
        val vx = speedMs * cos(bearingRad)
        val vy = speedMs * sin(bearingRad)

        val Pbefore = P.copy()
        val nu: FloatMatrix
        val H: FloatMatrix
        if (useHeading) {
            val z = FloatMatrix.column(vx, vy, bearingRad)
            H = FloatMatrix(3, 6)
            H[0, 2] = 1f
            H[1, 3] = 1f
            H[2, 4] = 1f
            nu = update(z, H, R, wrapAngleRow = 2)
        } else {
            val z = FloatMatrix.column(vx, vy)
            H = FloatMatrix(2, 6)
            H[0, 2] = 1f
            H[1, 3] = 1f
            val R2 = FloatMatrix.diag(R[0, 0], R[1, 1])
            nu = update(z, H, R2)
        }
        lastInnovation = nu

        if (adapt) {
            val Ht = H.transpose()
            val Radapt = if (useHeading) R else FloatMatrix.diag(R[0, 0], R[1, 1])
            val S = (H * Pbefore * Ht) + Radapt
            val K = Pbefore * Ht * (S.inverse() ?: return)
            sageHusa.step(Q, Radapt, nu, K, Pbefore, H)
        }
    }

    /** Apply a CNN-derived speed as a scalar speed measurement. */
    fun updateWithCnn(cnnSpeedMs: Float, rCnn: Float) {
        if (!cnnSpeedMs.isFinite() || !rCnn.isFinite()) return
        val psi = x[4, 0]
        val vx = cnnSpeedMs * cos(psi)
        val vy = cnnSpeedMs * sin(psi)
        val z = FloatMatrix.column(vx, vy)
        val H = FloatMatrix(2, 6)
        H[0, 2] = 1f
        H[1, 3] = 1f
        val safeR = rCnn.coerceAtLeast(0.5f)
        val Rcnn = FloatMatrix.diag(safeR, safeR)
        update(z, H, Rcnn)
    }
}
