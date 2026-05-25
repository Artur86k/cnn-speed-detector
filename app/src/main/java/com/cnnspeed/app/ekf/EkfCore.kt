package com.cnnspeed.app.ekf

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * CTRV-based EKF.
 *
 * State x = [px, py, vx, vy, ψ, ψ̇]ᵀ  (length 6)
 * Position in metres (arbitrary local frame), velocity in m/s, heading in rad,
 * yaw rate in rad/s.
 */
open class EkfCore {
    // State vector (6x1)
    var x: FloatMatrix = FloatMatrix(6, 1)

    // Covariance (6x6) - initialized large to allow fast convergence
    var P: FloatMatrix = FloatMatrix.identity(6) * 10f

    // Process noise (6x6) - tuned conservatively, may be adapted online
    var Q: FloatMatrix = FloatMatrix.diag(
        0.1f, 0.1f,   // position
        1.0f, 1.0f,   // velocity
        0.05f,        // heading
        0.1f          // yaw rate
    )

    private val eye = FloatMatrix.identity(6)
    private val epsP = 1e-6f

    /** Maximum plausible vehicle speed before we declare the filter diverged. */
    private val maxSpeedMs = 80f   // ≈ 288 km/h

    /** CTRV predict step. ax is forward (along-heading) acceleration in m/s². */
    fun predict(dt: Float, ax: Float, gz: Float) {
        if (dt <= 0f || dt > 0.2f) return
        if (!ax.isFinite() || !gz.isFinite()) return

        val px = x[0, 0]
        val py = x[1, 0]
        val vx = x[2, 0]
        val vy = x[3, 0]
        val psi = x[4, 0]
        val psiDot = x[5, 0]

        val v = kotlin.math.sqrt(vx * vx + vy * vy)
        // Cap accel contribution so a single bad sample can't blow up v.
        val axClipped = ax.coerceIn(-20f, 20f)
        val newV = (v + axClipped * dt).coerceIn(0f, maxSpeedMs)

        val newPsi = psi + psiDot * dt
        val newPx: Float
        val newPy: Float
        if (abs(psiDot) < 1e-3f) {
            newPx = px + newV * cos(psi) * dt
            newPy = py + newV * sin(psi) * dt
        } else {
            newPx = px + (newV / psiDot) * (sin(newPsi) - sin(psi))
            newPy = py + (newV / psiDot) * (-cos(newPsi) + cos(psi))
        }
        val newVx = newV * cos(newPsi)
        val newVy = newV * sin(newPsi)
        val newPsiDot = gz.coerceIn(-3f, 3f)

        x[0, 0] = newPx
        x[1, 0] = newPy
        x[2, 0] = newVx
        x[3, 0] = newVy
        x[4, 0] = wrap(newPsi)
        x[5, 0] = newPsiDot

        val F = FloatMatrix.identity(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = dt

        P = F * P * F.transpose() + Q
        P.addDiag(epsP)
        sanitize()
    }

    /**
     * Generic measurement update.
     *
     * @param wrapAngleRow row index in z whose innovation is an angle and
     *   must be wrapped into (−π, π]. -1 disables wrapping.
     */
    fun update(z: FloatMatrix, H: FloatMatrix, R: FloatMatrix, wrapAngleRow: Int = -1): FloatMatrix {
        val zPred = H * x
        val nu = z - zPred
        if (wrapAngleRow in 0 until nu.rows) {
            nu[wrapAngleRow, 0] = wrap(nu[wrapAngleRow, 0])
        }
        val Ht = H.transpose()
        val S = (H * P * Ht) + R
        val Sinv = S.inverse() ?: return nu          // fail-safe; skip update
        val K = P * Ht * Sinv

        x = x + K * nu
        // Joseph form keeps P symmetric and positive-semidefinite:
        //     P = (I - K H) P (I - K H)ᵀ + K R Kᵀ
        val IKH = eye - (K * H)
        val IKHt = IKH.transpose()
        val Kt = K.transpose()
        P = (IKH * P * IKHt) + (K * R * Kt)
        x[4, 0] = wrap(x[4, 0])
        sanitize()
        return nu
    }

    fun speedMs(): Float {
        val vx = x[2, 0]; val vy = x[3, 0]
        return kotlin.math.sqrt(vx * vx + vy * vy)
    }

    fun headingRad(): Float = x[4, 0]

    /** Reset state and covariance to the initial (high-uncertainty) values. */
    fun reset() {
        x = FloatMatrix(6, 1)
        P = FloatMatrix.identity(6) * 10f
    }

    /**
     * Recovery: clamp out-of-range values, reset on NaN/Inf or divergence.
     * Called after every predict/update.
     */
    private fun sanitize() {
        for (v in x.data) {
            if (!v.isFinite()) { reset(); return }
        }
        for (v in P.data) {
            if (!v.isFinite()) { reset(); return }
        }
        val s = speedMs()
        if (s > maxSpeedMs * 1.5f) {
            // Rescale velocity components to the cap rather than full reset
            // (so heading and position estimates survive).
            val scale = maxSpeedMs / s
            x[2, 0] = x[2, 0] * scale
            x[3, 0] = x[3, 0] * scale
        }
    }

    private fun wrap(a: Float): Float {
        var r = a
        while (r > Math.PI) r -= (2 * Math.PI).toFloat()
        while (r < -Math.PI) r += (2 * Math.PI).toFloat()
        return r
    }
}
