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

    /** CTRV predict step. ax is forward acceleration in m/s², gz is yaw rate from gyro. */
    fun predict(dt: Float, ax: Float, gz: Float) {
        if (dt <= 0f) return

        val px = x[0, 0]
        val py = x[1, 0]
        val vx = x[2, 0]
        val vy = x[3, 0]
        val psi = x[4, 0]
        val psiDot = x[5, 0]

        // Forward speed magnitude (sign retained via current heading)
        val v = kotlin.math.sqrt(vx * vx + vy * vy)
        val newV = (v + ax * dt).coerceAtLeast(0f)

        // CTRV motion update; if yaw rate ~ 0, fall back to straight line.
        val newPsi = psi + psiDot * dt
        val newPx: Float
        val newPy: Float
        if (abs(psiDot) < 1e-4f) {
            newPx = px + newV * cos(psi) * dt
            newPy = py + newV * sin(psi) * dt
        } else {
            newPx = px + (newV / psiDot) * (sin(newPsi) - sin(psi))
            newPy = py + (newV / psiDot) * (-cos(newPsi) + cos(psi))
        }
        val newVx = newV * cos(newPsi)
        val newVy = newV * sin(newPsi)
        val newPsiDot = gz  // measured directly, treat as noisy input

        x[0, 0] = newPx
        x[1, 0] = newPy
        x[2, 0] = newVx
        x[3, 0] = newVy
        x[4, 0] = wrap(newPsi)
        x[5, 0] = newPsiDot

        // Jacobian F = ∂f/∂x. Approximate (diagonal-heavy) – good enough for short dt.
        val F = FloatMatrix.identity(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = dt

        P = F * P * F.transpose() + Q
        P.addDiag(epsP)
    }

    /** Generic measurement update. z (m×1), H (m×n), R (m×m). */
    fun update(z: FloatMatrix, H: FloatMatrix, R: FloatMatrix): FloatMatrix {
        val zPred = H * x
        val nu = z - zPred                            // innovation (m×1)
        val Ht = H.transpose()
        val S = (H * P * Ht) + R                      // m×m
        val Sinv = S.inverse() ?: return nu           // fail-safe; skip update
        val K = P * Ht * Sinv                         // n×m

        x = x + K * nu
        // Joseph form would be more stable but we'll use the simple (I-KH)P form.
        val IKH = eye - (K * H)
        P = IKH * P
        x[4, 0] = wrap(x[4, 0])
        return nu
    }

    fun speedMs(): Float {
        val vx = x[2, 0]; val vy = x[3, 0]
        return kotlin.math.sqrt(vx * vx + vy * vy)
    }

    fun headingRad(): Float = x[4, 0]

    private fun wrap(a: Float): Float {
        var r = a
        while (r > Math.PI) r -= (2 * Math.PI).toFloat()
        while (r < -Math.PI) r += (2 * Math.PI).toFloat()
        return r
    }
}
