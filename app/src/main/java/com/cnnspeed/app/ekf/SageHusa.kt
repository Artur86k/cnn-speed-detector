package com.cnnspeed.app.ekf

import kotlin.math.pow

/**
 * Sage-Husa online estimator of process noise Q and measurement noise R from
 * the innovation sequence. Caller invokes [step] every time a measurement
 * update completes with a known-good measurement.
 *
 * b ∈ (0,1)  forgetting factor (closer to 1 = more stable, less adaptive)
 */
class SageHusa(
    private val b: Float = 0.97f,
    private val rMinDiag: Float = 0.01f,
    private val qMinDiag: Float = 1e-4f
) {
    private var k: Int = 0

    /**
     * Update Q and R in-place from the innovation [nu] (m×1), Kalman gain [K] (n×m),
     * predicted covariance [Ppred] (n×n) and measurement matrix [H] (m×n).
     */
    fun step(
        Q: FloatMatrix, R: FloatMatrix,
        nu: FloatMatrix, K: FloatMatrix,
        Ppred: FloatMatrix, H: FloatMatrix
    ) {
        k++
        val d = ((1f - b) / (1f - b.pow(k.toFloat()))).coerceIn(0f, 0.5f)
        val one = 1f - d

        // Q ← (1-d) Q + d · K νν^T K^T
        val Knu = K * nu                              // n×1
        val outer = Knu * Knu.transpose()             // n×n
        for (i in 0 until Q.rows) for (j in 0 until Q.cols) {
            Q[i, j] = one * Q[i, j] + d * outer[i, j]
        }
        for (i in 0 until Q.rows) if (Q[i, i] < qMinDiag) Q[i, i] = qMinDiag

        // R ← (1-d) R + d (νν^T - H Ppred H^T)
        val nnT = nu * nu.transpose()
        val hPh = H * Ppred * H.transpose()
        for (i in 0 until R.rows) for (j in 0 until R.cols) {
            val term = nnT[i, j] - hPh[i, j]
            R[i, j] = one * R[i, j] + d * term
        }
        for (i in 0 until R.rows) if (R[i, i] < rMinDiag) R[i, i] = rMinDiag
    }
}
