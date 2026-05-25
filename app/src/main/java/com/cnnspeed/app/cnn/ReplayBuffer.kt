package com.cnnspeed.app.cnn

import kotlin.math.sqrt

/** A single training sample: FFT feature vector + matching GNSS speed (m/s). */
data class TrainSample(val features: FloatArray, val targetMs: Float)

/**
 * Circular replay buffer. Rejects samples whose nearest neighbour in feature
 * space is within [epsilonL2] (diversity sampling on a low-dim signature).
 */
class ReplayBuffer(
    val capacity: Int = 500,
    private val epsilonL2: Float = 0.5f
) {
    private val buf = arrayOfNulls<TrainSample>(capacity)
    private var head = 0
    private var sz = 0
    private val lock = Any()

    fun size(): Int = synchronized(lock) { sz }

    /** Returns true if the sample was kept. */
    fun add(features: FloatArray, targetMs: Float): Boolean {
        val sig = signature(features)
        synchronized(lock) {
            if (sz > 0) {
                for (i in 0 until sz) {
                    val other = buf[i] ?: continue
                    if (l2Sig(other.features, sig) < epsilonL2) return false
                }
            }
            buf[head] = TrainSample(features.copyOf(), targetMs)
            head = (head + 1) % capacity
            if (sz < capacity) sz++
            return true
        }
    }

    fun sample(n: Int): List<TrainSample> {
        synchronized(lock) {
            if (sz == 0) return emptyList()
            val k = minOf(n, sz)
            val out = ArrayList<TrainSample>(k)
            val taken = BooleanArray(sz)
            val rng = java.util.Random()
            var picked = 0
            var attempts = 0
            while (picked < k && attempts < k * 8) {
                val idx = rng.nextInt(sz)
                if (!taken[idx]) {
                    buf[idx]?.let { out.add(it); taken[idx] = true; picked++ }
                }
                attempts++
            }
            return out
        }
    }

    private fun signature(f: FloatArray): FloatArray {
        // Coarse 8-d signature: rms over equal-width spectral bands.
        // Uses first (size - 12) values which are the spectral magnitudes.
        val specLen = f.size - 12
        if (specLen <= 0) return FloatArray(8)
        val sig = FloatArray(8)
        val band = specLen / 8
        for (b in 0 until 8) {
            var s = 0f
            val start = b * band
            val end = if (b == 7) specLen else start + band
            for (i in start until end) s += f[i] * f[i]
            sig[b] = sqrt(s / (end - start).coerceAtLeast(1))
        }
        return sig
    }

    private fun l2Sig(stored: FloatArray, querySig: FloatArray): Float {
        val sig = signature(stored)
        var s = 0f
        for (i in sig.indices) {
            val d = sig[i] - querySig[i]
            s += d * d
        }
        return sqrt(s)
    }
}
