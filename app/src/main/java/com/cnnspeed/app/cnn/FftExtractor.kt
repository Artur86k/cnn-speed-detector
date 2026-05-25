package com.cnnspeed.app.cnn

import com.cnnspeed.app.sensors.ImuSample
import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Sliding FFT extractor for the 3 accel channels. Maintains a parallel ring of
 * the last [windowSize] accel samples. When a new window's worth of samples has
 * accumulated past the previous one ([hop] = windowSize/2), emit a feature
 * vector for the CNN.
 *
 * Feature layout (786 floats):
 *   [257 mag bins ax | 257 mag bins ay | 257 mag bins az |
 *    centroid_x, centroid_y, centroid_z,
 *    rms_ax, rms_ay, rms_az,
 *    kurtosis_ax, kurtosis_ay, kurtosis_az,
 *    dominant_freq_ax, dominant_freq_ay, dominant_freq_az]
 */
class FftExtractor(
    val windowSize: Int = 512,
    val sampleRateHz: Float = 200f
) {
    val bins: Int = windowSize / 2 + 1
    val featureSize: Int = bins * 3 + 12

    private val hop = windowSize / 2
    private val ax = FloatArray(windowSize)
    private val ay = FloatArray(windowSize)
    private val az = FloatArray(windowSize)
    private var head = 0
    private var filled = 0
    private var sinceLastEmit = 0

    private val fft = FloatFFT_1D(windowSize.toLong())
    private val hann = FloatArray(windowSize) { i ->
        0.5f * (1f - cos(2.0 * Math.PI * i / (windowSize - 1)).toFloat())
    }

    /** Latest emitted feature vector (or null). Used by CalibrationEngine. */
    @Volatile var latestFeatures: FloatArray? = null
        private set

    /** Callback fired when a new feature vector is ready. */
    var onFeatures: ((FloatArray) -> Unit)? = null

    fun addSample(s: ImuSample) {
        ax[head] = s.ax; ay[head] = s.ay; az[head] = s.az
        head = (head + 1) % windowSize
        if (filled < windowSize) filled++
        sinceLastEmit++

        if (filled >= windowSize && sinceLastEmit >= hop) {
            sinceLastEmit = 0
            emitFeatures()
        }
    }

    private fun emitFeatures() {
        val features = FloatArray(featureSize)
        val cx = computeChannel(ax, features, 0)
        val cy = computeChannel(ay, features, bins)
        val cz = computeChannel(az, features, 2 * bins)

        var idx = 3 * bins
        features[idx++] = cx.centroid
        features[idx++] = cy.centroid
        features[idx++] = cz.centroid
        features[idx++] = cx.rms
        features[idx++] = cy.rms
        features[idx++] = cz.rms
        features[idx++] = cx.kurt
        features[idx++] = cy.kurt
        features[idx++] = cz.kurt
        features[idx++] = cx.dominantHz
        features[idx++] = cy.dominantHz
        features[idx]   = cz.dominantHz

        latestFeatures = features
        onFeatures?.invoke(features)
    }

    /** Per-channel summary stats. */
    private data class ChanStats(val rms: Float, val kurt: Float, val centroid: Float, val dominantHz: Float)

    private fun computeChannel(src: FloatArray, dst: FloatArray, dstOffset: Int): ChanStats {
        // Copy windowed samples from ring buffer into linear array in chronological order.
        val buf = FloatArray(windowSize)
        for (i in 0 until windowSize) {
            val idx = (head + i) % windowSize  // oldest first
            buf[i] = src[idx] * hann[i]
        }

        // Time-domain stats on the windowed signal (after de-meaning).
        var mean = 0f
        for (v in buf) mean += v
        mean /= windowSize
        var sumSq = 0f; var sum4 = 0f
        for (v in buf) {
            val d = v - mean
            val d2 = d * d
            sumSq += d2
            sum4 += d2 * d2
        }
        val variance = sumSq / windowSize
        val rms = sqrt(variance)
        val kurt = if (variance > 1e-9f) (sum4 / windowSize) / (variance * variance) - 3f else 0f

        // FFT in place. Output layout: [re0, re_{N/2}, re1, im1, re2, im2, ...]
        val fftBuf = buf.copyOf()
        fft.realForward(fftBuf)

        var maxMag = 0f
        var maxBin = 0
        var energy = 0f
        var energyTimesFreq = 0f
        val nyquist = sampleRateHz / 2f
        val freqStep = nyquist / (bins - 1)

        // Bin 0 (DC)
        val mag0 = kotlin.math.abs(fftBuf[0])
        dst[dstOffset] = mag0

        // Bin N/2 (Nyquist) lives at index 1 in JTransforms layout
        val magNyq = kotlin.math.abs(fftBuf[1])
        dst[dstOffset + bins - 1] = magNyq

        // Bins 1..N/2-1
        for (k in 1 until bins - 1) {
            val re = fftBuf[2 * k]
            val im = fftBuf[2 * k + 1]
            val mag = sqrt(re * re + im * im)
            dst[dstOffset + k] = mag
            if (mag > maxMag) { maxMag = mag; maxBin = k }
            energy += mag
            energyTimesFreq += mag * (k * freqStep)
        }
        if (mag0 > maxMag) { maxMag = mag0; maxBin = 0 }
        if (magNyq > maxMag) { maxMag = magNyq; maxBin = bins - 1 }

        val centroid = if (energy > 1e-9f) energyTimesFreq / energy else 0f
        val dominantHz = maxBin * freqStep
        return ChanStats(rms = rms, kurt = kurt, centroid = centroid, dominantHz = dominantHz)
    }
}
