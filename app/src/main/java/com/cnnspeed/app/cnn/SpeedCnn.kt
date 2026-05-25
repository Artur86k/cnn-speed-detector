package com.cnnspeed.app.cnn

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Frozen TFLite backbone (786 → 64). The trainable head (64 → 1) lives in
 * [OnlineTrainer] and runs in pure Kotlin (Option B in the spec) so we don't
 * depend on TFLite's experimental on-device training.
 */
class SpeedCnn(context: Context) {

    private val interpreter: Interpreter?
    val embeddingSize: Int = 64
    val featureSize: Int = 786

    init {
        interpreter = try {
            val asset = context.assets.openFd("speed_cnn.tflite")
            val buf: MappedByteBuffer = FileInputStream(asset.fileDescriptor).channel
                .map(FileChannel.MapMode.READ_ONLY, asset.startOffset, asset.declaredLength)
            Interpreter(buf, Interpreter.Options().apply { setNumThreads(2) })
        } catch (_: Exception) {
            null  // model asset missing — fall back to pass-through
        }
    }

    /**
     * Forward the FFT feature vector through the backbone to obtain a 64-dim
     * embedding. If the model isn't available, return a dimensional reduction
     * fallback so the head can still train (degraded performance, but the app
     * stays alive).
     */
    fun embed(features: FloatArray): FloatArray {
        val out = FloatArray(embeddingSize)
        val itp = interpreter ?: run { return fallbackEmbedding(features, out) }
        val input = Array(1) { features }
        val output = Array(1) { out }
        return try {
            itp.run(input, output)
            output[0]
        } catch (_: Exception) {
            fallbackEmbedding(features, out)
        }
    }

    private fun fallbackEmbedding(features: FloatArray, out: FloatArray): FloatArray {
        // Average-pool the 786-vec down to 64-d so the head can still learn from raw FFT.
        val band = features.size / embeddingSize
        for (i in 0 until embeddingSize) {
            var s = 0f
            val start = i * band
            val end = if (i == embeddingSize - 1) features.size else start + band
            for (k in start until end) s += features[k]
            out[i] = s / (end - start).coerceAtLeast(1)
        }
        return out
    }

    fun close() { interpreter?.close() }
}
