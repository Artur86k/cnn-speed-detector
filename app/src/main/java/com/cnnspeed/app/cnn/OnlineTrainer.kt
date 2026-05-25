package com.cnnspeed.app.cnn

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Trainable linear head sitting on top of [SpeedCnn]'s 64-d embedding. Uses
 * plain SGD against GNSS-derived targets when the calibration gate is open.
 *
 * The head is f(x) = w·x̂ + b with x̂ ∈ ℝ^64 the L2-normalized embedding and
 * target = speed in m/s. Normalizing the embedding keeps gradients bounded
 * regardless of raw FFT magnitude, which prevents the SGD from diverging on
 * high-RMS samples (e.g. bumpy road, phone drop).
 *
 * Persists to filesDir/speed_cnn_head.bin every save() call.
 */
class OnlineTrainer(
    context: Context,
    private val cnn: SpeedCnn,
    private val replayBuffer: ReplayBuffer,
    private val lr: Float = 1e-3f,
    private val batchSize: Int = 64,
    private val trainEveryNSamples: Int = 30,
    private val minBufferToInfer: Int = 100,
    private val maxAbsGradient: Float = 5f,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val mutex = Mutex()
    private val weightsFile: File = File(context.filesDir, "speed_cnn_head.bin")

    @Volatile private var W: FloatArray = FloatArray(cnn.embeddingSize)
    @Volatile private var b: Float = 0f

    @Volatile var currentLoss: Float = 1f
        private set

    private val lossWindow = 32
    private val recentSquaredErr = FloatArray(lossWindow)
    private var lossIdx = 0
    private var lossFilled = 0

    private var newSampleCount: Int = 0

    init {
        loadIfExists()
    }

    /** Run the head on a single feature vector. Returns speed in m/s, or null. */
    fun infer(features: FloatArray): Float? {
        if (replayBuffer.size() < minBufferToInfer) return null
        if (!features.allFinite()) return null
        val emb = cnn.embed(features)
        if (!emb.allFinite()) return null
        val embN = l2Normalize(emb)
        val out = dot(W, embN) + b
        return if (out.isFinite()) out else null
    }

    private fun forward(embN: FloatArray): Float = dot(W, embN) + b

    fun notifyNewSample() {
        newSampleCount++
        if (newSampleCount >= trainEveryNSamples && replayBuffer.size() >= batchSize) {
            newSampleCount = 0
            scope.launch { trainBatch() }
        }
    }

    private suspend fun trainBatch() = mutex.withLock {
        val batch = replayBuffer.sample(batchSize)
        if (batch.isEmpty()) return@withLock
        var totalSq = 0f
        for (sample in batch) {
            if (!sample.features.allFinite() || !sample.targetMs.isFinite()) continue
            if (sample.targetMs !in 0f..80f) continue
            val emb = cnn.embed(sample.features)
            if (!emb.allFinite()) continue
            val embN = l2Normalize(emb)

            val pred = forward(embN)
            if (!pred.isFinite()) continue
            // Clip error to keep gradients bounded.
            val rawErr = pred - sample.targetMs
            val err = rawErr.coerceIn(-maxAbsGradient, maxAbsGradient)
            val sq = rawErr * rawErr
            totalSq += sq

            for (i in W.indices) {
                W[i] -= lr * err * embN[i]
            }
            b -= lr * err

            recentSquaredErr[lossIdx] = sq
            lossIdx = (lossIdx + 1) % lossWindow
            if (lossFilled < lossWindow) lossFilled++
        }
        // Hard recovery if SGD diverged for any reason.
        if (!W.allFinite() || !b.isFinite()) {
            for (i in W.indices) W[i] = 0f
            b = 0f
        }
        currentLoss = recentLossMean()
        save()
    }

    private fun recentLossMean(): Float {
        if (lossFilled == 0) return 1f
        var s = 0f
        for (i in 0 until lossFilled) s += recentSquaredErr[i]
        return s / lossFilled
    }

    private fun dot(a: FloatArray, c: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * c[i]
        return s
    }

    /** Return [v] divided by its L2 norm (floor 1e-6). Does not mutate input. */
    private fun l2Normalize(v: FloatArray): FloatArray {
        var s = 0f
        for (x in v) s += x * x
        val n = sqrt(s).coerceAtLeast(1e-6f)
        return FloatArray(v.size) { i -> v[i] / n }
    }

    private fun FloatArray.allFinite(): Boolean {
        for (x in this) if (!x.isFinite()) return false
        return true
    }

    private fun save() {
        try {
            val buf = ByteBuffer.allocate(4 + W.size * 4 + 4).order(ByteOrder.LITTLE_ENDIAN)
            buf.putInt(W.size)
            for (v in W) buf.putFloat(v)
            buf.putFloat(b)
            FileOutputStream(weightsFile).use { it.write(buf.array()) }
        } catch (_: Exception) { /* best effort */ }
    }

    private fun loadIfExists() {
        if (!weightsFile.exists()) return
        try {
            val bytes = FileInputStream(weightsFile).use { it.readBytes() }
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val n = buf.int
            if (n != W.size) return
            for (i in W.indices) W[i] = buf.float
            b = buf.float
            if (!W.allFinite() || !b.isFinite()) {
                for (i in W.indices) W[i] = 0f
                b = 0f
            }
        } catch (_: Exception) { /* ignore */ }
    }
}
