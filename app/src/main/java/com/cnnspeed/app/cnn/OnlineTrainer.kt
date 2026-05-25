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

/**
 * Trainable linear head sitting on top of [SpeedCnn]'s 64-d embedding. Uses
 * plain SGD against GNSS-derived targets when the calibration gate is open.
 *
 * The head is f(x) = w·x + b with x ∈ ℝ^64 and target = speed in m/s.
 *
 * Persists to filesDir/speed_cnn_head.bin every save() call.
 */
class OnlineTrainer(
    context: Context,
    private val cnn: SpeedCnn,
    private val replayBuffer: ReplayBuffer,
    private val lr: Float = 1e-4f,
    private val batchSize: Int = 64,
    private val trainEveryNSamples: Int = 30,
    private val minBufferToInfer: Int = 100,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val mutex = Mutex()
    private val weightsFile: File = File(context.filesDir, "speed_cnn_head.bin")

    @Volatile private var W: FloatArray = FloatArray(cnn.embeddingSize)
    @Volatile private var b: Float = 0f

    @Volatile var currentLoss: Float = 1f
        private set

    /** Rolling MSE on last [lossWindow] samples. */
    private val lossWindow = 32
    private val recentSquaredErr = FloatArray(lossWindow)
    private var lossIdx = 0
    private var lossFilled = 0

    private var newSampleCount: Int = 0

    init {
        loadIfExists()
    }

    /** Run the head on a single embedding. Returns speed in m/s. */
    fun infer(features: FloatArray): Float? {
        if (replayBuffer.size() < minBufferToInfer) return null  // "TRAINING..."
        val emb = cnn.embed(features)
        return dot(W, emb) + b
    }

    /** Plain inference without the minBuffer guard – used internally. */
    private fun forward(emb: FloatArray): Float = dot(W, emb) + b

    /** Called by CalibrationEngine when a fresh (features, gnssSpeed) sample was added. */
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
            val emb = cnn.embed(sample.features)
            val pred = forward(emb)
            val err = pred - sample.targetMs
            val sq = err * err
            totalSq += sq
            // SGD step
            for (i in W.indices) W[i] -= lr * err * emb[i]
            b -= lr * err
            // Track rolling MSE
            recentSquaredErr[lossIdx] = sq
            lossIdx = (lossIdx + 1) % lossWindow
            if (lossFilled < lossWindow) lossFilled++
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
        } catch (_: Exception) { /* ignore */ }
    }
}
