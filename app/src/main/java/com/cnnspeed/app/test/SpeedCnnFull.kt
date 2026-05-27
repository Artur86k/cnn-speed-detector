package com.cnnspeed.app.test

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Wraps the speed-regression TFLite trained on the 22-session corpus.
 *
 *   Input shape : (1, T, C) float32   default (1, 400, 6) = 2 s @ 200 Hz
 *   Output shape: (1, 1)    float32   predicted speed in m/s
 *
 * Per-channel normalisation stats are loaded from assets/cnn_norm.json (mean,
 * std written there by train.py). The same constants must be applied at
 * inference or predictions will be garbage.
 */
class SpeedCnnFull(context: Context) {

    private val interpreter: Interpreter
    val tSteps:  Int          // window length in samples (e.g. 400)
    val nChan:   Int          // channel count           (e.g. 6)
    val mean:    FloatArray   // length nChan
    val std:     FloatArray   // length nChan
    val targetHz: Float

    private val inputBuf: ByteBuffer
    private val outputBuf = Array(1) { FloatArray(1) }

    init {
        val asset = context.assets.openFd(MODEL_ASSET)
        val mapped: MappedByteBuffer = FileInputStream(asset.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, asset.startOffset, asset.declaredLength)
        interpreter = Interpreter(mapped, Interpreter.Options().apply { setNumThreads(2) })

        val cfg = JSONObject(context.assets.open(NORM_ASSET).bufferedReader().use { it.readText() })
        tSteps   = cfg.getInt("input_t")
        nChan    = cfg.getInt("input_c")
        targetHz = cfg.getDouble("target_hz").toFloat()
        mean     = cfg.getJSONArray("mean").let { FloatArray(it.length()) { i -> it.getDouble(i).toFloat() } }
        std      = cfg.getJSONArray("std" ).let { FloatArray(it.length()) { i -> it.getDouble(i).toFloat() } }
        require(mean.size == nChan && std.size == nChan) { "norm length != C" }

        inputBuf = ByteBuffer.allocateDirect(4 * tSteps * nChan).order(ByteOrder.nativeOrder())
    }

    /**
     * Predict speed in m/s from a (T, C) window of raw vehicle-frame features
     * (accel_fwd, accel_lat, accel_vert, gyro_x, gyro_y, gyro_z). Normalisation
     * happens inside.
     */
    fun predict(window: FloatArray): Float {
        require(window.size == tSteps * nChan) {
            "window size ${window.size} != T*C (${tSteps * nChan})"
        }
        inputBuf.rewind()
        // Apply (x - mean) / std per channel in-place into the input buffer.
        var i = 0
        for (t in 0 until tSteps) {
            for (c in 0 until nChan) {
                val z = (window[i] - mean[c]) / std[c]
                inputBuf.putFloat(z)
                i++
            }
        }
        inputBuf.rewind()
        interpreter.run(inputBuf, outputBuf)
        return outputBuf[0][0]
    }

    fun close() = interpreter.close()

    companion object {
        const val MODEL_ASSET = "speed_cnn_full.tflite"
        const val NORM_ASSET  = "cnn_norm.json"
    }
}
