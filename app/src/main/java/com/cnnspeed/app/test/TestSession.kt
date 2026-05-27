package com.cnnspeed.app.test

import kotlin.math.abs
import kotlin.math.sqrt

/** One observation: GPS truth (m/s) paired with the latest CNN prediction (m/s). */
data class TestSample(
    val tNanos: Long,
    val gpsMs: Float,
    val cnnMs: Float,
    val hasGps: Boolean,
)

/** Bin definition for per-speed-band MAE statistics. */
data class SpeedBin(val loMs: Float, val hiMs: Float, val label: String)

/** Aggregate of the test session — predictions vs GPS, computed lazily. */
class TestSession {

    private val samples: MutableList<TestSample> = ArrayList()

    @Synchronized
    fun add(s: TestSample) { samples.add(s) }

    @Synchronized
    fun clear() { samples.clear() }

    @Synchronized
    fun snapshot(): List<TestSample> = ArrayList(samples)

    @Synchronized
    fun stats(): Stats {
        val labelled = samples.filter { it.hasGps }
        if (labelled.isEmpty()) return Stats.empty()
        val errs = FloatArray(labelled.size) { labelled[it].cnnMs - labelled[it].gpsMs }
        val gps  = FloatArray(labelled.size) { labelled[it].gpsMs }
        val maeMs  = errs.map { abs(it) }.average().toFloat()
        val rmseMs = sqrt(errs.map { it * it }.average()).toFloat()
        val biasMs = errs.average().toFloat()
        val perBin = BINS.map { bin ->
            val keep = BooleanArray(labelled.size) { gps[it] in bin.loMs..bin.hiMs }
            val n = keep.count { it }
            val mae = if (n == 0) 0f else
                (errs.indices.filter { keep[it] }.map { abs(errs[it]) }.average()).toFloat()
            BinStats(bin, n, mae)
        }
        return Stats(
            nTotal       = samples.size,
            nLabelled    = labelled.size,
            maeMs        = maeMs,
            rmseMs       = rmseMs,
            biasMs       = biasMs,
            bins         = perBin,
        )
    }

    data class BinStats(val bin: SpeedBin, val n: Int, val maeMs: Float)
    data class Stats(
        val nTotal:    Int,
        val nLabelled: Int,
        val maeMs:     Float,
        val rmseMs:    Float,
        val biasMs:    Float,
        val bins:      List<BinStats>,
    ) {
        companion object { fun empty() = Stats(0, 0, 0f, 0f, 0f, emptyList()) }
    }

    companion object {
        val BINS = listOf(
            SpeedBin( 0f,  1f, "0-1 m/s"),
            SpeedBin( 1f,  5f, "1-5"),
            SpeedBin( 5f, 15f, "5-15"),
            SpeedBin(15f, 25f, "15-25"),
            SpeedBin(25f, 50f, "25+"),
        )
    }
}
