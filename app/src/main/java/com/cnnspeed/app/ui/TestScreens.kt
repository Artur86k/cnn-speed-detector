package com.cnnspeed.app.ui

import android.app.Application
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.cnnspeed.app.test.TestEngine
import com.cnnspeed.app.test.TestEngineState
import com.cnnspeed.app.test.TestSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import androidx.compose.foundation.Canvas
import androidx.compose.ui.geometry.Offset

class TestViewModel(app: Application) : AndroidViewModel(app) {
    private val engine = TestEngine(app)
    val state: StateFlow<TestEngineState> = engine.state

    private val _statsTick = MutableStateFlow(0)
    val statsTick: StateFlow<Int> = _statsTick.asStateFlow()

    init {
        engine.start()
        // Refresh stats once per second so the Stats screen tracks new samples.
        viewModelScope.launch(Dispatchers.Default) {
            while (isActive) { delay(1000); _statsTick.value = _statsTick.value + 1 }
        }
    }

    fun setRecording(on: Boolean)   = engine.setRecording(on)
    fun clear()                      = engine.clearSession()
    fun sessionSnapshot()            = engine.session.snapshot()
    fun stats(): TestSession.Stats   = engine.session.stats()

    override fun onCleared() { super.onCleared(); engine.stop() }
}

// ---------- LIVE SCREEN -------------------------------------------------------

@Composable
fun LiveScreen(onOpenStats: () -> Unit, vm: TestViewModel = viewModel()) {
    val s by vm.state.collectAsState()

    Column(
        modifier = Modifier.fillMaxSize().padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            "CNN MODEL TEST",
            color = Color(0xFFCCCCCC),
            fontSize = 12.sp, fontFamily = FontFamily.Monospace
        )

        SpeedField(
            label = "GPS speed",
            valueKmh = s.gpsSpeedMs * 3.6f,
            color = Color(0xFF00E5FF),
            ok = s.hasFreshGps,
            subline = if (s.hasFreshGps) "${s.satCount} sats" else "no fix",
        )
        SpeedField(
            label = "CNN prediction",
            valueKmh = s.cnnSpeedMs * 3.6f,
            color = Color(0xFFFF6D00),
            ok = s.samplesReady,
            subline = if (s.samplesReady) "ring full · heading %.0f°".format(s.headingDeg)
                       else "warming up… ${s.samplesInRing} samples",
        )

        Spacer(Modifier.weight(1f))

        // Recording controls
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = { vm.setRecording(!s.recording) },
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (s.recording) Color(0xFFFF5252) else Color(0xFF00C853)
                )
            ) {
                Text(if (s.recording) "STOP REC" else "START REC",
                    fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
            }
            OutlinedButton(
                onClick = { vm.clear() },
                modifier = Modifier.weight(1f),
            ) { Text("CLEAR", fontFamily = FontFamily.Monospace) }
        }
        Button(
            onClick = onOpenStats,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF455A64)),
        ) {
            Text("VIEW STATS  ·  ${s.nRecorded} samples",
                fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
        }

        if (s.lastError != null) {
            Text("error: ${s.lastError}", color = Color(0xFFFF5252),
                fontFamily = FontFamily.Monospace, fontSize = 10.sp)
        }
    }
}

@Composable
private fun SpeedField(label: String, valueKmh: Float, color: Color, ok: Boolean, subline: String) {
    val border = if (ok) color.copy(alpha = 0.7f) else Color(0xFF444444)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFF111111), RoundedCornerShape(12.dp))
            .border(1.dp, border, RoundedCornerShape(12.dp))
            .padding(16.dp),
    ) {
        Text(label, color = Color(0xFFAAAAAA), fontSize = 12.sp,
            fontFamily = FontFamily.Monospace)
        Spacer(Modifier.height(4.dp))
        Row(verticalAlignment = Alignment.Bottom) {
            Text(
                "%.1f".format(valueKmh),
                color = if (ok) color else Color(0xFF666666),
                fontSize = 56.sp,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.width(6.dp))
            Text("km/h", color = Color(0xFF888888),
                fontSize = 16.sp, fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(bottom = 10.dp))
        }
        Text(subline, color = Color(0xFF888888),
            fontSize = 11.sp, fontFamily = FontFamily.Monospace)
    }
}

// ---------- STATS SCREEN ------------------------------------------------------

@Composable
fun StatsScreen(onBack: () -> Unit, vm: TestViewModel = viewModel()) {
    val tick by vm.statsTick.collectAsState()
    // remember per tick so recomposition picks up fresh stats
    val stats = remember(tick) { vm.stats() }
    val samples = remember(tick) { vm.sessionSnapshot() }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(onClick = onBack) {
                Text("←", fontFamily = FontFamily.Monospace)
            }
            Spacer(Modifier.width(12.dp))
            Text("RESULTS", color = Color(0xFFCCCCCC),
                fontSize = 14.sp, fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold)
        }

        if (stats.nLabelled == 0) {
            Text("No labelled samples yet. Hit START REC on the Live screen while driving with GPS lock.",
                color = Color(0xFF888888), fontSize = 13.sp,
                fontFamily = FontFamily.Monospace)
            return@Column
        }

        // ----- Summary card
        StatsCard {
            StatLine("samples",   "${stats.nLabelled} (of ${stats.nTotal} total)")
            StatLine("MAE",       "%.2f m/s   ·   %.1f km/h".format(stats.maeMs, stats.maeMs * 3.6f))
            StatLine("RMSE",      "%.2f m/s   ·   %.1f km/h".format(stats.rmseMs, stats.rmseMs * 3.6f))
            StatLine("bias",      "%+.2f m/s  ·  %+.1f km/h".format(stats.biasMs, stats.biasMs * 3.6f))
        }

        // ----- Time-series chart
        Text("GPS truth (cyan)  vs.  CNN prediction (orange)",
            color = Color(0xFFAAAAAA), fontSize = 11.sp,
            fontFamily = FontFamily.Monospace)
        TimeSeriesChart(samples = samples,
            modifier = Modifier.fillMaxWidth().height(220.dp))

        // ----- Scatter
        Text("Scatter   pred vs truth   (diagonal = perfect)",
            color = Color(0xFFAAAAAA), fontSize = 11.sp,
            fontFamily = FontFamily.Monospace)
        ScatterChart(samples = samples,
            modifier = Modifier.fillMaxWidth().height(220.dp))

        // ----- Per-bin MAE bars
        Text("MAE by speed bin", color = Color(0xFFAAAAAA),
            fontSize = 11.sp, fontFamily = FontFamily.Monospace)
        StatsCard {
            stats.bins.forEach { b ->
                Row(modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Text(b.bin.label.padEnd(8),
                        color = Color(0xFFCCCCCC), fontSize = 12.sp,
                        fontFamily = FontFamily.Monospace,
                        modifier = Modifier.width(70.dp))
                    Text("n=${b.n}".padEnd(7),
                        color = Color(0xFF888888), fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        modifier = Modifier.width(70.dp))
                    if (b.n == 0) {
                        Text("-", color = Color(0xFF555555),
                            fontFamily = FontFamily.Monospace, fontSize = 12.sp)
                    } else {
                        val frac = (b.maeMs / 6f).coerceIn(0f, 1f)   // scale to ~6 m/s
                        Box(modifier = Modifier
                            .height(14.dp)
                            .weight(frac.coerceAtLeast(0.02f))
                            .background(Color(0xFFFF6D00), RoundedCornerShape(2.dp)))
                        Spacer(Modifier.width(8.dp))
                        Text("%.2f m/s".format(b.maeMs),
                            color = Color(0xFFCCCCCC),
                            fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                    }
                }
            }
        }
    }
}

@Composable
private fun StatsCard(content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFF111111), RoundedCornerShape(10.dp))
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
        content = content,
    )
}

@Composable
private fun StatLine(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth()) {
        Text(label, color = Color(0xFF888888), fontSize = 12.sp,
            fontFamily = FontFamily.Monospace, modifier = Modifier.width(80.dp))
        Text(value, color = Color(0xFFEEEEEE), fontSize = 13.sp,
            fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun TimeSeriesChart(samples: List<com.cnnspeed.app.test.TestSample>, modifier: Modifier) {
    Canvas(modifier = modifier
        .background(Color(0xFF0A0A0A), RoundedCornerShape(8.dp))
        .padding(8.dp)) {
        if (samples.isEmpty()) return@Canvas
        val labelled = samples.filter { it.hasGps }
        if (labelled.isEmpty()) return@Canvas
        val maxV = (labelled.flatMap { listOf(it.gpsMs, it.cnnMs) }.maxOrNull()!! * 3.6f).coerceAtLeast(20f)
        val w = size.width; val h = size.height
        fun xFor(i: Int) = w * i / (labelled.size - 1).coerceAtLeast(1)
        fun yFor(v: Float) = h - (v / maxV) * h
        // GPS truth
        for (i in 1 until labelled.size) {
            drawLine(
                color = Color(0xFF00E5FF),
                start = Offset(xFor(i-1), yFor(labelled[i-1].gpsMs * 3.6f)),
                end   = Offset(xFor(i),   yFor(labelled[i  ].gpsMs * 3.6f)),
                strokeWidth = 2f,
            )
        }
        for (i in 1 until labelled.size) {
            drawLine(
                color = Color(0xFFFF6D00),
                start = Offset(xFor(i-1), yFor(labelled[i-1].cnnMs * 3.6f)),
                end   = Offset(xFor(i),   yFor(labelled[i  ].cnnMs * 3.6f)),
                strokeWidth = 2f,
            )
        }
    }
}

@Composable
private fun ScatterChart(samples: List<com.cnnspeed.app.test.TestSample>, modifier: Modifier) {
    Canvas(modifier = modifier
        .background(Color(0xFF0A0A0A), RoundedCornerShape(8.dp))
        .padding(8.dp)) {
        val labelled = samples.filter { it.hasGps }
        if (labelled.isEmpty()) return@Canvas
        val maxV = (labelled.flatMap { listOf(it.gpsMs, it.cnnMs) }.maxOrNull()!! * 3.6f).coerceAtLeast(20f)
        val w = size.width; val h = size.height
        // Diagonal
        drawLine(
            color = Color(0xFF444444),
            start = Offset(0f, h), end = Offset(w, 0f),
            strokeWidth = 1.5f,
        )
        labelled.forEach { s ->
            val x = (s.gpsMs * 3.6f / maxV) * w
            val y = h - (s.cnnMs * 3.6f / maxV) * h
            drawCircle(color = Color(0xFFFF6D00), radius = 3f, center = Offset(x, y))
        }
    }
}
