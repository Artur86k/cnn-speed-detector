package com.cnnspeed.app

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.cnnspeed.app.ui.SpeedCard
import com.cnnspeed.app.ui.SpeedState
import com.cnnspeed.app.ui.SpeedViewModel

class MainActivity : ComponentActivity() {

    private val permLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* engine reads permission state on its own; no-op */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        permLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color(0xFF0A0A0A)
                ) {
                    val vm: SpeedViewModel = viewModel()
                    val state by vm.state.collectAsState()
                    Dashboard(state, onGnssToggle = { enabled -> vm.setGnssEnabled(enabled) })
                }
            }
        }
    }
}

@Composable
private fun Dashboard(s: SpeedState, onGnssToggle: (Boolean) -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(top = 24.dp, bottom = 8.dp)
    ) {
        Text(
            "IMU + GNSS + CNN SPEED",
            color = Color(0xFFCCCCCC),
            fontSize = 12.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )
        SpeedCard(
            icon = "📡",
            title = if (s.gnssEnabled) "GPS Speed (Raw GNSS)" else "GPS Speed (Raw GNSS, NOT FUSED)",
            speedText = "%.1f km/h".format(s.gnssSpeedKmh),
            subtitle = "HDOP %.1f  ·  %d sat".format(s.hdop, s.satCount),
            color = Color(0xFF00E5FF)
        )
        SpeedCard(
            icon = "🔵",  // 🔵
            title = "EKF Speed (IMU + GNSS calibrated)",
            speedText = "%.1f km/h".format(s.ekfSpeedKmh),
            subtitle = "Q %.3f  ·  R %.3f".format(s.ekfQ, s.ekfR),
            color = Color(0xFF69FF47)
        )
        SpeedCard(
            icon = "🧠",
            title = "EKF + CNN Speed (Neural Network)",
            speedText = if (s.cnnReady) "%.1f km/h".format(s.cnnEkfSpeedKmh) else "TRAINING…",
            subtitle = "CNN loss %.4f".format(s.cnnLoss),
            color = Color(0xFFFF6D00)
        )

        Spacer(Modifier.weight(1f))

        StatusBar(s)
        Spacer(Modifier.height(8.dp))
        GnssToggleRow(disabled = !s.gnssEnabled, onToggle = { onGnssToggle(!it) })
        Spacer(Modifier.height(8.dp))
        LogPanel(s.logLines)
    }
}

@Composable
private fun GnssToggleRow(disabled: Boolean, onToggle: (Boolean) -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            "DISABLE GNSS",
            color = if (disabled) Color(0xFFFF5252) else Color(0xFFAAAAAA),
            fontFamily = FontFamily.Monospace,
            fontSize = 13.sp,
            modifier = Modifier.weight(1f)
        )
        Switch(
            checked = disabled,
            onCheckedChange = onToggle,
            colors = SwitchDefaults.colors(
                checkedThumbColor = Color(0xFFFF5252),
                checkedTrackColor = Color(0xFF441111),
                uncheckedThumbColor = Color(0xFF666666),
                uncheckedTrackColor = Color(0xFF222222)
            )
        )
    }
}

@Composable
private fun StatusBar(s: SpeedState) {
    val accentColor = if (s.calibrating) Color(0xFF00E676) else Color(0xFFFFA000)
    val label = if (s.calibrating) "● CALIBRATING" else "● COASTING"
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            label,
            color = accentColor,
            fontFamily = FontFamily.Monospace,
            fontSize = 13.sp
        )
        Spacer(Modifier.width(12.dp))
        Text(
            "%d sats · HDOP %.1f".format(s.satCount, s.hdop),
            color = Color(0xFFAAAAAA),
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp
        )
    }
}

@Composable
private fun LogPanel(lines: List<String>) {
    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .height(80.dp)
            .padding(horizontal = 16.dp)
            .background(Color(0xFF050505))
            .padding(8.dp)
    ) {
        items(lines) { line ->
            Text(
                line,
                color = Color(0xFF6FCF97),
                fontFamily = FontFamily.Monospace,
                fontSize = 10.sp
            )
        }
    }
}
