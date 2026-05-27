package com.cnnspeed.app

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.viewmodel.compose.viewModel
import com.cnnspeed.app.ui.LiveScreen
import com.cnnspeed.app.ui.StatsScreen
import com.cnnspeed.app.ui.TestViewModel

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
                    color = Color(0xFF050505),
                ) {
                    // Shared ViewModel keeps the TestEngine alive across screens.
                    val vm: TestViewModel = viewModel()
                    var showStats by remember { mutableStateOf(false) }
                    if (showStats) {
                        StatsScreen(onBack = { showStats = false }, vm = vm)
                    } else {
                        LiveScreen(onOpenStats = { showStats = true }, vm = vm)
                    }
                }
            }
        }
    }
}
