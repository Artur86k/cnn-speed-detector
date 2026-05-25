package com.cnnspeed.app.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.cnnspeed.app.calibration.CalibrationEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

data class SpeedState(
    val gnssSpeedKmh: Float = 0f,
    val ekfSpeedKmh: Float = 0f,
    val cnnEkfSpeedKmh: Float = 0f,
    val hdop: Float = 99f,
    val satCount: Int = 0,
    val ekfQ: Float = 0f,
    val ekfR: Float = 0f,
    val cnnLoss: Float = 1f,
    val calibrating: Boolean = false,
    val cnnReady: Boolean = false,
    val gnssEnabled: Boolean = true,
    val logLines: List<String> = emptyList()
)

class SpeedViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = CalibrationEngine(app)
    private val _state = MutableStateFlow(SpeedState())
    val state: StateFlow<SpeedState> = _state.asStateFlow()

    private val logBuffer = ArrayDeque<String>(5)

    init {
        engine.start()
        viewModelScope.launch(Dispatchers.Default) {
            while (isActive) {
                val s = engine.state
                val line = "v_gnss=%.1f v_ekf=%.1f v_cnn=%.1f Q=%.3f R=%.3f L=%.3f".format(
                    s.gnssSpeedMs * 3.6f, s.ekfSpeedMs * 3.6f, s.cnnEkfSpeedMs * 3.6f,
                    s.ekfQMean, s.ekfRMean, s.cnnLoss
                )
                if (logBuffer.size >= 5) logBuffer.removeFirst()
                logBuffer.addLast(line)
                _state.value = SpeedState(
                    gnssSpeedKmh = s.gnssSpeedMs * 3.6f,
                    ekfSpeedKmh = s.ekfSpeedMs * 3.6f,
                    cnnEkfSpeedKmh = s.cnnEkfSpeedMs * 3.6f,
                    hdop = s.hdop,
                    satCount = s.satCount,
                    ekfQ = s.ekfQMean,
                    ekfR = s.ekfRMean,
                    cnnLoss = s.cnnLoss,
                    calibrating = s.calibrating,
                    cnnReady = s.cnnReady,
                    gnssEnabled = engine.gnssEnabled,
                    logLines = logBuffer.toList()
                )
                delay(100L)  // 10 Hz
            }
        }
    }

    fun setGnssEnabled(enabled: Boolean) {
        engine.gnssEnabled = enabled
    }

    override fun onCleared() {
        super.onCleared()
        engine.stop()
    }
}
