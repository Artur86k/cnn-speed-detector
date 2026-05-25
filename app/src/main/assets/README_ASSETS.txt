Drop the generated `speed_cnn.tflite` here.

Generate with:
    pip install tensorflow==2.14
    python scripts/generate_tflite.py
    cp speed_cnn.tflite app/src/main/assets/

If the file is missing, the app still runs — SpeedCnn falls back to an
average-pool embedding so the Kotlin head can still train.
