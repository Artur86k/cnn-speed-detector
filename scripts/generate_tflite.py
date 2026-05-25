"""
Generate the frozen 786 -> 64 backbone used by SpeedCnn.

In this app the final 64 -> 1 regression head is trained on-device in pure Kotlin
(Option B from the spec), so the TFLite model output is the 64-d embedding
rather than the speed itself.

Run once and copy the resulting speed_cnn.tflite into app/src/main/assets/.

Requires: pip install tensorflow==2.14
"""
import tensorflow as tf

INPUT_DIM = 786
EMB_DIM = 64

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(INPUT_DIM,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(EMB_DIM, activation='relu', name='embedding'),
])
model.compile(optimizer='adam', loss='mse')

# Random init is fine — the on-device SGD head will adapt to whatever features
# the random projection produces (random-features / ELM style).

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

out_path = 'speed_cnn.tflite'
with open(out_path, 'wb') as f:
    f.write(tflite_model)
print(f"Saved {out_path} — {len(tflite_model)} bytes")
print("Copy to app/src/main/assets/speed_cnn.tflite")
