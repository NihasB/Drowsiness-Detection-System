# model_trainer.py
"""
Optional: synthetic data training to create a small LSTM model that classifies
short sequences of EAR values as 'drowsy' vs 'alert'. This uses synthetic sequences,
so no external dataset is required.
"""
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

MODEL_PATH = "models/ear_seq_model.h5"
SEQ_LEN = 30

def generate_synthetic(n_samples=2000, seq_len=SEQ_LEN):
    X = []
    y = []
    for _ in range(n_samples):
        # decide label randomly
        label = np.random.rand() < 0.5  # True -> drowsy
        if label:
            # drowsy: many low EAR values, with small noise
            base = np.random.uniform(0.10, 0.18)
            seq = base + np.random.normal(0, 0.02, seq_len)
            # optionally insert small peaks
            for _ in range(np.random.randint(0, 3)):
                idx = np.random.randint(0, seq_len)
                seq[idx] += np.random.uniform(0.02, 0.05)
        else:
            # alert: mostly higher EAR
            base = np.random.uniform(0.23, 0.35)
            seq = base + np.random.normal(0, 0.02, seq_len)
            for _ in range(np.random.randint(0, 3)):
                idx = np.random.randint(0, seq_len)
                seq[idx] -= np.random.uniform(0.02, 0.04)
        seq = np.clip(seq, 0.05, 0.5)
        X.append(seq.reshape(seq_len, 1))
        y.append(1 if label else 0)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

def build_model(seq_len=SEQ_LEN):
    model = Sequential([
        LSTM(32, input_shape=(seq_len, 1), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    os.makedirs("models", exist_ok=True)
    X, y = generate_synthetic(n_samples=2000)
    model = build_model()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.15, callbacks=callbacks)
    model.save(MODEL_PATH)
    print("Saved model to", MODEL_PATH)

if __name__ == "__main__":
    main()
