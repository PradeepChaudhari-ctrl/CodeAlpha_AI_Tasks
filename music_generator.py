"""
CodeAlpha AI Internship — Task 3: Music Generation with AI
Author: CodeAlpha Intern
Description: LSTM-based music generator using music21 for MIDI processing.
             Trains on a built-in sample sequence and generates new music.
             Run:  python music_generator.py
"""

import random
import os

# ─────────────────────────── Try imports ─────────────────────────────
try:
    import numpy as np
    from music21 import stream, note, chord, instrument, tempo, midi
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    FULL_MODE = True
except ImportError:
    FULL_MODE = False
    print("⚠️  Some libraries missing. Running in DEMO mode (generates MIDI without training).")
    print("   Install with:  pip install tensorflow music21 numpy\n")


# ─────────────────────────── Constants ───────────────────────────────
SEQ_LENGTH   = 20
EPOCHS       = 50
BATCH_SIZE   = 32
LATENT_UNITS = 128
OUTPUT_DIR   = "generated_music"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Built-in sample sequences (classical / jazz motifs)
SAMPLE_SEQUENCES = [
    ["C4","E4","G4","C5","G4","E4","C4","D4","F4","A4","D5","A4","F4","D4"],
    ["A4","C5","E5","A5","G5","E5","C5","A4","F4","A4","C5","E5","D5","B4"],
    ["E4","G4","B4","E5","D5","B4","G4","E4","F4","A4","C5","F5","E5","C5"],
    ["C4","D4","E4","F4","G4","A4","B4","C5","B4","A4","G4","F4","E4","D4"],
    ["G4","B4","D5","G5","F5","D5","B4","G4","A4","C5","E5","A5","G5","E5"],
]

ALL_NOTES = sorted({n for seq in SAMPLE_SEQUENCES for n in seq})
NOTE_TO_INT = {n: i for i, n in enumerate(ALL_NOTES)}
INT_TO_NOTE = {i: n for n, i in NOTE_TO_INT.items()}
VOCAB_SIZE  = len(ALL_NOTES)


# ─────────────────────────── Data Prep ───────────────────────────────
def build_dataset(sequences, seq_len=SEQ_LENGTH):
    """Convert note sequences into (X, y) integer arrays."""
    flat = [n for seq in sequences for n in seq]
    encoded = [NOTE_TO_INT[n] for n in flat]
    X, y = [], []
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i: i + seq_len])
        y.append(encoded[i + seq_len])
    return np.array(X), np.array(y)


# ─────────────────────────── Model ───────────────────────────────────
def build_model(vocab_size: int, seq_len: int) -> "tf.keras.Model":
    model = Sequential([
        Embedding(vocab_size, 64, input_length=seq_len),
        LSTM(LATENT_UNITS, return_sequences=True),
        Dropout(0.3),
        LSTM(LATENT_UNITS),
        Dropout(0.3),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    return model


# ─────────────────────────── Generation ──────────────────────────────
def generate_notes(model, seed_seq: list[int], n_notes: int = 50,
                   temperature: float = 0.8) -> list[str]:
    """Auto-regressively generate notes from a seed sequence."""
    generated = list(seed_seq)
    for _ in range(n_notes):
        x = np.array([generated[-SEQ_LENGTH:]])
        preds = model.predict(x, verbose=0)[0].astype("float64")
        # Temperature scaling
        preds = np.log(preds + 1e-10) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_idx = np.random.choice(len(preds), p=preds)
        generated.append(next_idx)
    return [INT_TO_NOTE[i] for i in generated[-n_notes:]]


# ─────────────────────────── MIDI Export ─────────────────────────────
def notes_to_midi(note_list: list[str], output_path: str,
                  bpm: int = 120, duration: float = 0.5):
    """Convert a list of note strings to a MIDI file."""
    score = stream.Score()
    part  = stream.Part()
    part.append(instrument.Piano())
    part.append(tempo.MetronomeMark(number=bpm))

    for note_str in note_list:
        try:
            n = note.Note(note_str)
            n.duration.quarterLength = duration
            part.append(n)
        except Exception:
            # Skip invalid notes silently
            pass

    score.append(part)
    midi_file = midi.translate.music21ObjectToMidiFile(score)
    midi_file.open(output_path, "wb")
    midi_file.write()
    midi_file.close()
    print(f"🎵 MIDI saved → {output_path}")


# ─────────────────────────── Demo mode ───────────────────────────────
def demo_generate(output_path: str):
    """Generate a random music21 MIDI without ML training."""
    print("\n🎹 Demo Mode: Generating music using rule-based patterns…")
    scale = ["C4","D4","E4","F4","G4","A4","B4","C5",
             "D5","E5","F5","G5","A5","B5","C6"]
    pattern = []
    prev = random.choice(scale[:8])
    for _ in range(60):
        idx = scale.index(prev)
        step = random.choice([-2, -1, 0, 1, 1, 2])
        idx  = max(0, min(len(scale) - 1, idx + step))
        prev = scale[idx]
        pattern.append(prev)

    notes_to_midi(pattern, output_path)


# ─────────────────────────── Main ────────────────────────────────────
def main():
    print("=" * 60)
    print("  CodeAlpha AI Internship — Task 3: Music Generation")
    print("=" * 60)

    output_path = os.path.join(OUTPUT_DIR, "generated_music.mid")

    if not FULL_MODE:
        demo_generate(output_path)
        return

    # ── Full ML pipeline ──────────────────────────────────────────
    print("\n📂 Preparing dataset…")
    X, y = build_dataset(SAMPLE_SEQUENCES)
    print(f"   Samples: {len(X)}  |  Vocab: {VOCAB_SIZE}  |  Seq len: {SEQ_LENGTH}")

    print("\n🏗️  Building LSTM model…")
    model = build_model(VOCAB_SIZE, SEQ_LENGTH)

    checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="loss", patience=10, restore_best_weights=True, verbose=1),
    ]

    print(f"\n🚀 Training for up to {EPOCHS} epochs…")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

    print("\n🎼 Generating music sequence…")
    seed = list(X[random.randint(0, len(X) - 1)])
    generated = generate_notes(model, seed, n_notes=60, temperature=0.8)
    print(f"   Generated notes: {generated[:20]}…")

    notes_to_midi(generated, output_path)
    print("\n✅ Done! Open the MIDI file with any media player or DAW.")
    print(f"   File: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
