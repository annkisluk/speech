#!/bin/bash

# 1. Create Data Directory
echo "Creating data folder..."
mkdir -p data
cd data

# --- PART 1: NOISEX-92 SETUP ---
echo "Downloading NOISEX-92..."
git clone https://github.com/speechdnn/Noises.git

echo "Sorting noises into Pre-train and Incremental..."
mkdir -p pretrain_noises
mkdir -p incremental_noises

# Move the 10 "Base" tasks (Pre-train) — all from NOISEX-92
cp Noises/NoiseX-92/babble.wav pretrain_noises/
cp Noises/NoiseX-92/buccaneer1.wav pretrain_noises/
cp Noises/NoiseX-92/buccaneer2.wav pretrain_noises/
cp Noises/NoiseX-92/destroyerengine.wav pretrain_noises/
cp Noises/NoiseX-92/factory1.wav pretrain_noises/
cp Noises/NoiseX-92/factory2.wav pretrain_noises/
cp Noises/NoiseX-92/hfchannel.wav pretrain_noises/
cp Noises/NoiseX-92/leopard.wav pretrain_noises/
cp Noises/NoiseX-92/pink.wav pretrain_noises/
cp Noises/NoiseX-92/white.wav pretrain_noises/

# Move NOISEX-92 incremental noises (sessions 3 & 4)
cp Noises/NoiseX-92/destroyerops.wav incremental_noises/
cp Noises/NoiseX-92/machinegun.wav incremental_noises/

# --- PART 2: ESC-50 SETUP ---
echo "Downloading ESC-50..."
git clone https://github.com/karolpiczak/ESC-50.git

echo "Creating concatenated ESC-50 noise files for incremental sessions..."
# Concatenate all 40 clips of each category into a single long wav file
# so the mixing function can use them like NOISEX-92 single-file noises.
python3 - <<'PYEOF'
import csv, os, numpy as np, soundfile as sf, librosa

esc_audio = "ESC-50/audio"
esc_meta  = "ESC-50/meta/esc50.csv"
out_dir   = "incremental_noises"

categories = {
    "clock_alarm": "alarm.wav",
    "coughing":    "cough.wav",
}

with open(esc_meta) as f:
    rows = list(csv.DictReader(f))

for cat, out_name in categories.items():
    clips = sorted([r["filename"] for r in rows if r["category"] == cat])
    print(f"  {cat}: {len(clips)} clips -> {out_name}")
    audio_parts = []
    for clip in clips:
        y, sr = librosa.load(os.path.join(esc_audio, clip), sr=None)
        audio_parts.append(y)
    combined = np.concatenate(audio_parts)
    # Save at original 44100 Hz; prepare_data.py resamples to 8 kHz
    sf.write(os.path.join(out_dir, out_name), combined, sr)
    print(f"    -> {len(combined)/sr:.1f}s at {sr} Hz")

print("  ESC-50 concatenated noises ready.")
PYEOF

# --- PART 3: LIBRISPEECH SETUP ---
echo "Downloading LibriSpeech (This may take time)..."

# Dev-Clean (Validation)
curl -L -O https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzvf dev-clean.tar.gz
rm dev-clean.tar.gz

# Test-Clean (Final Exam)
curl -L -O https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzvf test-clean.tar.gz
rm test-clean.tar.gz

# Train-Clean-100 (Training Data)
curl -L -O https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzvf train-clean-100.tar.gz
rm train-clean-100.tar.gz

echo "Done! Your data folder is ready."