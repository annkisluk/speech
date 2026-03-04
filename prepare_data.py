
import os
import shutil
import argparse
import numpy as np
import soundfile as sf
import librosa
import json
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Tuple, Dict

# CONFIGURATION
class Config:    
    #PATHS
    LIBRISPEECH_ROOT = "data/LibriSpeech"
    PRETRAIN_NOISE_DIR = "data/pretrain_noises"
    INCREMENTAL_NOISE_DIR = "data/incremental_noises"
    OUTPUT_ROOT = "data/final_data"  # Clean organized output
    
    #AUDIO PARAMETERS
    TARGET_SR = 8000  # 8 kHz 
    
    # SNR CONFIGURATION
    TRAIN_SNR_LEVELS = [-5, 0, 5, 10]  # 4 levels for training
    VAL_TEST_SNR_RANGE = (-5, 10)      # Random SNR for val/test
    
    # SESSION 0 (PRE-TRAIN) SPECIFICATIONS 
    SESSION0_TRAIN_UTTERANCES = 1010
    SESSION0_TRAIN_SPEAKERS = 101
    SESSION0_VAL_UTTERANCES = 1206
    SESSION0_VAL_SPEAKERS = 10
    SESSION0_TEST_UTTERANCES = 651
    SESSION0_TEST_SPEAKERS = 8
    
    # INCREMENTAL SESSIONS SPECIFICATIONS
    INCREMENTAL_TRAIN_UTTERANCES = 303
    INCREMENTAL_TRAIN_SPEAKERS = 101
    INCREMENTAL_VAL_UTTERANCES = 1206
    INCREMENTAL_VAL_SPEAKERS = 10
    INCREMENTAL_TEST_UTTERANCES = 651
    INCREMENTAL_TEST_SPEAKERS = 8
    
    # NOISE ASSIGNMENTS
    SESSION0_NOISES = [
        "babble.wav", "buccaneer1.wav", "buccaneer2.wav",
        "destroyerengine.wav", "factory1.wav", "factory2.wav",
        "hfchannel.wav", "leopard.wav", "pink.wav", "white.wav"
    ]
    
    INCREMENTAL_SESSIONS = [
        {"id": 1, "noise": "alarm.wav", "name": "alarm"},          # ESC-50 clock_alarm
        {"id": 2, "noise": "cough.wav", "name": "cough"},          # ESC-50 coughing
        {"id": 3, "noise": "destroyerops.wav", "name": "destroyerops"},  # NOISEX-92
        {"id": 4, "noise": "machinegun.wav", "name": "machinegun"},      # NOISEX-92
    ]
    
    # REPRODUCIBILITY
    RANDOM_SEED = 42

# UTILITY FUNCTIONS

def set_seed(seed: int):
    #Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

def print_header(text: str):
    #Print formatted section header
    print(f"\n{'='*80}")
    print(f"{text.center(80)}")
    print(f"{'='*80}\n")

def load_audio(file_path: str, target_sr: int = 8000) -> np.ndarray:
    #Load and resample audio file to target sample rate
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def save_audio(audio: np.ndarray, file_path: str, sr: int = 8000):
    #Save audio file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, audio, sr)

def calculate_rms(audio: np.ndarray) -> float:
    #Calculate RMS energy of audio signal
    return np.sqrt(np.mean(audio ** 2))

def add_noise_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    # Match noise length to clean speech
    if len(noise) < len(clean):
        # Repeat noise if shorter
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)[:len(clean)]
    else:
        # Randomly crop noise if longer
        start_idx = random.randint(0, len(noise) - len(clean))
        noise = noise[start_idx:start_idx + len(clean)]
    
    # Calculate RMS values
    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)
    
    # Scale noise to achieve target SNR
    # SNR(dB) = 20*log10(clean_rms / noise_rms)
    # noise_rms_target = clean_rms / (10^(SNR/20))
    if noise_rms > 0:
        target_noise_rms = clean_rms / (10 ** (snr_db / 20))
        scaling_factor = target_noise_rms / noise_rms
        noise_scaled = noise * scaling_factor
    else:
        noise_scaled = noise
    
    # Mix clean and scaled noise
    noisy = clean + noise_scaled
    
    return noisy

# LIBRISPEECH DATA LOADING

def get_librispeech_speakers(librispeech_root: str) -> Dict[str, List[str]]:

    print(" Scanning LibriSpeech dataset")
    speakers_files = {}
    
    for subset in ["train-clean-100", "dev-clean", "test-clean"]:
        subset_path = os.path.join(librispeech_root, subset)
        
        if not os.path.exists(subset_path):
            print(f"Warning: {subset} not found, skipping")
            continue
        
        print(f"Scanning {subset}")
        
        for speaker_id in os.listdir(subset_path):
            speaker_path = os.path.join(subset_path, speaker_id)
            
            if not os.path.isdir(speaker_path):
                continue
            
            if speaker_id not in speakers_files:
                speakers_files[speaker_id] = []
            
            # Recursively find all .flac files
            for root, dirs, files in os.walk(speaker_path):
                for file in files:
                    if file.endswith('.flac'):
                        speakers_files[speaker_id].append(os.path.join(root, file))
    
    # Filter out speakers with no files
    speakers_files = {k: v for k, v in speakers_files.items() if len(v) > 0}
    
    total_files = sum(len(files) for files in speakers_files.values())
    print(f"Found {len(speakers_files)} speakers with {total_files} total files\n")
    
    return speakers_files

def select_speakers_and_files(
    speakers_dict: Dict[str, List[str]],
    num_speakers: int,
    num_utterances: int,
    exclude_speakers: List[str] = None
) -> Tuple[List[str], List[str]]:
    if exclude_speakers is None:
        exclude_speakers = []
    
    # Each selected speaker must supply ceil(num_utterances / num_speakers) utterances,
    # so pre-filter to speakers that have at least that many files.
    min_files_per_speaker = (num_utterances + num_speakers - 1) // num_speakers
    available = {
        k: v for k, v in speakers_dict.items()
        if k not in exclude_speakers and len(v) >= min_files_per_speaker
    }
    
    if len(available) < num_speakers:
        raise ValueError(
            f"Not enough speakers with >= {min_files_per_speaker} utterances! "
            f"Need {num_speakers}, but only {len(available)} qualify after exclusions."
        )
    
    # Randomly select speakers
    selected_speakers = random.sample(list(available.keys()), num_speakers)
    
    # Calculate utterances per speaker (distribute evenly)
    base_count = num_utterances // num_speakers
    extra_count = num_utterances % num_speakers
    
    selected_files = []
    
    for i, speaker_id in enumerate(selected_speakers):
        # First 'extra_count' speakers get one extra utterance
        count = base_count + (1 if i < extra_count else 0)
        selected_files.extend(random.sample(available[speaker_id], count))
    
    return selected_speakers, selected_files

# DATASET CREATION

def create_mixed_dataset(
    clean_files: List[str],
    noise_files: List[str],
    output_dir: str,
    split_name: str,
    config: Config,
    mix_all_noises: bool = False
):
    clean_dir = os.path.join(output_dir, split_name, "clean")
    noisy_dir = os.path.join(output_dir, split_name, "noisy")
    # Remove stale files from any previous run with different utterance counts
    shutil.rmtree(clean_dir, ignore_errors=True)
    shutil.rmtree(noisy_dir, ignore_errors=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    
    # Load all noise files into memory
    print(f"  Loading {len(noise_files)} noise file(s)...")
    noises = {}
    for noise_path in noise_files:
        noise_name = os.path.basename(noise_path).replace('.wav', '')
        noises[noise_name] = load_audio(noise_path, config.TARGET_SR)
    
    # Determine SNR
    is_train = (split_name == "train")
    # Train: all 4 discrete SNR levels per utterance (×4 samples)
    # Val/test: one randomly chosen discrete SNR level per utterance (×1 sample)
    snr_levels = config.TRAIN_SNR_LEVELS if is_train else ["random_discrete"]
    
    # Calculate expected output count
    if mix_all_noises:
        expected_count = len(clean_files) * len(noises) * len(snr_levels)
        print(f"  Creating {len(clean_files)} utterances × {len(noises)} noises × {len(snr_levels)} SNRs = {expected_count} samples")
    else:
        expected_count = len(clean_files) * len(snr_levels)
        print(f"  Creating {len(clean_files)} utterances × {len(snr_levels)} SNR(s) = {expected_count} samples")
    
    metadata = []
    
    # Process each clean utterance
    for clean_path in tqdm(clean_files, desc=f"  {split_name}"):
        # Load clean audio
        clean_audio = load_audio(clean_path, config.TARGET_SR)
        
        # Extract identifiers
        filename = os.path.basename(clean_path).replace('.flac', '')
        speaker_id = os.path.basename(os.path.dirname(os.path.dirname(clean_path)))
        
        # Save clean audio once
        clean_filename = f"{speaker_id}_{filename}.wav"
        save_audio(clean_audio, os.path.join(clean_dir, clean_filename), config.TARGET_SR)
        
        # Determine which noises to use
        if mix_all_noises:
            # Use ALL noise types (Session 0 training only)
            noise_names = list(noises.keys())
        else:
            # Use ONE random noise (incremental training, all val/test)
            noise_names = [random.choice(list(noises.keys()))]
        
        # Create noisy versions
        for noise_name in noise_names:
            noise_audio = noises[noise_name]
            
            for snr in snr_levels:
                # Determine actual SNR value
                if snr == "random_discrete":  # Random discrete SNR for val/test
                    snr_value = random.choice(config.TRAIN_SNR_LEVELS)
                else:
                    snr_value = snr
                
                # Mix clean + noise at target SNR
                noisy_audio = add_noise_at_snr(clean_audio, noise_audio, snr_value)
                
                # Save noisy audio
                noisy_filename = f"{speaker_id}_{filename}_snr{snr_value:.1f}_{noise_name}.wav"
                save_audio(noisy_audio, os.path.join(noisy_dir, noisy_filename), config.TARGET_SR)
                
                # Record metadata
                metadata.append({
                    "clean_file": clean_filename,
                    "noisy_file": noisy_filename,
                    "speaker_id": speaker_id,
                    "utterance_id": filename,
                    "noise_type": noise_name,
                    "snr_db": float(snr_value),
                    "duration_sec": float(len(clean_audio) / config.TARGET_SR)
                })
    
    # Save metadata
    metadata_path = os.path.join(output_dir, split_name, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"{split_name} complete: {len(metadata)} noisy files + {len(clean_files)} clean files\n")

def create_session(
    session_id: int,
    session_name: str,
    noise_files: List[str],
    speakers_dict: Dict[str, List[str]],
    config: Config,
    exclude_speakers: List[str] = None
) -> List[str]:
    if exclude_speakers is None:
        exclude_speakers = []
    
    output_dir = os.path.join(config.OUTPUT_ROOT, session_name)
    
    print_header(f"SESSION {session_id}: {session_name.upper()}")
    
    # Get specifications
    is_pretrain = (session_id == 0)
    if is_pretrain:
        train_utt, train_spk = config.SESSION0_TRAIN_UTTERANCES, config.SESSION0_TRAIN_SPEAKERS
        val_utt, val_spk = config.SESSION0_VAL_UTTERANCES, config.SESSION0_VAL_SPEAKERS
        test_utt, test_spk = config.SESSION0_TEST_UTTERANCES, config.SESSION0_TEST_SPEAKERS
    else:
        train_utt, train_spk = config.INCREMENTAL_TRAIN_UTTERANCES, config.INCREMENTAL_TRAIN_SPEAKERS
        val_utt, val_spk = config.INCREMENTAL_VAL_UTTERANCES, config.INCREMENTAL_VAL_SPEAKERS
        test_utt, test_spk = config.INCREMENTAL_TEST_UTTERANCES, config.INCREMENTAL_TEST_SPEAKERS
    
    print(f"Noise files: {[os.path.basename(f) for f in noise_files]}")
    print(f"  Train: {train_utt} utterances, {train_spk} speakers")
    print(f"  Val:   {val_utt} utterances, {val_spk} speakers")
    print(f"  Test:  {test_utt} utterances, {test_spk} speakers\n")
    
    # Select speakers and files
    print("Selecting speakers and utterances.")
    
    train_speakers, train_files = select_speakers_and_files(
        speakers_dict, train_spk, train_utt, exclude_speakers
    )
    print(f"  Train: {len(train_speakers)} speakers, {len(train_files)} files")
    
    val_speakers, val_files = select_speakers_and_files(
        speakers_dict, val_spk, val_utt, exclude_speakers + train_speakers
    )
    print(f"  Val:   {len(val_speakers)} speakers, {len(val_files)} files")
    
    test_speakers, test_files = select_speakers_and_files(
        speakers_dict, test_spk, test_utt, exclude_speakers + train_speakers + val_speakers
    )
    print(f"Test:  {len(test_speakers)} speakers, {len(test_files)} files\n")
    
    # Create datasets
    print("Creating training dataset.")
    create_mixed_dataset(
        train_files, noise_files, output_dir, "train", config,
        mix_all_noises=is_pretrain  # Only Session 0 mixes with all noises
    )
    
    print("Creating validation dataset.")
    create_mixed_dataset(val_files, noise_files, output_dir, "val", config,
                         mix_all_noises=is_pretrain)
    
    print("Creating test dataset.")
    create_mixed_dataset(test_files, noise_files, output_dir, "test", config,
                         mix_all_noises=is_pretrain)
    
    # Save session summary
    summary = {
        "session_id": session_id,
        "session_name": session_name,
        "noise_types": [os.path.basename(f).replace('.wav', '') for f in noise_files],
        "train": {
            "num_speakers": len(train_speakers),
            "num_utterances": len(train_files),
            "num_noisy_files": len(train_files) * len(config.TRAIN_SNR_LEVELS) * (len(noise_files) if is_pretrain else 1),
            "speakers": train_speakers
        },
        "val": {
            "num_speakers": len(val_speakers),
            "num_utterances": len(val_files),
            "speakers": val_speakers
        },
        "test": {
            "num_speakers": len(test_speakers),
            "num_utterances": len(test_files),
            "speakers": test_speakers
        }
    }
    
    with open(os.path.join(output_dir, "session_info.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"{session_name} complete\n")
    
    return train_speakers + val_speakers + test_speakers


def main(skip_session0: bool = False):
    config = Config()
    set_seed(config.RANDOM_SEED)
    
    print_header("INCREMENTAL SPEECH ENHANCEMENT - DATA PREPARATION")
    
    print("Configuration:")
    print(f" Target sample rate: {config.TARGET_SR} Hz")
    print(f"Training SNRs: {config.TRAIN_SNR_LEVELS} dB")
    print(f"Val/Test SNR: random discrete from {config.TRAIN_SNR_LEVELS} dB")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"Output: {config.OUTPUT_ROOT}")
    if skip_session0:
        print("  --skip_session0: Session 0 will NOT be regenerated.\n")
    
    # Create output directory
    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
    
    # Load LibriSpeech data
    speakers_dict = get_librispeech_speakers(config.LIBRISPEECH_ROOT)
    
    # Track used speakers to avoid overlap
    session0_speakers = []
    
    if skip_session0:
        # Load session0 speakers from existing session_info.json
        s0_info_path = os.path.join(config.OUTPUT_ROOT, "session0_pretrain", "session_info.json")
        if not os.path.exists(s0_info_path):
            raise FileNotFoundError(
                f"Cannot skip session 0: {s0_info_path} not found. "
                "Run without --skip_session0 first."
            )
        with open(s0_info_path) as f:
            s0_info = json.load(f)
        session0_speakers = (
            s0_info["train"]["speakers"]
            + s0_info["val"]["speakers"]
            + s0_info["test"]["speakers"]
        )
        print(f"  Loaded {len(session0_speakers)} session 0 speakers from existing info.")
    else:
        # SESSION 0: Pre-training with 10 noises
        session0_noises = [
            os.path.join(config.PRETRAIN_NOISE_DIR, n) for n in config.SESSION0_NOISES
        ]
        missing = [f for f in session0_noises if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"Missing noise files: {missing}")
        
        session0_speakers = create_session(
            session_id=0,
            session_name="session0_pretrain",
            noise_files=session0_noises,
            speakers_dict=speakers_dict,
            config=config
        )
    
    # SESSIONS 1-4: Incremental learning
    
    for sess_info in config.INCREMENTAL_SESSIONS:
        noise_path = os.path.join(config.INCREMENTAL_NOISE_DIR, sess_info["noise"])
        
        if not os.path.exists(noise_path):
            raise FileNotFoundError(f"Missing noise file: {noise_path}")
        
        # Incremental sessions use DIFFERENT speakers than Session 0
        create_session(
            session_id=sess_info["id"],
            session_name=f"session{sess_info['id']}_incremental_{sess_info['name']}",
            noise_files=[noise_path],
            speakers_dict=speakers_dict,
            config=config,
            exclude_speakers=session0_speakers  # Exclude Session 0 speakers
        )
    
    # Create overall summary
    
    print_header("DATA PREPARATION COMPLETE")
    
    summary = {
        "config": {
            "sample_rate_hz": config.TARGET_SR,
            "train_snr_levels_db": config.TRAIN_SNR_LEVELS,
            "val_test_snr_range_db": list(config.VAL_TEST_SNR_RANGE),
            "random_seed": config.RANDOM_SEED
        },
        "sessions": {
            "session0": {
                "type": "pretrain",
                "num_noises": len(config.SESSION0_NOISES),
                "noise_types": [n.replace('.wav', '') for n in config.SESSION0_NOISES],
                "expected_train_samples": 40400
            }
        }
    }
    
    for sess_info in config.INCREMENTAL_SESSIONS:
        summary["sessions"][f"session{sess_info['id']}"] = {
            "type": "incremental",
            "noise_type": sess_info["name"],
            "expected_train_samples": 1212
        }
    
    summary_path = os.path.join(config.OUTPUT_ROOT, "dataset_info.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Dataset summary: {summary_path}")
    print(f"All data saved to: {config.OUTPUT_ROOT}")
    
    print("SUCCESS")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare speech enhancement data")
    parser.add_argument(
        "--skip_session0", action="store_true",
        help="Skip session 0 (pretrain) generation — useful when pretrain data already exists"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(skip_session0=args.skip_session0)