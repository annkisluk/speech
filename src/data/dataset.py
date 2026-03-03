import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Tuple, Optional, Dict, List
import random

from ..utils.audio import load_audio, pad_audio_batch


class SpeechEnhancementDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 8000,
        max_length: Optional[int] = None,
        normalize: bool = True,
        random_crop: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.normalize = normalize
        self.random_crop = random_crop
        
        # Paths to clean and noisy folders
        self.clean_dir = self.data_dir / split / "clean"
        self.noisy_dir = self.data_dir / split / "noisy"
        self.metadata_path = self.data_dir / split / "metadata.json"
        
        # Validate paths exist
        assert self.clean_dir.exists(), f"Clean dir not found: {self.clean_dir}"
        assert self.noisy_dir.exists(), f"Noisy dir not found: {self.noisy_dir}"
        assert self.metadata_path.exists(), f"Metadata not found: {self.metadata_path}"
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {split} dataset: {len(self.metadata)} samples from {data_dir}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Get metadata for this sample
        item = self.metadata[idx]
        
        # Construct file paths
        noisy_path = self.noisy_dir / item['noisy_file']
        clean_path = self.clean_dir / item['clean_file']
        
        # Load audio files
        noisy, _ = load_audio(str(noisy_path), self.sample_rate, self.normalize)
        clean, _ = load_audio(str(clean_path), self.sample_rate, self.normalize)
        
        # Ensure same length first
        min_len = min(noisy.shape[1], clean.shape[1])
        noisy = noisy[:, :min_len]
        clean = clean[:, :min_len]
        
        # Crop to max_length if specified
        if self.max_length is not None and min_len > self.max_length:
            if self.random_crop:
                # Random crop: pick a random start index (different each call)
                start = random.randint(0, min_len - self.max_length)
            else:
                # Deterministic: always take from the beginning
                start = 0
            noisy = noisy[:, start:start + self.max_length]
            clean = clean[:, start:start + self.max_length]
            min_len = self.max_length
        
        # Prepare info dict
        info = {
            'speaker_id': item['speaker_id'],
            'utterance_id': item['utterance_id'],
            'noise_type': item['noise_type'],
            'snr_db': item['snr_db'],
            'length': min_len
        }
        
        return noisy, clean, info
    
    def get_noise_types(self) -> List[str]:
        #Get list of unique noise types in this dataset
        noise_types = set(item['noise_type'] for item in self.metadata)
        return sorted(list(noise_types))
    
    def get_speaker_ids(self) -> List[str]:
        #Get list of unique speaker IDs in this dataset
        speaker_ids = set(item['speaker_id'] for item in self.metadata)
        return sorted(list(speaker_ids))


class MultiSessionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        session_ids: List[int],
        split: str = "test",
        sample_rate: int = 8000,
        normalize: bool = True
    ):

        self.data_root = Path(data_root)
        self.session_ids = session_ids
        self.split = split
        self.sample_rate = sample_rate
        self.normalize = normalize
        
        # Load all sessions
        self.datasets = []
        # Track which session each sample belongs to
        self.session_labels = []  
        
        for session_id in session_ids:
            if session_id == 0:
                session_dir = self.data_root / "session0_pretrain"
            else:
                # Find the incremental session directory
                session_dirs = list(self.data_root.glob(f"session{session_id}_incremental_*"))
                if not session_dirs:
                    raise ValueError(f"Session {session_id} not found in {data_root}")
                session_dir = session_dirs[0]
            
            # Create dataset for this session
            dataset = SpeechEnhancementDataset(
                data_dir=str(session_dir),
                split=split,
                sample_rate=sample_rate,
                normalize=normalize
            )
            
            self.datasets.append(dataset)
            # Track which session each sample belongs to
            self.session_labels.extend([session_id] * len(dataset))
        
        print(f"Multi-session dataset: {len(self)} total samples from sessions {session_ids}")
    
    def __len__(self) -> int:
        return sum(len(ds) for ds in self.datasets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Find which dataset this index belongs to
        cumulative = 0
        for dataset_idx, dataset in enumerate(self.datasets):
            if idx < cumulative + len(dataset):
                # This is the right dataset
                local_idx = idx - cumulative
                noisy, clean, info = dataset[local_idx]
                
                # Add session information
                info['session_id'] = self.session_ids[dataset_idx]
                
                return noisy, clean, info
            
            cumulative += len(dataset)
        
        raise IndexError(f"Index {idx} out of range")


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:

    noisy_list = []
    clean_list = []
    lengths = []
    info_list = []
    
    for noisy, clean, info in batch:
        noisy_list.append(noisy.squeeze(0))  
        clean_list.append(clean.squeeze(0))  
        lengths.append(noisy.shape[1])
        info_list.append(info)
    
    # Pad to max length in batch
    noisy_batch = pad_audio_batch(noisy_list)  
    clean_batch = pad_audio_batch(clean_list)  
    
    # Add channel dimension
    noisy_batch = noisy_batch.unsqueeze(1)  
    clean_batch = clean_batch.unsqueeze(1)  
    
    # Convert lengths to tensor
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return noisy_batch, clean_batch, lengths, info_list


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = False,
    prefetch_factor: int = 2
) -> DataLoader:
    use_persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,  # Keep all samples
        persistent_workers=use_persistent_workers,
        prefetch_factor=prefetch_factor if use_persistent_workers else None
    )


# Utility Functions

def get_session_dataloaders(
    data_root: str,
    session_id: int,
    batch_size_train: int = 2,
    batch_size_val: int = 4,
    batch_size_test: int = 4,
    num_workers: int = 2,
    pin_memory: bool = False,
    sample_rate: int = 8000
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # Determine session directory
    data_root = Path(data_root)
    if session_id == 0:
        session_dir = data_root / "session0_pretrain"
    else:
        session_dirs = list(data_root.glob(f"session{session_id}_incremental_*"))
        if not session_dirs:
            raise ValueError(f"Session {session_id} not found")
        session_dir = session_dirs[0]
    
    # Create datasets
    train_dataset = SpeechEnhancementDataset(
        data_dir=str(session_dir),
        split="train",
        sample_rate=sample_rate,
        max_length=4 * sample_rate,  # 4 seconds max (prevents OOM)
        random_crop=True  # Random segment each epoch
    )

    val_dataset = SpeechEnhancementDataset(
        data_dir=str(session_dir),
        split="val",
        sample_rate=sample_rate,
        max_length=4 * sample_rate,
        random_crop=False  # Deterministic for reproduciblity
    )

    test_dataset = SpeechEnhancementDataset(
        data_dir=str(session_dir),
        split="test",
        sample_rate=sample_rate,
        max_length=4 * sample_rate,
        random_crop=False  # Deterministic for reproduciblity
    )
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

