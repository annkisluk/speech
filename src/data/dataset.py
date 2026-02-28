"""
PyTorch Dataset Classes for Speech Enhancement

Handles loading of noisy-clean speech pairs for training and evaluation.

Paper Reference: Section IV.A - Data organization with train/val/test splits
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Tuple, Optional, Dict, List
import random

from ..utils.audio import load_audio, pad_audio_batch


class SpeechEnhancementDataset(Dataset):
    """
    Dataset for speech enhancement training
    
    Paper structure:
    - Each split (train/val/test) has:
      - clean/ folder with clean speech
      - noisy/ folder with noisy speech
      - metadata.json with pairing information
    
    Usage:
        dataset = SpeechEnhancementDataset(
            data_dir="data/final_data/session0_pretrain",
            split="train",
            sample_rate=8000
        )
        
        noisy, clean, info = dataset[0]
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 8000,
        max_length: Optional[int] = None,
        normalize: bool = True
    ):
        """
        Args:
            data_dir: Path to session directory (e.g., "data/final_data/session0_pretrain")
            split: One of ["train", "val", "test"]
            sample_rate: Target sample rate 
            max_length: Maximum audio length in samples (None = no limit)
            normalize: Whether to normalize audio to [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.normalize = normalize
        
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
        """
        Get a noisy-clean speech pair
        
        Returns:
            noisy: Noisy speech tensor [1, T]
            clean: Clean speech tensor [1, T]
            info: Dictionary with metadata
        """
        # Get metadata for this sample
        item = self.metadata[idx]
        
        # Construct file paths
        noisy_path = self.noisy_dir / item['noisy_file']
        clean_path = self.clean_dir / item['clean_file']
        
        # Load audio files
        noisy, _ = load_audio(str(noisy_path), self.sample_rate, self.normalize)
        clean, _ = load_audio(str(clean_path), self.sample_rate, self.normalize)
        
        # Trim to max_length if specified
        if self.max_length is not None:
            noisy = noisy[:, :self.max_length]
            clean = clean[:, :self.max_length]
        
        # Ensure same length 
        min_len = min(noisy.shape[1], clean.shape[1])
        noisy = noisy[:, :min_len]
        clean = clean[:, :min_len]
        
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
        """Get list of unique noise types in this dataset"""
        noise_types = set(item['noise_type'] for item in self.metadata)
        return sorted(list(noise_types))
    
    def get_speaker_ids(self) -> List[str]:
        """Get list of unique speaker IDs in this dataset"""
        speaker_ids = set(item['speaker_id'] for item in self.metadata)
        return sorted(list(speaker_ids))


class MultiSessionDataset(Dataset):
    """
    Dataset that combines multiple sessions for evaluation
    
    Usage:
        # Evaluate on cumulative test sets after Session 2
        dataset = MultiSessionDataset(
            data_root="data/final_data",
            session_ids=[1, 2],  # Sessions 1 and 2
            split="test"
        )
    """
    
    def __init__(
        self,
        data_root: str,
        session_ids: List[int],
        split: str = "test",
        sample_rate: int = 8000,
        normalize: bool = True
    ):
        """
        Args:
            data_root: Root directory containing all sessions
            session_ids: List of session IDs to include 
            split: One of ["train", "val", "test"]
            sample_rate: Target sample rate
            normalize: Whether to normalize audio
        """
        self.data_root = Path(data_root)
        self.session_ids = session_ids
        self.split = split
        self.sample_rate = sample_rate
        self.normalize = normalize
        
        # Load all sessions
        self.datasets = []
        self.session_labels = []  # Track which session each sample belongs to
        
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
        """
        Get a sample from the combined dataset
        
        Returns:
            noisy, clean, info (info includes 'session_id')
        """
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
    """
    Custom collate function for DataLoader
    
    Handles variable-length audio by padding to the longest in the batch
    
    Args:
        batch: List of (noisy, clean, info) tuples
    
    Returns:
        noisy_batch: Padded noisy audio [B, 1, T_max]
        clean_batch: Padded clean audio [B, 1, T_max]
        lengths: Original lengths before padding [B]
        info_list: List of info dicts
    """
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
    """
    Create DataLoader with appropriate settings
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
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
    """
    Create train/val/test dataloaders for a single session
    
    Args:
        data_root: Root data directory
        session_id: Session ID (0 for pretrain, 1-5 for incremental)
        batch_size_train: Training batch size 
        batch_size_val: Validation batch size
        batch_size_test: Test batch size
        num_workers: Number of data loading workers
        sample_rate: Audio sample rate 
    
    Returns:
        train_loader, val_loader, test_loader
    """
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
        max_length=4 * sample_rate  # 4 seconds max — prevents OOM with large models
    )

    val_dataset = SpeechEnhancementDataset(
        data_dir=str(session_dir),
        split="val",
        sample_rate=sample_rate,
        max_length=4 * sample_rate
    )

    test_dataset = SpeechEnhancementDataset(
        data_dir=str(session_dir),
        split="test",
        sample_rate=sample_rate,
        max_length=4 * sample_rate
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



# Demo and Testing

if __name__ == "__main__":
    print("Testing dataset classes...")
    
    # Test single session dataset
    print("\n1. Testing single session dataset:")
    try:
        dataset = SpeechEnhancementDataset(
            data_dir="data/final_data/session0_pretrain",
            split="train",
            sample_rate=8000
        )
        
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Noise types: {dataset.get_noise_types()}")
        print(f"   Number of speakers: {len(dataset.get_speaker_ids())}")
        
        # Get one sample
        noisy, clean, info = dataset[0]
        print(f"\n   Sample 0:")
        print(f"     Noisy shape: {noisy.shape}")
        print(f"     Clean shape: {clean.shape}")
        print(f"     Info: {info}")
        
    except Exception as e:
        print(f"   Could not load dataset: {e}")
        print(f"   (This is expected if data hasn't been prepared yet)")
    
    # Test dataloader
    print("\n2. Testing dataloader with collate_fn:")
    try:
        loader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # Use 0 for testing
        )
        
        batch = next(iter(loader))
        noisy_batch, clean_batch, lengths, info_list = batch
        
        print(f"   Batch shapes:")
        print(f"     Noisy: {noisy_batch.shape}")
        print(f"     Clean: {clean_batch.shape}")
        print(f"     Lengths: {lengths}")
        print(f"     Batch size: {len(info_list)}")
        
    except Exception as e:
        print(f"   Could not create dataloader: {e}")
    
    print("\n✓ Dataset classes ready!")