"""
Configuration Management for LNA Project

This module defines configuration classes following the paper's specifications.
Allows easy modification for experimental improvements.

"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class DataConfig:

    # Paths
    data_root: str = "data/final_data"
    
    # Audio parameters (Section II, III)
    sample_rate: int = 8000  # 8 kHz as per paper
    # Batch sizes 
    train_batch_size: int = 2
    val_batch_size: int = 2
    test_batch_size: int = 2
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class SepFormerConfig:
    """
    SepFormer backbone configuration
    Architecture:
    - Encoder: 1D Conv (512 filters, kernel=16, stride=8)
    - Masking Network: Multi-layer Transformer
      - N=8 layers
      - Multi-Head Attention (8 heads)
      - Feed-Forward Network (FFN)
    - Decoder: Transposed Conv
    """
    # Encoder parameters
    N: int = 256              # Number of filters in bottleneck
    L: int = 16               # Length of filters (samples)
    B: int = 256              # Number of channels in bottleneck and residual paths
    H: int = 512              # Number of channels in convolutional blocks
    P: int = 3                # Kernel size in convolutional blocks
    X: int = 8                # Number of convolutional blocks in each repeat
    R: int = 4                # Number of repeats
    
    # Transformer parameters
    num_layers: int = 8       # Transformer layers per direction per DPT block
    num_blocks: int = 2       # Number of DPT blocks (total layers = num_blocks * 2 * num_layers = 32)
    nhead: int = 8            # Paper: 8 attention heads
    d_ffn: int = 1024        # Feed-forward dimension
    dropout: float = 0.1
    
    # Activation
    activation: str = "relu"
    
    # Normalization
    norm_type: str = "gLN"    # global Layer Normalization
    
    # Causal mode
    causal: bool = False      # Non-causal for offline processing


@dataclass
class AdapterConfig:
    """
    Adapter module configuration
    
    Paper Reference: Section III.C - Learning Noise Adapters
    
    From paper:
    "Both adapters utilize common components, namely a down-projection 
    linear layer with parameters W*,d ∈ R^(C×Ĉ) and b* ∈ R^Ĉ, 
    as well as an up-projection linear layer with parameters W*,u ∈ R^(Ĉ×C)"
    
    Key parameter: Ĉ (bottleneck dimension)
    Paper tests: Ĉ = 1 (extreme bottleneck)
    """
    # Bottleneck dimension (Ĉ)
    bottleneck_dim: int = 1   
    
    # Adapter placement 
    use_ffl_adapter: bool = True   # Feed-Forward Layer Adapter
    use_mha_adapter: bool = True   # Multi-Head Attention Adapter
    
    # Activation function
    activation: str = "relu"       # Between down and up projection
    
    # Initialization
    init_scale: float = 0.01      # Small initialization for stability


@dataclass
class SelectorConfig:
    """
    Noise Selector configuration
    """
    # Selector type 
    selector_type: str = "kmeans"  # Options: "kmeans", "meanshift", "gmm"
    
    # K-Means parameters 
    n_clusters: int = 20          
    
    # Mean-Shift parameters 
    bandwidth: Optional[float] = None  # Auto-estimate if None
    
    # GMM parameters 
    n_components: int = 20
    covariance_type: str = "full"
    
    # Feature extraction
    use_mean_pooling: bool = True     
    feature_dim: int = 256            
    
    # Clustering algorithm parameters
    random_state: int = 42
    max_iter: int = 300


@dataclass
class TrainingConfig:
    """
    Training configuration
    - Loss: SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
    - Optimizer: Adam
    - Learning rate: 15e-5
    - Batch size: 2
    - Pre-train epochs: 40
    - Incremental epochs: 20
    """
    # Loss function
    loss_type: str = "si_snr"     
    
    # Optimizer parameters 
    optimizer: str = "adam"
    learning_rate: float = 15e-5  
    weight_decay: float = 0.0
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "plateau"  # Reduce on plateau
    patience: int = 3
    factor: float = 0.5
    
    # Training epochs 
    pretrain_epochs: int = 40 #change to 40   
    incremental_epochs: int = 20 #change to 20 
    
    # Gradient clipping
    max_grad_norm: float = 5.0
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Checkpointing
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    
    # Validation
    validate_every_n_epochs: int = 1
    
    # Device
    device: str = "cuda"  # Options: "cuda", "cpu", "mps"
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Logging
    log_every_n_steps: int = 100
    use_tensorboard: bool = True
    use_wandb: bool = False


@dataclass
class IncrementalConfig:
    """
    Incremental Learning specific configuration
    Key principle: Freeze pre-trained model and train separate 
    domain-specific adapters and decoders
    """
    # What to freeze during incremental learning
    freeze_backbone: bool = True             # Freeze entire backbone (encoder + masking)
    freeze_encoder: bool = True      
    freeze_masking_network: bool = True  
    freeze_previous_adapters: bool = True  
    freeze_previous_decoders: bool = True  
    
    # What to train in each incremental session
    train_new_adapter: bool = True       
    train_new_decoder: bool = True       
    
    # Adapter management
    create_new_adapter_per_session: bool = True
    create_new_decoder_per_session: bool = True


@dataclass
class EvaluationConfig:
    """
    Evaluation metrics configuration
    - SI-SNR (Scale-Invariant SNR)
    - SDR (Signal-to-Distortion Ratio) 
    - PESQ (Perceptual Evaluation of Speech Quality)"
    """
    metrics: List[str] = field(default_factory=lambda: ["si_snr", "sdr", "pesq"])
    
    # PESQ parameters
    pesq_mode: str = "nb"  # "nb" (narrowband) for 8kHz, "wb" for 16kHz
    
    # Evaluation on cumulative test sets
    evaluate_on_cumulative_testsets: bool = True


@dataclass
class ProjectConfig:
    """
    Master configuration combining all components
    """
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    sepformer: SepFormerConfig = field(default_factory=SepFormerConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    incremental: IncrementalConfig = field(default_factory=IncrementalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Project metadata
    project_name: str = "lna_speech_enhancement"
    experiment_name: str = "baseline"
    seed: int = 42
    
    # Output paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ProjectConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def __post_init__(self):
        """Create output directories"""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


def get_default_config() -> ProjectConfig:
    """
    Returns default configuration following paper specifications
    """
    return ProjectConfig()


def get_experiment_config(experiment_name: str) -> ProjectConfig:
    """
    Get configuration for specific experiments
    
    Args:
        experiment_name: One of ["baseline", "larger_adapter", "meanshift_selector"]
    
    Returns:
        Configured ProjectConfig object
    """
    config = get_default_config()
    config.experiment_name = experiment_name
    
    if experiment_name == "baseline":
        # Paper's exact configuration
        config.adapter.bottleneck_dim = 1
        config.selector.selector_type = "kmeans"
        config.selector.n_clusters = 20
        
    elif experiment_name == "larger_adapter":
        # Experiment: Larger adapter capacity
        config.adapter.bottleneck_dim = 16
        
    elif experiment_name == "meanshift_selector":
        # Improvement: Mean-Shift instead of K-Means
        config.selector.selector_type = "meanshift"
        config.selector.bandwidth = None  # Auto-estimate
        
    elif experiment_name == "gmm_selector":
        # Improvement: GMM clustering
        config.selector.selector_type = "gmm"
        config.selector.n_components = 20
    
    return config


if __name__ == "__main__":
    # Demo: Create and save configurations
    
    # Baseline config (paper's method)
    baseline_config = get_experiment_config("baseline")
    baseline_config.to_yaml("configs/baseline.yaml")
    print("Created configs/baseline.yaml")
    
    # Experimental configs
    larger_adapter_config = get_experiment_config("larger_adapter")
    larger_adapter_config.to_yaml("configs/larger_adapter.yaml")
    print("Created configs/larger_adapter.yaml")
    
    meanshift_config = get_experiment_config("meanshift_selector")
    meanshift_config.to_yaml("configs/meanshift_selector.yaml")
    print("Created configs/meanshift_selector.yaml")
    
    print("\nConfiguration system ready!")
    print("Modify configs in config.py or load from YAML files")