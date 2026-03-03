# Learning Noise Adapters for Incremental Speech Enhancement

A PyTorch reimplementation of the paper:

> **Learning Noise Adapters for Incremental Speech Enhancement**
> Yang et al., IEEE Signal Processing Letters, Vol. 31, 2024

## Pretrained Checkpoints

Model weights are hosted on Hugging Face (too large for GitHub):

**https://huggingface.co/Annkisluk/lna-speech**

| File | Description |
|------|-------------|
| `session0_pretrain/best_model.pt` | Pretrained backbone (40 epochs, best val SI-SNR 13.62 dB) |
| `session1_incremental/best_model.pt` | After session 1 (Alarm) |
| `session2_incremental/best_model.pt` | After session 2 (Cough) |
| `session3_incremental/best_model.pt` | After session 3 (Destroyer Ops) |
| `session4_incremental/best_model.pt` | After session 4 (Machine Gun) |

Download with:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="Annkisluk/lna-speech", filename="session4_incremental/best_model.pt")
```

## Project Structure

```
set_up_data.sh          # Download & prepare raw data (LibriSpeech + NOISEX-92 + ESC-50)
prepare_data.py         # Build session datasets from raw data
run_pipeline.py         # Main entry point: pretrain / incremental / evaluate
src/
  models/               # LNAModel, SepFormer encoder/decoder, Adapters
  training/             # Pretraining and incremental training loops
  data/                 # Dataset classes and dataloaders
  selectors/            # K-Means noise selector
  evaluation/           # Metrics (SI-SNR, SDR, PESQ) and evaluation loop
  utils/                # Audio I/O, config dataclasses
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare data
bash set_up_data.sh
python prepare_data.py --mode pretrain
python prepare_data.py --mode incremental

# 3. Run pipeline
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_pipeline.py --mode pretrain
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_pipeline.py --mode incremental
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_pipeline.py --mode evaluate
```

## Results (Session 4, LibriSpeech)

| Method | Alarm | Cough | DD | MG | Avg SI-SNR |
|--------|-------|-------|----|----|------------|
| Pre-trained (frozen) | 11.xx | 7.xx | 13.xx | 14.xx | ~11.5 dB |
| FT-seq | degrades on old | degrades on old | degrades on old | best | ~9.5 dB |
| **LNA (ours)** | **14.44** | **9.31** | **11.17** | **14.38** | **12.33 dB** |

Paper reports LNA avg 16.46 dB on WSJ0. The ~4 dB gap is attributed to using LibriSpeech (diverse, variable-quality recordings) vs the paper's WSJ0 (clean, narrow-domain read speech), which makes noise feature clustering harder for the K-Means selector.

## Architecture

- **Backbone**: SepFormer-style encoder-masking-decoder (25.61M params)
  - Encoder: `Conv1d(1, 256, k=16, s=8)` → ReLU
  - Masking network: 2 × DualPathTransformer blocks (8 intra + 8 inter layers each = 32 total)
  - Decoder: `ConvTranspose1d(256, 1, k=16, s=8)`
- **Adapters**: Ĉ=1 bottleneck, parallel MHA + FFL adapters in all 32 transformer layers
  - 513 params/adapter × 64 adapters/session = 32,832 params per session (0.128% of backbone)
- **Selector**: K-Means (k=20) on mean-pooled encoder features

## Key Training Fixes (vs naive reimplementation)

1. **Random crop**: Training uses random 4s segments from long audio files (median 13s). Deterministic cropping caused the model to memorize fixed segments and plateau at epoch 9.
2. **Constant LR**: Paper uses Adam lr=15e-5 with no scheduler. ReduceLROnPlateau was trapping the model in local minima.
3. **No early stopping**: Paper trains for exactly 40 pretrain + 20 incremental epochs.
