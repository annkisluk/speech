# Learning Noise Adapters for Incremental Speech Enhancement

A PyTorch reimplementation of the paper:

> **Learning Noise Adapters for Incremental Speech Enhancement**
> Yang et al., IEEE Signal Processing Letters, Vol. 31, 2024

## Pretrained Checkpoints

Model weights and selectors are hosted on Hugging Face (too large for GitHub):

**https://huggingface.co/Annkisluk/lna-speech**

| File | Description |
|------|-------------|
| `session0_pretrain/lna_pretrained.pt` | Pretrained LNA backbone (40 epochs, 10 NOISEX-92 noise types) |
| `session1_incremental/lna_session1.pt` | After incremental session 1 (Alarm) |
| `session1_incremental/selector_upto_session1.pkl` | K-Means selector for sessions 1 |
| `session2_incremental/lna_session2.pt` | After incremental session 2 (Cough) |
| `session2_incremental/selector_upto_session2.pkl` | K-Means selector for sessions 1–2 |
| `session3_incremental/lna_session3.pt` | After incremental session 3 (Destroyer Ops) |
| `session3_incremental/selector_upto_session3.pkl` | K-Means selector for sessions 1–3 |
| `session4_incremental/lna_session4.pt` | After incremental session 4 (Machine Gun) |
| `session4_incremental/selector_upto_session4.pkl` | K-Means selector for sessions 1–4 |

Download with:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="Annkisluk/lna-speech",
                       filename="session4_incremental/lna_session4.pt")
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

Or evaluate only (using saved checkpoints):
```bash
python src/evaluation/evaluate.py \
  --checkpoint_dir checkpoints/ \
  --data_root data/final_data \
  --sessions 1 2 3 4 \
  --output results/final_results.json
```

## Data Protocol

| Session | Split | Utterances | × Noises | × SNR | = Samples |
|---------|-------|-----------|----------|-------|-----------|
| 0 (pretrain) | train | 1,010 | × 10 | × 4 | 40,400 |
| 0 (pretrain) | val   | 1,206 | × 10 | × 1 | 12,060 |
| 0 (pretrain) | test  | 651   | × 10 | × 1 | 6,510  |
| 1–4 (incremental) | train | 303 | × 1 | × 4 | 1,212 |
| 1–4 (incremental) | val   | 1,206 | × 1 | × 1 | 1,206 |
| 1–4 (incremental) | test  | 651   | × 1 | × 1 | 651   |

SNR levels: {−5, 0, 5, 10} dB. Sample rate: 8 kHz. Speech: LibriSpeech train-clean-100.
Val/test use one randomly-drawn discrete SNR per sample. Incremental val uses the current session's noise only (single-domain).

## Results (LibriSpeech, 8 kHz)

Evaluation protocol: after session t, test domains 1..t separately (651 samples each) using checkpoint Θ_t, then average.

### Per-domain SI-SNR (dB) after each session

| After session | Alarm | Cough | Destr. Ops | Machine Gun | Avg SI-SNR | Avg Selector Acc |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 16.31 | — | — | — | 16.31 | 100.0% |
| 2 | 14.46 | 14.05 | — | — | 14.26 | 82.9% |
| 3 | 13.64 | 11.22 | 11.86 | — | 12.24 | 67.3% |
| 4 | 12.55 | 9.41 | 11.47 | 14.67 | **12.02** | 59.0% |

Paper (WSJ0) reports LNA avg 16.46 dB. The ~4 dB gap is expected: LibriSpeech is far more acoustically diverse than WSJ0, making the K-Means noise selector harder to train and reducing routing accuracy.

## Architecture

- **Backbone**: SepFormer-style encoder-masking-decoder (25.61M params)
  - Encoder: `Conv1d(1, 256, k=16, s=8)` → ReLU
  - Masking network: 2 × DualPathTransformer blocks (8 intra + 8 inter layers each = 32 total)
  - Decoder: `ConvTranspose1d(256, 1, k=16, s=8)`
- **Adapters**: Ĉ=1 bottleneck, parallel MHA + FFL adapters in all 32 transformer layers
  - ~513 params/adapter × 64 adapters/session ≈ 32,832 params per session (0.128% of backbone)
- **Selector**: K-Means (k=20) on mean-pooled encoder features

## Key Implementation Notes

1. **Random crop**: Training uses random 4s segments from variable-length audio files. Deterministic cropping causes the model to memorize fixed segments and plateau early.
2. **Constant LR**: Adam lr=1.5×10⁻⁴ with no scheduler, matching the paper. ReduceLROnPlateau traps the model in local minima.
3. **No early stopping**: Exactly 40 pretrain epochs + 20 incremental epochs per session.
4. **Discrete SNR**: Val/test SNR drawn i.i.d. from {−5, 0, 5, 10} dB (not continuous uniform).
5. **Single-domain val during incremental training**: Validation uses only the current session's noise domain (not all seen domains), to avoid reusing old data.
6. **Per-session evaluation**: `evaluate_cumulative` loads checkpoint Θ_t (not the final Θ_4) when evaluating after session t, correctly measuring continual learning performance.
