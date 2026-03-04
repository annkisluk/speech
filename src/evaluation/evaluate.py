import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List
from tqdm import tqdm

from ..models.lna_model import LNAModel
from ..data.dataset import MultiSessionDataset, SpeechEnhancementDataset, create_dataloader
from ..evaluation.metrics import MetricsCalculator
from ..selectors.noise_selector import create_selector
from ..utils.config import ProjectConfig, get_default_config


def _forward_with_chunking(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    session_id: int,
    chunk_size: int = 32000,
    chunk_overlap: int = 8000
) -> torch.Tensor:

    if noisy.shape[0] != 1:
        return model(noisy, session_id=session_id)
    
    total_len = noisy.shape[-1]
    if total_len <= chunk_size:
        return model(noisy, session_id=session_id)
    
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    
    device = noisy.device
    # Use Hann window for smooth blending (prevents clicks in PESQ)
    window = torch.hann_window(chunk_size, device=device)
    window = window.view(1, 1, -1)
    
    out_len = ((total_len - 1) // step) * step + chunk_size
    acc = torch.zeros((1, 1, out_len), device=device)
    weight = torch.zeros((1, 1, out_len), device=device)
    
    for start in range(0, out_len, step):
        if start >= total_len:
            break
        end = start + chunk_size
        chunk = noisy[..., start:min(end, total_len)]
        if chunk.shape[-1] < chunk_size:
            pad_len = chunk_size - chunk.shape[-1]
            chunk = F.pad(chunk, (0, pad_len))
        
        enhanced_chunk = model(chunk, session_id=session_id)
        
        acc[..., start:end] += enhanced_chunk * window
        weight[..., start:end] += window
    
    enhanced = acc / weight.clamp_min(1e-8)
    return enhanced[..., :total_len]


def evaluate_model_on_session(
    model: LNAModel,
    test_loader,
    session_id: int,  # if None, uses per-sample info['session_id'] for routing
    device: str = 'cuda',
    selector = None,
    use_chunking: bool = True,
    chunk_size: int = 32000,
    chunk_overlap: int = 8000
) -> Dict[str, float]:

    model.eval()
    model.to(device)
    
    metrics_calc = MetricsCalculator(
        sample_rate=8000,
        metrics=['si_snr', 'sdr', 'pesq']
    )
    
    all_metrics = []
    selector_correct = 0
    selector_total = 0
    
    desc = f"Session {session_id}" if session_id is not None else "Cumulative"
    
    with torch.no_grad():
        for noisy, clean, lengths, info in tqdm(test_loader, desc=desc):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # If using selector, predict domain
            if selector:
                features = model.get_encoder_features(noisy)
                B_feat, N_feat, L_feat = features.shape
                enc_lengths = (lengths - 16) // 8 + 1  # encoder stride=8, kernel=16, no padding
                
                for i in range(len(noisy)):
                    real_len = min(enc_lengths[i].item(), L_feat)
                    feat_pooled = features[i, :, :real_len].mean(dim=1).cpu().numpy()
                    predicted_session = selector.predict(feat_pooled)
                    true_session = info[i].get('session_id', session_id)
                    
                    # Track selector accuracy
                    if predicted_session == true_session:
                        selector_correct += 1
                    selector_total += 1
                    
                    # Use predicted session's decoder with chunking if needed
                    if use_chunking:
                        enhanced = _forward_with_chunking(
                            model, noisy[i:i+1], predicted_session,
                            chunk_size, chunk_overlap
                        )
                    else:
                        enhanced = model(noisy[i:i+1], session_id=predicted_session)
                    
                    # Trim to actual length (remove batch padding)
                    L = lengths[i].item()
                    enhanced_trimmed = enhanced[0, 0, :L]  # [T]
                    clean_trimmed = clean[i, 0, :L]        # [T]
                    
                    sample_metrics = metrics_calc.calculate_all(
                        enhanced_trimmed, clean_trimmed
                    )
                    all_metrics.append(sample_metrics)
            else:
                # Use ground-truth per-sample session_id if session_id is None,
                # else use the fixed session_id for all samples
                for i in range(len(noisy)):
                    if session_id is None:
                        sid = info[i].get('session_id', 0)
                    else:
                        sid = session_id
                    
                    if use_chunking:
                        enhanced = _forward_with_chunking(
                            model, noisy[i:i+1], sid,
                            chunk_size, chunk_overlap
                        )
                    else:
                        enhanced = model(noisy[i:i+1], session_id=sid)
                    
                    # Trim to actual length (remove batch padding)
                    L = lengths[i].item()
                    enhanced_trimmed = enhanced[0, 0, :L]  # [T]
                    clean_trimmed = clean[i, 0, :L]        # [T]
                    
                    sample_metrics = metrics_calc.calculate_all(
                        enhanced_trimmed, clean_trimmed
                    )
                    all_metrics.append(sample_metrics)
    
    # Aggregate metrics
    aggregated = metrics_calc.aggregate_metrics(all_metrics)
    
    if selector:
        aggregated['selector_accuracy'] = 100 * selector_correct / selector_total
    
    return aggregated


def evaluate_cumulative(
    checkpoint_dir: str,
    data_root: str,
    session_ids: List[int],
    config: ProjectConfig,
    output_path: str = None
) -> Dict:
    """
    Evaluate after each incremental session t using the checkpoint Θ_t and
    the selector trained up to session t.  Per session t we test domains 1..t
    separately (651 samples each) and average, matching Table II of the paper.

    checkpoint_dir  – base checkpoints folder, e.g. "checkpoints/"
                      Session t model  : <checkpoint_dir>/session{t}_incremental/lna_session{t}.pt
                      Session t selector: <checkpoint_dir>/session{t}_incremental/selector_upto_session{t}.pkl
    """

    device = config.training.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    print("CUMULATIVE EVALUATION")
    
    # Evaluate after each session: test each seen domain separately (651 each),
    # then average across domains — matching Table II of the paper.
    results = {}

    for session_id in session_ids:
        print(f"\n{'='*80}")
        print(f"Evaluating after Session {session_id} (domains 1..{session_id}, 651 each)".center(80))
        print(f"{'='*80}\n")

        # Load the checkpoint that corresponds to *this* session (Θ_t, not Θ_final)
        sess_model_path = Path(checkpoint_dir) / f"session{session_id}_incremental" / f"lna_session{session_id}.pt"
        sess_selector_path = Path(checkpoint_dir) / f"session{session_id}_incremental" / f"selector_upto_session{session_id}.pkl"

        if not sess_model_path.exists():
            print(f"  WARNING: model checkpoint not found at {sess_model_path} — skipping session {session_id}")
            continue
        if not sess_selector_path.exists():
            print(f"  WARNING: selector not found at {sess_selector_path} — skipping session {session_id}")
            continue

        print(f"  Loading model   : {sess_model_path}")
        model = LNAModel(
            n_basis=config.sepformer.N,
            kernel_size=config.sepformer.L,
            num_layers=config.sepformer.num_layers,
            num_blocks=config.sepformer.num_blocks,
            nhead=config.sepformer.nhead,
            dim_feedforward=config.sepformer.d_ffn,
            dropout=config.sepformer.dropout,
            adapter_bottleneck_dim=config.adapter.bottleneck_dim,
            max_sessions=6
        )
        # Add adapters only for sessions seen up to session_id (Θ_t has 1..t)
        for sid in range(1, session_id + 1):
            model.add_new_session(session_id=sid, bottleneck_dim=config.adapter.bottleneck_dim)
        model.load_checkpoint(str(sess_model_path))

        print(f"  Loading selector: {sess_selector_path}")
        selector = create_selector(
            selector_type=config.selector.selector_type,
            feature_dim=config.sepformer.N
        )
        selector.load(str(sess_selector_path))

        domain_results = {}

        # Each domain: sessions 1..session_id only (session 0 is pre-train, not evaluated)
        for domain_id in range(1, session_id + 1):
            # Each domain: its own 651-sample test set
            if domain_id == 0:
                domain_dir = Path(data_root) / "session0_pretrain"
            else:
                domain_dirs = list(Path(data_root).glob(f"session{domain_id}_incremental_*"))
                if not domain_dirs:
                    print(f"  Warning: domain {domain_id} not found, skipping")
                    continue
                domain_dir = domain_dirs[0]

            domain_dataset = SpeechEnhancementDataset(
                data_dir=str(domain_dir),
                split="test",
                sample_rate=8000
            )
            domain_loader = create_dataloader(
                domain_dataset,
                batch_size=config.data.test_batch_size,
                shuffle=False,
                num_workers=config.data.num_workers
            )

            print(f"  Domain {domain_id} ({domain_dir.name}, {len(domain_dataset)} samples):")
            domain_metrics = evaluate_model_on_session(
                model=model,
                test_loader=domain_loader,
                session_id=domain_id,  # fixed routing: use this domain's adapter
                device=device,
                selector=selector
            )
            domain_results[f'domain_{domain_id}'] = domain_metrics
            for metric, value in domain_metrics.items():
                print(f"    {metric}: {value:.4f}")

        # Average metrics across all seen domains (equal weight, 651 each)
        all_keys = [k for k in next(iter(domain_results.values())).keys()]
        avg_metrics = {
            k: float(np.mean([domain_results[d][k] for d in domain_results if k in domain_results[d]]))
            for k in all_keys
        }

        results[f'session_{session_id}'] = {
            'per_domain': domain_results,
            'average': avg_metrics
        }

        print(f"\n  --- Session {session_id} Average across {len(domain_results)} domains ---")
        for metric, value in avg_metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    return results


def compare_methods(
    results_dir: str,
    methods: List[str] = ['baseline', 'lna']
) -> Dict:

    print("METHOD COMPARISON:")
    
    results_dir = Path(results_dir)
    comparison = {}
    
    for method in methods:
        method_file = results_dir / f"{method}_results.json"
        if method_file.exists():
            with open(method_file, 'r') as f:
                comparison[method] = json.load(f)
        else:
            print(f"Warning: {method_file} not found")
    
    # Print comparison table
    if comparison:
        print("\nComparison (SI-SNR in dB):")
        print(f"{'Session':<15} {'Baseline':<15} {'LNA':<15} {'Improvement':<15}")
        
        for session_key in comparison[methods[0]].keys():
            session_id = session_key.split('_')[1]
            
            baseline_snr = comparison.get(methods[0], {}).get(session_key, {}).get('si_snr_mean', 0)
            lna_snr = comparison.get(methods[1], {}).get(session_key, {}).get('si_snr_mean', 0)
            improvement = lna_snr - baseline_snr
            
            print(f"{session_id:<15} {baseline_snr:<15.2f} {lna_snr:<15.2f} {improvement:<15.2f}")
        
        print("-" * 60)
    
    return comparison


def main():
    """Main function for CLI"""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Base checkpoints directory (e.g. checkpoints/). Per-session models are resolved automatically."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/final_data",
        help="Root directory with session data"
    )
    parser.add_argument(
        "--sessions",
        type=int,
        nargs='+',
        default=[1, 2, 3, 4],
        help="Session IDs to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = ProjectConfig.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Override device
    if args.device:
        config.training.device = args.device
    
    # Evaluate
    results = evaluate_cumulative(
        checkpoint_dir=args.checkpoint_dir,
        data_root=args.data_root,
        session_ids=args.sessions,
        config=config,
        output_path=args.output
    )
    
    print("\n Evaluation complete")


if __name__ == "__main__":
    main()