#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.pretrain import train_pretrain
from src.training.incremental import train_all_incremental_sessions, fit_all_selectors
from src.evaluation.evaluate import evaluate_cumulative
from src.utils.config import get_default_config


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoint_paths = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_paths:
        return None

    def _epoch_num(path: Path) -> int:
        stem = path.stem
        return int(stem.split("checkpoint_epoch_")[-1])

    return max(checkpoint_paths, key=_epoch_num)


def run_pipeline(
    mode: str = "all",
    data_root: str = "data/final_data",
    device: str = "cuda"
):
    # Load configuration
    config = get_default_config()
    config.training.device = device
    
    print("LNA TRAINING PIPELINE")
    
    print(f"Mode: {mode}")
    print(f"Data root: {data_root}")
    print(f"Device: {device}")
    print()
    
    # Paths
    pretrained_model_path = Path("checkpoints/session0_pretrain/lna_pretrained.pt")
    final_model_path = Path("checkpoints/session4_incremental/lna_session4.pt")
    final_selector_path = Path("checkpoints/session4_incremental/selector_upto_session4.pkl")
    results_path = Path("results/final_results.json")
    
    # Step 1: Pre-training (Session 0)
    if mode in ["all", "pretrain"]:
        print("# STEP 1: PRE-TRAINING (SESSION 0)".center(78) + " #")
        
        resume_pretrain = _find_latest_checkpoint(
            Path(config.checkpoint_dir) / "session0_pretrain"
        )
        if resume_pretrain:
            print(f"Resuming pre-training from: {resume_pretrain}")

        history, metrics = train_pretrain(
            config=config,
            data_root=data_root,
            resume_from=str(resume_pretrain) if resume_pretrain else None
        )
        
        print(f"\n Pre-training complete!")
        print(f" Model saved: {pretrained_model_path}")
    
    # Step 2: Incremental Learning (Sessions 1-5)
    if mode in ["all", "incremental"]:
        print("# STEP 2: INCREMENTAL LEARNING (SESSIONS 1-4)".center(78) + " #")
        
        if not pretrained_model_path.exists():
            print(f"Error: Pre-trained model not found: {pretrained_model_path}")
            print("run pretraining first")
            return
        
        results = train_all_incremental_sessions(
            config=config,
            pretrained_model_path=str(pretrained_model_path),
            data_root=data_root,
            session_ids=[1, 2, 3, 4],
            resume_if_exists=True
        )
        
        print(f"\n Incremental training complete")
        print(f"  Final model: {final_model_path}")
        print(f"  Final selector: {final_selector_path}")
    
    # Step 2b: Fit selectors only (when model checkpoints exist but selector .pkl files are missing)
    if mode in ["selectors"]:
        print("# STEP 2b: FIT SELECTORS FROM EXISTING CHECKPOINTS".center(78) + " #")
        final_selector_path = fit_all_selectors(
            config=config,
            pretrained_model_path=str(pretrained_model_path),
            data_root=data_root,
            session_ids=[1, 2, 3, 4]
        )
        print(f"\n Selectors fitted. Final selector: {final_selector_path}")

    # Step 3: Evaluation
    if mode in ["all", "evaluate"]:
        print("# STEP 3: EVALUATION".center(78) + " #")
        
        checkpoint_dir = Path(config.checkpoint_dir)
        # Verify at least the final session checkpoint exists
        if not final_model_path.exists():
            print(f"Error: Trained model not found: {final_model_path}")
            print("run incremental training first")
            return
        
        results = evaluate_cumulative(
            checkpoint_dir=str(checkpoint_dir),
            data_root=data_root,
            session_ids=[1, 2, 3, 4],
            config=config,
            output_path=str(results_path)
        )
        
        print(f"\n Evaluation complete!")
        print(f" Results saved: {results_path}")
    
    print("PIPELINE COMPLETE!")


def main():
    parser = argparse.ArgumentParser(
        description="Run LNA Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete pipeline
    python run_pipeline.py --mode all

    # Run only pre-training
    python run_pipeline.py --mode pretrain

    # Run only incremental training
    python run_pipeline.py --mode incremental

    # Run only evaluation
    python run_pipeline.py --mode evaluate
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "pretrain", "incremental", "selectors", "evaluate"],
        help="Which part of pipeline to run"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/final_data",
        help="Root directory with session data"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            mode=args.mode,
            data_root=args.data_root,
            device=args.device
        )
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()