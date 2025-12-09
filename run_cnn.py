#!/usr/bin/env python3
"""
Run CNN Training on CIFAR-10 (Standalone)

Usage:
    python run_cnn.py                    # Train with default 20 epochs
    python run_cnn.py --epochs 50        # Train with 50 epochs
    python run_cnn.py --sweep            # Epoch sweep: test 5,10,15,20,25,30,40,50 epochs
    python run_cnn.py --sweep --epochs 5,10,20,30  # Custom epoch sweep
"""

import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='CIFAR-10 CNN Training - Test different epoch values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cnn.py                       # Train with 20 epochs (default)
  python run_cnn.py --epochs 50           # Train with 50 epochs
  python run_cnn.py --sweep               # Test epochs: 5,10,15,20,25,30,40,50
  python run_cnn.py --sweep --epochs 10,20,30,50  # Custom epoch values to test
  python run_cnn.py --early-stopping      # Enable early stopping (patience=5)
        """
    )
    parser.add_argument('--epochs', type=str, default='20',
                        help='Number of epochs (single value or comma-separated for sweep mode)')
    parser.add_argument('--sweep', action='store_true',
                        help='Enable epoch sweep mode to test multiple epoch values')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping (stops if no improvement for 5 epochs)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CIFAR-10 CNN Training")
    print("=" * 70)
    
    # Build command arguments
    cmd = [sys.executable, "train_cnn.py"]
    
    if args.sweep:
        # Epoch sweep mode
        if ',' in args.epochs:
            epoch_list = args.epochs
        else:
            epoch_list = "5,10,15,20,25,30,40,50"
        
        print(f"\nEPOCH SWEEP MODE")
        print(f"   Testing epochs: {epoch_list}")
        print(f"   This will train multiple models and compare accuracy")
        cmd.extend(["--sweep", "--epochs", epoch_list])
    else:
        # Single training mode
        epochs = int(args.epochs.split(',')[0])  # Take first value if comma-separated
        print(f"\nSingle Training Mode")
        print(f"   Epochs: {epochs}")
        cmd.extend(["--epochs", str(epochs)])
    
    if args.early_stopping:
        print(f"   Early Stopping: ENABLED (patience=5)")
        cmd.append("--early-stopping")
    
    cmd.extend(["--batch-size", str(args.batch_size)])
    cmd.extend(["--lr", str(args.lr)])
    
    print(f"\n   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"\n   Training: 50,000 images")
    print(f"   Evaluation: 2,000 stratified test images (200 per class)")
    print("-" * 70)
    
    # Run training
    print("\nStarting CNN training...")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("CNN TRAINING COMPLETE!")
        print("=" * 70)
        
        if args.sweep:
            print("\nGenerated files:")
            print("   - results_cnn/epoch_sweep_comparison.png  - Accuracy vs Epochs chart")
            print("   - results_cnn/epoch_sweep_results.json    - All sweep results")
            print("   - results_cnn/best_cnn_model.pth          - Best model weights")
        else:
            print("\nGenerated files (in results_cnn/):")
            print("   - best_cnn_model.pth           - Trained model weights")
            print("   - training_history.png         - Loss/accuracy plots")
            print("   - training_history.json        - Training metrics data")
            print("   - confusion_matrix.png         - Confusion matrix visualization")
        
        print("\nNext step: Run 'python compare_models.py' to compare with GPT-4o")
    else:
        print(f"\nTraining failed with exit code: {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
