#!/usr/bin/env python3
"""
Run CNN Training on CIFAR-10 (Standalone)
This script runs only the CNN training without any GPT-4o API calls.
No API key required, no cost.
"""

import subprocess
import sys


def main():
    """Run CNN training on CIFAR-10"""
    print("=" * 70)
    print("🧠 CIFAR-10 CNN Training (Standalone)")
    print("=" * 70)
    print("\nThis will train a custom CNN on CIFAR-10.")
    print("  • Training: 50,000 images")
    print("  • Evaluation: 2,000 stratified test images (200 per class)")
    print("  • Architecture: 2 conv layers + FC head")
    print("  • Training: 20 epochs with Adam optimizer")
    print("  • Expected time: ~10-30 min (GPU) or ~1-2 hours (CPU)")
    print("\n⚖️  Same 2,000 test images will be used for GPT-4o comparison")
    print("💰 Cost: FREE (no API calls)")
    print("-" * 70)
    
    # Run CNN training
    print("\n🚀 Starting CNN training...")
    result = subprocess.run([sys.executable, "train_cnn.py"])
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ CNN TRAINING COMPLETE!")
        print("=" * 70)
        print("\n📁 Generated files:")
        print("   • best_cnn_model.pth           - Trained model weights")
        print("   • training_history.png         - Loss/accuracy plots")
        print("   • training_history.json        - Training metrics data")
        print("   • stratified_test_indices.json - Fixed 2,000 test image indices")
        print("\n💡 Next step: Run 'python compare_models.py' to compare with GPT-4o")
        print("   (Note: GPT-4o comparison requires API key and costs money)")
        print("   (The same 2,000 test images will be used for fair comparison)")
    else:
        print(f"\n❌ Training failed with exit code: {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()

