"""
Complete Pipeline Runner
Runs the entire comparison pipeline in sequence
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print("="*80)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error running: {cmd}")
        print(f"   Return code: {result.returncode}")
        response = input("\nContinue anyway? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(1)
    else:
        print(f"\n✅ {description} completed successfully!")
    
    return result.returncode


def check_env():
    """Check if .env file exists"""
    if not os.path.exists('.env'):
        print("\n⚠️  WARNING: .env file not found!")
        print("GPT-4o evaluation will be skipped unless you create .env with:")
        print("OPENAI_API_KEY=your_key_here")
        response = input("\nContinue anyway? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(0)


def main():
    """Run the full pipeline"""
    print("="*80)
    print("CIFAR-10: CNN vs GPT-4o Vision - Full Pipeline")
    print("="*80)
    print("\nThis will:")
    print("  1. Train a custom CNN on CIFAR-10 (~10-30 min on GPU)")
    print("  2. Evaluate both CNN and GPT-4o Vision on 2000 test images")
    print("  3. Generate comprehensive comparison visualizations")
    print("\n⚠️  Note: GPT-4o evaluation costs ~$10-20 and takes 30-60 minutes")
    
    response = input("\nProceed with full pipeline? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Pipeline cancelled.")
        sys.exit(0)
    
    # Check environment
    check_env()
    
    # Step 1: Train CNN
    run_command(
        "python train_cnn.py",
        "Step 1: Training Custom CNN"
    )
    
    # Step 2: Compare models
    run_command(
        "python compare_models.py",
        "Step 2: Comparing CNN vs GPT-4o Vision"
    )
    
    print("\n" + "="*80)
    print("🎉 FULL PIPELINE COMPLETE!")
    print("="*80)
    print("\n📊 Check the generated PNG files and comparison_report.txt for results!")


if __name__ == "__main__":
    main()

