# Training Guide: CNN 32×32 vs 224×224

## Quick Start

### Train 32×32 Model (Baseline)
```bash
python train_cnn_32.py --epochs 200 --early-stopping --patience 20
```
- **Expected time**: ~30-40 minutes
- **Expected accuracy**: 71.5-72%
- **Results saved to**: `results_cnn/`

### Train 224×224 Model (Upscaled)
```bash
python train_cnn_224.py --epochs 200 --early-stopping --patience 20 --batch-size 64
```
- **Expected time**: 5-8 hours
- **Expected accuracy**: 75-76%
- **Results saved to**: `results_cnn_224/`

## Key Differences

| Feature | train_cnn_32.py | train_cnn_224.py |
|---------|-----------------|------------------|
| **Input size** | 32×32 (native) | 224×224 (upscaled) |
| **Upscaling** | None | Bilinear interpolation (same as GPT-4o) |
| **FC layer** | 4,096 inputs | 200,704 inputs |
| **Training time** | ~26 min/100 epochs | ~4-5 min/epoch |
| **Results directory** | `results_cnn/` | `results_cnn_224/` |
| **Batch size** | 128 (default) | 64 (recommended) |

## Command Line Options

Both scripts support the same options:

```bash
--epochs EPOCHS          # Number of epochs (default: 100)
--early-stopping         # Enable early stopping
--patience PATIENCE      # Early stopping patience (default: 5)
--flatline-patience N    # Flatline detection patience (default: 20)
--batch-size SIZE        # Batch size (default: 128)
--lr RATE                # Learning rate (default: 0.001)
--sweep                  # Epoch sweep mode
```

## Recommended Commands

### Standard Training (32×32)
```bash
python train_cnn_32.py --epochs 200 --early-stopping --patience 20
```

### Standard Training (224×224)
```bash
python train_cnn_224.py --epochs 200 --early-stopping --patience 20 --batch-size 64
```

### Quick Test (32×32, 20 epochs)
```bash
python train_cnn_32.py --epochs 20
```

### Quick Test (224×224, 10 epochs)
```bash
python train_cnn_224.py --epochs 10 --batch-size 64
```

## Output Files

### Results Directory Structure

**32×32 (`results_cnn/`):**
- `best_cnn_model.pth` - Trained model weights
- `training_history.png` - Loss/accuracy plots
- `training_history.json` - Training metrics
- `confusion_matrix.png` - Confusion matrix
- `evaluation_results.json` - Final evaluation

**224×224 (`results_cnn_224/`):**
- Same structure as above
- Separate directory to avoid conflicts

## Early Stopping Behavior

Both scripts have:
1. **Early stopping** (if enabled): Stops if accuracy doesn't improve for `patience` epochs
2. **Flatline detection** (always enabled): Stops if accuracy doesn't change for 20 epochs

Training will automatically stop when:
- No improvement for 20 epochs (if `--early-stopping --patience 20`)
- OR accuracy flatlines for 20 epochs
- OR reaches max epochs

## Monitoring Training

Watch for these messages:
- `"Best model saved! (Test Acc: XX.XX%)"` - New best accuracy found
- `"Early stopping triggered. No improvement for 20 epochs."` - Stopped due to no improvement
- `"Flatline detection triggered. Accuracy flatlined for 20 epochs."` - Stopped due to flatline

## Expected Results

### 32×32 Model
- **Accuracy**: 71.5-72%
- **Best epoch**: Usually around 20-50
- **Training time**: 30-40 minutes (with early stopping)

### 224×224 Model
- **Accuracy**: 75-76%
- **Best epoch**: Usually around 60-100
- **Training time**: 5-8 hours (with early stopping)

## Troubleshooting

### Out of Memory (224×224)
- Reduce batch size: `--batch-size 32`
- This will slow down training but use less memory

### Training Too Slow (224×224)
- Use smaller batch size (32) if memory constrained
- Consider using mixed precision (requires code modification)
- Reduce max epochs if you just want quick results

### Results Overwriting
- 32×32 saves to `results_cnn/`
- 224×224 saves to `results_cnn_224/`
- They won't overwrite each other

## Comparison

After training both, compare:
- Accuracy: 32×32 vs 224×224 vs GPT-4o (96.8%)
- Per-class performance: Especially cat/dog confusion
- Training time: Cost vs benefit analysis

