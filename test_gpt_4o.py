"""
GPT-4o Vision Evaluation on CIFAR-10 - NO UPSCALING
CS260 Final Project

This script tests GPT-4o with RAW 32x32 pixel images (no upscaling)
to compare against the upscaled 224x224 version.

Same methodology, just no resize step.
"""

import os
import io
import json
import time
import base64
from datetime import datetime
from dotenv import load_dotenv
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Configuration - NO UPSCALING
CONFIG = {
    'model': 'gpt-4o',
    'max_tokens': 10,
    'image_size': 32,  # RAW 32x32 - NO UPSCALING!
    'upscale': False,  # Disabled!
    'checkpoint_interval': 50,
    'retry_attempts': 3,
    'retry_delay': 2,
    'rate_limit_delay': 0.5,
}


def load_cifar10_test():
    """Load CIFAR-10 test dataset"""
    print("📦 Loading CIFAR-10 test dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_data = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"✅ Loaded {len(test_data)} test images")
    return test_data


def load_stratified_indices(filename='stratified_subset_2000.json'):
    """Load the stratified subset indices"""
    print(f"📄 Loading indices from {filename}...")
    
    with open(filename, 'r') as f:
        subset_info = json.load(f)
    
    indices = subset_info['indices']
    print(f"✅ Loaded {len(indices)} image indices")
    return indices, subset_info


def tensor_to_base64_raw(tensor):
    """
    Convert PyTorch tensor to base64-encoded PNG - NO UPSCALING!
    
    Args:
        tensor: PyTorch tensor (C, H, W) with values in [0, 1]
    
    Returns:
        Base64-encoded PNG string (32x32 pixels)
    """
    # Convert tensor → uint8 array (H, W, C)
    arr = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    
    # NO UPSCALING - Keep original 32x32 size!
    # img = img.resize((224, 224), Image.BILINEAR)  # COMMENTED OUT!
    
    # Encode as PNG
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()
    
    # Convert to base64
    return base64.b64encode(png_bytes).decode('utf-8')


def normalize_prediction(pred):
    """Normalize prediction to match CIFAR-10 classes"""
    pred = pred.lower().strip()
    
    if pred in CLASSES:
        return pred
    
    variations = {
        'car': 'automobile',
        'auto': 'automobile',
        'vehicle': 'automobile',
        'plane': 'airplane',
        'aeroplane': 'airplane',
        'jet': 'airplane',
        'boat': 'ship',
        'vessel': 'ship',
        'kitty': 'cat',
        'kitten': 'cat',
        'puppy': 'dog',
        'doggy': 'dog',
        'toad': 'frog',
        'pony': 'horse',
        'pickup': 'truck',
        'lorry': 'truck',
    }
    
    if pred in variations:
        return variations[pred]
    
    return pred


def load_checkpoint(checkpoint_file):
    """Load checkpoint if exists"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_file, results, processed_indices):
    """Save checkpoint"""
    checkpoint = {
        'results': results,
        'processed_indices': processed_indices,
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)


def evaluate_gpt4o_no_upscale(test_data, indices, output_dir='results_gpt4o_32x32'):
    """
    Evaluation with RAW 32x32 images (no upscaling)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_file = os.path.join(output_dir, 'checkpoint.json')
    results_file = os.path.join(output_dir, f'gpt4o_32x32_results_{timestamp}.json')
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found!")
        return None
    
    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized")
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        results = checkpoint['results']
        processed_set = set(checkpoint['processed_indices'])
        print(f"📂 Resuming from checkpoint: {len(processed_set)} images already processed")
    else:
        results = []
        processed_set = set()
    
    remaining_indices = [idx for idx in indices if idx not in processed_set]
    
    print(f"\n🔍 Evaluating GPT-4o Vision on {len(remaining_indices)} images...")
    print(f"   Model: {CONFIG['model']}")
    print(f"   ⚠️  Image size: {CONFIG['image_size']}x{CONFIG['image_size']} (RAW - NO UPSCALING!)")
    print(f"   Estimated cost: ~${len(remaining_indices) * 0.01:.2f}")
    print()
    
    pbar = tqdm(remaining_indices, desc="Processing (32x32)", unit="img")
    
    correct = sum(1 for r in results if r['correct'])
    total_processed = len(results)
    
    for i, idx in enumerate(pbar):
        img_tensor, label = test_data[idx]
        true_label = CLASSES[label]
        
        # Convert to base64 - NO UPSCALING!
        img_base64 = tensor_to_base64_raw(img_tensor)
        
        prediction = None
        error_msg = None
        raw_prediction = None
        
        for attempt in range(CONFIG['retry_attempts']):
            try:
                response = client.chat.completions.create(
                    model=CONFIG['model'],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Classify this image as exactly one of: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Return only the label, nothing else."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=CONFIG['max_tokens']
                )
                
                raw_prediction = response.choices[0].message.content.strip().lower()
                prediction = normalize_prediction(raw_prediction)
                break
            except Exception as e:
                error_msg = str(e)
                if attempt < CONFIG['retry_attempts'] - 1:
                    time.sleep(CONFIG['retry_delay'] * (attempt + 1))
                else:
                    prediction = 'error'
                    print(f"\n⚠️  Failed: {error_msg}")
        
        is_correct = (prediction == true_label)
        result = {
            'index': idx,
            'true_label': true_label,
            'prediction': prediction,
            'raw_prediction': raw_prediction,
            'correct': is_correct,
            'error': error_msg if prediction == 'error' else None,
            'image_size': '32x32'  # Mark as raw size
        }
        results.append(result)
        processed_set.add(idx)
        
        if is_correct:
            correct += 1
        total_processed += 1
        
        accuracy = correct / total_processed * 100
        pbar.set_postfix({
            'Acc': f'{accuracy:.1f}%',
            'Correct': f'{correct}/{total_processed}'
        })
        
        if (i + 1) % CONFIG['checkpoint_interval'] == 0:
            save_checkpoint(checkpoint_file, results, list(processed_set))
        
        time.sleep(CONFIG['rate_limit_delay'])
    
    pbar.close()
    save_checkpoint(checkpoint_file, results, list(processed_set))
    
    # Calculate metrics
    final_results = calculate_metrics(results)
    final_results['config'] = CONFIG
    final_results['timestamp'] = timestamp
    final_results['total_images'] = len(indices)
    final_results['image_size'] = '32x32 (no upscaling)'
    final_results['raw_results'] = results
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    print_summary(final_results)
    
    return final_results


def calculate_metrics(results):
    """Calculate evaluation metrics"""
    valid_results = [r for r in results if r['prediction'] != 'error']
    errors = [r for r in results if r['prediction'] == 'error']
    
    if not valid_results:
        return {'error': 'No valid results'}
    
    correct = sum(1 for r in valid_results if r['correct'])
    accuracy = correct / len(valid_results)
    
    class_correct = {c: 0 for c in CLASSES}
    class_total = {c: 0 for c in CLASSES}
    class_predicted = {c: 0 for c in CLASSES}
    
    for r in valid_results:
        true_label = r['true_label']
        pred_label = r['prediction']
        
        class_total[true_label] += 1
        if pred_label in class_predicted:
            class_predicted[pred_label] += 1
        
        if r['correct']:
            class_correct[true_label] += 1
    
    per_class_metrics = {}
    for c in CLASSES:
        tp = class_correct[c]
        total_true = class_total[c]
        total_pred = class_predicted.get(c, 0)
        
        recall = tp / total_true if total_true > 0 else 0
        precision = tp / total_pred if total_pred > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'correct': tp,
            'total': total_true
        }
    
    confusion_matrix = [[0] * 10 for _ in range(10)]
    for r in valid_results:
        true_idx = CLASSES.index(r['true_label'])
        pred_label = r['prediction']
        if pred_label in CLASSES:
            pred_idx = CLASSES.index(pred_label)
            confusion_matrix[true_idx][pred_idx] += 1
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total_valid': len(valid_results),
        'total_errors': len(errors),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': confusion_matrix,
        'macro_precision': np.mean([m['precision'] for m in per_class_metrics.values()]),
        'macro_recall': np.mean([m['recall'] for m in per_class_metrics.values()]),
        'macro_f1': np.mean([m['f1'] for m in per_class_metrics.values()])
    }


def print_summary(results):
    """Print evaluation summary"""
    print("\n" + "=" * 60)
    print("📊 GPT-4o VISION EVALUATION - 32x32 RAW (NO UPSCALING)")
    print("=" * 60)
    
    print(f"\n🎯 Overall Accuracy: {results['accuracy']*100:.2f}%")
    print(f"   Correct: {results['correct']}/{results['total_valid']}")
    print(f"   Errors: {results['total_errors']}")
    
    print(f"\n📈 Macro Metrics:")
    print(f"   Precision: {results['macro_precision']*100:.2f}%")
    print(f"   Recall:    {results['macro_recall']*100:.2f}%")
    print(f"   F1 Score:  {results['macro_f1']*100:.2f}%")
    
    print(f"\n📊 Per-Class Performance:")
    print("-" * 45)
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    
    for c in CLASSES:
        m = results['per_class_metrics'][c]
        print(f"{c:<12} {m['precision']*100:>9.1f}% {m['recall']*100:>9.1f}% {m['f1']*100:>9.1f}%")
    
    print("-" * 45)
    print("\n✅ Evaluation complete!")
    print("=" * 60)


def main():
    """Main function"""
    print("=" * 60)
    print("  GPT-4o Vision Evaluation - RAW 32x32 (NO UPSCALING)")
    print("  CS260 Final Project - Ablation Study")
    print("=" * 60)
    
    test_data = load_cifar10_test()
    indices, subset_info = load_stratified_indices('stratified_subset_2000.json')
    
    estimated_cost = len(indices) * 0.01
    print(f"\n⚠️  COST WARNING: {len(indices)} API calls (~${estimated_cost:.2f})")
    print(f"⚠️  Testing with RAW 32x32 images (no upscaling)")
    print()
    
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    results = evaluate_gpt4o_no_upscale(test_data, indices, output_dir='results_gpt4o_32x32')
    
    if results:
        print(f"\n🎉 GPT-4o (32x32) Accuracy: {results['accuracy']*100:.2f}%")
        print(f"📊 Compare with 224x224 upscaled version!")


if __name__ == "__main__":
    main()


