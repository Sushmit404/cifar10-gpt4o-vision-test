"""
CIFAR-10 Classification with GPT-4o Vision
CS260 Final Project

This script:
1. Downloads CIFAR-10 test dataset
2. Selects sample images
3. Converts them to PNG format
4. Sends them to GPT-4o Vision API
5. Compares predictions vs ground truth
"""

import os
import io
from dotenv import load_dotenv
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from openai import OpenAI

# Load environment variables
load_dotenv()

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']


def download_cifar10():
    """Download CIFAR-10 test dataset"""
    print("Downloading CIFAR-10 dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_data = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"Downloaded! Test dataset size: {len(test_data)}")
    return test_data


def select_sample_images(test_data, indices=None):
    """Select sample images from test dataset"""
    if indices is None:
        indices = [0, 1, 2, 3, 4]
    
    images = []
    labels = []
    
    for idx in indices:
        img, label = test_data[idx]
        images.append(img)
        labels.append(CLASSES[label])
    
    print(f"Selected {len(indices)} images")
    print(f"Ground-truth labels: {labels}")
    
    return images, labels, indices


def tensor_to_png_bytes(tensor):
    """
    Convert PyTorch tensor to PNG bytes for GPT-4o Vision
    
    Args:
        tensor: PyTorch tensor (C, H, W) with values in [0, 1]
    
    Returns:
        PNG image as bytes
    """
    # Convert tensor -> uint8 array (H, W, C)
    arr = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    
    # Upscale to 224x224 for better vision performance
    img = img.resize((224, 224), Image.BILINEAR)
    
    # Encode as PNG in memory
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def classify_with_gpt4o(image_bytes, api_key):
    """
    Send image to GPT-4o Vision for classification
    
    Args:
        image_bytes: PNG image as bytes
        api_key: OpenAI API key
    
    Returns:
        Predicted class label
    """
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o",
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
                            "url": f"data:image/png;base64,{image_bytes.decode() if isinstance(image_bytes, bytes) else image_bytes}"
                        }
                    }
                ]
            }
        ],
        max_tokens=10
    )
    
    return response.choices[0].message.content.strip().lower()


def visualize_results(images, true_labels, predictions, indices):
    """Display images with their true labels and predictions"""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    
    if len(images) == 1:
        axes = [axes]
    
    for i, (img, true_label, pred) in enumerate(zip(images, true_labels, predictions)):
        # Convert tensor to numpy for display
        img_np = img.numpy().transpose(1, 2, 0)
        
        axes[i].imshow(img_np)
        axes[i].axis('off')
        
        # Color code: green if correct, red if wrong
        color = 'green' if true_label == pred else 'red'
        axes[i].set_title(f"True: {true_label}\nPred: {pred}", 
                         color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('cifar10_gpt4o_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to cifar10_gpt4o_results.png")
    plt.show()


def main():
    """Main pipeline"""
    print("Starting CIFAR-10 + GPT-4o Vision Pipeline\n")
    
    # Step 1: Download CIFAR-10
    test_data = download_cifar10()
    print()
    
    # Step 2: Select sample images
    # You can change these indices to test different images
    sample_indices = [0, 1, 2, 3, 4]
    images, true_labels, indices = select_sample_images(test_data, sample_indices)
    print()
    
    # Step 3: Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found!")
        print("Please create a .env file with your API key:")
        print("OPENAI_API_KEY=your_key_here")
        return
    
    print("OpenAI API key loaded")
    print()
    
    # Step 4: Test with GPT-4o Vision
    print("Testing images with GPT-4o Vision...\n")
    predictions = []
    
    for i, (img, true_label) in enumerate(zip(images, true_labels)):
        print(f"Processing image {indices[i]}...", end=" ")
        png_bytes = tensor_to_png_bytes(img)
        
        try:
            # Convert bytes to base64 for API
            import base64
            img_base64 = base64.b64encode(png_bytes).decode('utf-8')
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
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
                max_tokens=10
            )
            
            prediction = response.choices[0].message.content.strip().lower()
            predictions.append(prediction)
            
            match = "[OK]" if prediction == true_label else "[WRONG]"
            print(f"{match} True: {true_label} | Predicted: {prediction}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            predictions.append("error")
    
    print()
    
    # Step 5: Calculate accuracy
    correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    accuracy = (correct / len(true_labels)) * 100
    print(f"Accuracy: {correct}/{len(true_labels)} ({accuracy:.1f}%)")
    print()
    
    # Step 6: Visualize results
    print("Creating visualization...")
    visualize_results(images, true_labels, predictions, indices)
    
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
