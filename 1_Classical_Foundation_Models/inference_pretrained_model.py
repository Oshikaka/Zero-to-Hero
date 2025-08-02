import torch
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from mlp import MLP

def load_model_from_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    args = argparse.Namespace(**checkpoint['args'])
    
    model = MLP(
        input_dim=args.input_dim, 
        hidden_dims=args.hidden_dims, 
        output_dim=args.output_dim,
        dropout=getattr(args, 'dropout', 0.5), 
        batch_norm=getattr(args, 'batch_norm', True)  
    )
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Model structure mismatch!")
        print(f"Saved model config: {args}")
        print(f"Error: {e}")
        raise
    
    model.eval()
    return model, args, checkpoint['best_acc']


def visualize_prediction(model, dataloader, device='cpu'):
    model.to(device)
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    # Plot first 5 predictions
    for i in range(5):
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Predicted: {preds[i].item()}, Label: {labels[i].item()}')
        plt.axis('off')
        os.makedirs('prediction', exist_ok=True)
        plt.savefig(f'prediction/{i}.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./checkpoints/best_model.pth', help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # Load model and config
    model, model_args, best_acc = load_model_from_checkpoint(args.ckpt)
    print(f"Best accuracy from checkpoint: {best_acc:.4f}")

    # Load MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Predict and visualize
    visualize_prediction(model, test_loader)

if __name__ == '__main__':
    main()