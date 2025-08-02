import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class MyReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return torch.clamp_(x, min=0) #inplace means that the input tensor will be modified directly, notice the _
        else:
            return torch.clamp(x, min=0)
        
class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.dim = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (scale and shift)
        # nn.Parameter makes it a learnable parameter, can be autograded
        self.gamma = nn.Parameter(torch.ones(num_features)) # Scale parameter 
        self.beta = nn.Parameter(torch.zeros(num_features)) # Shift parameter

        # Running stats for inference (not learnable)
        # register_buffer allows it to be stable during training and inference, can cross devices
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            # Compute batch mean and variance
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Normalize
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            # Update running statistics with Exponential Moving Average (EMA)
            # This is a common practice to maintain a stable mean and variance during training
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Scale and shift
        out = self.gamma * x_hat + self.beta  # reshape the feature space
        return out
        

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        layers.append(nn.Flatten())

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(MyReLU()) # or use nn.ReLU(inplace=False)
            if batch_norm:
                layers.append(MyBatchNorm1d(dim)) # or use nn.BatchNorm1d(dim)
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim)) # No Softmax here, as it will be handled by CrossEntropyLoss
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            acc = correct / total if total > 0 else 0.0
    return acc

# argparse.ArgumentParser cannot parse lists directly, so we need a custom function
# to parse the hidden dimensions from a string input.
def parse_hidden_dims(hidden_dims_str):
    if hidden_dims_str:
        return [int(dim) for dim in hidden_dims_str.split(',')]
    else:
        return []
    
def parse_args():
    parser = argparse.ArgumentParser(description="MLP Model")
    parser.add_argument('--input_dim', type=int, default=784, help='Input dimension')
    parser.add_argument('--hidden_dims', type=parse_hidden_dims, default='128,64', help='Hidden dimensions as comma-separated values')
    parser.add_argument('--output_dim', type=int, default=10, help='Output dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--batch_norm', type=bool, default=False, help='Use batch normalization')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training  epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')         
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',help='Directory to save checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(args.batch_norm)
    print(f'[Config] {args}')

    # ===================== Data Preparation =====================
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f'[Data] Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')

    # ===================== Model Initialization =====================
    model = MLP(input_dim=args.input_dim, hidden_dims=args.hidden_dims, output_dim=args.output_dim, dropout=args.dropout, batch_norm=args.batch_norm)
    model.to(device)
    print(f'[Model] {model}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f'[TensorBoard] Logging to {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f'[Checkpoint] Directory: {args.ckpt_dir}')
    torch.manual_seed(args.seed)
    print(f'[Seed] Random seed set to {args.seed}')

    # ===================== Training Loop =====================
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, test_loader, device)

        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Eval Acc: {val_acc:.4f}')
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/eval', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(args.ckpt_dir, 'best_model.pth')

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': vars(args),  
            }, checkpoint_path)
            print(f'[Checkpoint] Saved best model at epoch {epoch+1} with val_acc = {val_acc:.4f}')

    writer.close()
    print(f'Best validation accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    main()
    
    # To run this script, use the command:
    # source .venv/bin/activate
    # python MLP/mlp.py --ckpt_dir checkpoints
    # You can also specify other arguments like --input_dim, --hidden_dims, etc.

    # Use `tensorboard --logdir=logs` to visualize training progress
