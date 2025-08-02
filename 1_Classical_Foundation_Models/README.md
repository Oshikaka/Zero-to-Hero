# MLP Implementation for MNIST Classification

This module implements:
- [x] **Linear Regression + L1,L2 Regularization**  
  Focus: Autograd / MSE / SGD  
  Suggested Task: Hand-code forward + backward

- [x] **Logistic Regression**  
  Focus: Binary Classification / Sigmoid + BCE  
  Suggested Task: Simple MNIST classification

- [x] **MLP (Multi-Layer Perceptron)**  
  Focus: Fully-connected layers / Dropout / BatchNorm  
  Suggested Task: MNIST classification

- [ ] **CNN (Convolutional Neural Network)**  
  Focus: Convolution / Pooling / Kernel understanding  
  Suggested Task: CIFAR10 classification

- [ ] **RNN (Recurrent Neural Network)**  
  Focus: Sequence modeling / Hidden state propagation  
  Suggested Task: Text sentiment classification

## File Structure

```
1_Classical_Foundation_Models/
├── mlp.py                          # Main training script
├── inference_pretrained_model.py   # Inference script
├── linear_regression.ipynb         # Linear regression examples
├── logistic_regression.ipynb       # Logistic regression examples
├── data/                           # MNIST dataset (auto-downloaded)
├── logs/                           # TensorBoard logs
└── checkpoints/                    # Saved model checkpoints
    └── best_model.pth
```



## MLP

### Custom Components
- **MyReLU**: Custom ReLU activation with optional in-place operation
- **MyBatchNorm1d**: Custom 1D Batch Normalization implementation with learnable parameters (γ, β) and running statistics
- **MLP**: Flexible multi-layer perceptron with configurable architecture

### Training Features
- Configurable hidden layer dimensions
- Dropout regularization
- Batch normalization (optional)
- TensorBoard logging
- Model checkpointing (saves best model)
- GPU support (automatic CUDA detection)

## Architecture

```
Input (784) → Flatten → Linear → ReLU → [BatchNorm] → Dropout → ... → Linear (10)
```

## Usage

### Basic Training
```bash
# Activate virtual environment
source .venv/bin/activate

# Train with default parameters
cd 1_Classical_Foundation_Models
python mlp.py

# Custom configuration
python mlp.py --hidden_dims 256,128,64 --epochs 20 --batch_size 64 --learning_rate 0.001
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_dim` | int | 784 | Input dimension (28×28 for MNIST) |
| `--hidden_dims` | str | '128,64' | Hidden layer dimensions (comma-separated) |
| `--output_dim` | int | 10 | Output dimension (10 classes for MNIST) |
| `--dropout` | float | 0.5 | Dropout rate |
| `--batch_norm` | flag | False | Enable batch normalization |
| `--batch_size` | int | 32 | Training batch size |
| `--epochs` | int | 10 | Number of training epochs |
| `--learning_rate` | float | 0.001 | Adam optimizer learning rate |
| `--log_dir` | str | 'logs' | TensorBoard log directory |
| `--ckpt_dir` | str | 'checkpoints' | Model checkpoint directory |
| `--seed` | int | 42 | Random seed for reproducibility |

### Example Commands

```bash
# Train with batch normalization and more epochs
python MLP/mlp.py --batch_norm --epochs 50 --hidden_dims 512,256,128

# Train with higher learning rate and larger batch size
python MLP/mlp.py --learning_rate 0.01 --batch_size 128

# Train with custom architecture
python MLP/mlp.py --hidden_dims 1024,512,256,128,64 --dropout 0.3
```

## Monitoring Training

### TensorBoard
```bash
# Start TensorBoard server
tensorboard --logdir=logs

# Open browser and go to: http://localhost:6006
```

**Metrics tracked:**
- Training loss per epoch
- Validation accuracy per epoch

## Model Checkpoints

The script automatically saves the best model (highest validation accuracy) to:
```
checkpoints/best_model.pth
```

**Checkpoint contents:**
```python
{
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_acc': best_accuracy,
    'args': training_arguments_dict
}
```


## Custom Components Explained

### MyBatchNorm1d
- Implements batch normalization: `output = γ * (x - μ) / σ + β`
- **Training mode**: Uses batch statistics
- **Inference mode**: Uses running statistics (exponential moving average)
- Learnable parameters: `γ` (scale) and `β` (shift)

### MyReLU
- Custom ReLU implementation using `torch.clamp`
- Optional in-place operation for memory efficiency
- Equivalent to `nn.ReLU()` but educational implementation

## Performance

**Typical results with default settings:**
- Training time: ~2-3 minutes on GPU
- Best validation accuracy: ~97-98%
- Model size: ~100KB

## Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
tensorboard>=2.10.0
matplotlib>=3.5.0  # For inference visualization
```

## Use pretrained model to do inference

1. **Inference**: Use `inference_pretrained_model.py` to load and test saved models
2. **Visualization**: Check TensorBoard logs for training curves
3. **Experimentation**: Try different architectures and hyperparameters