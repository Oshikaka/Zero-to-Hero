# Zero-to-Hero
Models of Deep Learning, LLMs and Generative AI.

## Clone repo
```bash
git clone https://github.com/Oshikaka/Zero-to-Hero.git
cd Zero-to-Hero
```

## Installation
### Using `uv` (recommended)
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pyproject.toml
uv pip install -e .
```

### Using `pip`
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
# Make sure the vitual environment is activated
source .venv/bin/activate
python MLP/mlp.py
```

## Structure
### 🤡 Phase 1: Classical Foundation Models (Master Core Mechanisms)
In `1_Classical_Foundation_Models`.

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

### Phase 2: Mainstream Architectures (Foundation for LLMs & Generative Models)

- [ ] **Transformer (Vanilla)**  
  Focus: Multi-head attention / LayerNorm / Positional Encoding  
  Suggested Task: IMDB / Simple translation task

- [ ] **BERT (Lite)**  
  Focus: Masked LM / Segment Embeddings  
  Suggested Task: SST-2 / MLM pretraining task

- [ ] **GPT (Lite)**  
  Focus: Causal LM / Decoder-only transformer  
  Suggested Task: Train a character-level GPT on Shakespeare-style text

- [ ] **VAE (Variational Autoencoder)**  
  Focus: Encoder + Decoder / KL Divergence  
  Suggested Task: MNIST generation

- [ ] **GAN (Generative Adversarial Network)**  
  Focus: Generator vs Discriminator / BCELoss  
  Suggested Task: Generate handwritten digits

- [ ] **Diffusion (DDPM)**  
  Focus: Forward + Reverse noise process / U-Net predicting noise  
  Suggested Task: Implement small-size image denoising from scratch

- [ ] **CLIP (Lite)**  
  Focus: Image-text contrastive learning / Cosine similarity  
  Suggested Task: Align image + text embeddings

- [ ] **Vision Transformer (ViT)**  
  Focus: Patch embedding / MLP head  
  Suggested Task: CIFAR10 classification

