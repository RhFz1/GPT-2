# NanoGPT Implementation

A PyTorch implementation of GPT-2, featuring distributed training capabilities and efficient attention mechanisms. This implementation is designed for both educational purposes and practical training of transformer-based language models.

## 🌟 Key Features

- GPT-2 architecture implementation with modular components
- Distributed Data Parallel (DDP) training support
- Dynamic learning rate scheduling with warmup
- Efficient attention mechanism using `scaled_dot_product_attention`
- Memory-efficient training with gradient accumulation
- Support for loading pretrained GPT-2 weights
- Mixed precision training support
- Custom lightweight data loader

## 🏗️ Architecture Overview

### Core Components

1. **ModelConfig**
   - Configurable parameters including:
     - `block_size`: Maximum sequence length (default: 1024)
     - `vocab_size`: Size of vocabulary (default: 50257)
     - `n_layer`: Number of transformer layers (default: 12)
     - `n_head`: Number of attention heads (default: 12)
     - `n_embd`: Embedding dimension (default: 768)
     - `attn_pdrop`: Attention dropout probability (default: 0.3)

2. **Main Model Components**

   - **MLP Block**
     ```python
     class MLP(nn.Module):
         # Features:
         # - 4x expansion in hidden layer
         # - GELU activation
         # - Dropout for regularization
     ```

   - **Causal Self Attention**
     ```python
     class CausalSelfAttention(nn.Module):
         # Key features:
         # - Multi-head attention implementation
         # - Causal masking for autoregressive property
         # - Efficient attention using scaled_dot_product_attention
         # - Proper scaling of attention scores
     ```

   - **Transformer Block**
     ```python
     class Block(nn.Module):
         # Components:
         # - Layer normalization
         # - Self-attention mechanism
         # - Feed-forward network
         # - Residual connections
     ```

### 🔍 Implementation Details

1. **Attention Mechanism**
   - Uses PyTorch's optimized `scaled_dot_product_attention`
   - Implements causal masking for autoregressive generation
   - Splits attention into multiple heads for parallel processing
   - Proper scaling of attention scores by `1/sqrt(head_dimension)`

2. **Weight Initialization**
   - Custom initialization strategy for different layer types
   - Special scaling for projection layers marked with `NANOGPT_SCALE_INIT`
   - Normal distribution initialization with configurable standard deviation

3. **Training Features**
   - Gradient accumulation for large effective batch sizes
   - Learning rate scheduling with:
     - Linear warmup
     - Cosine decay
     - Minimum learning rate floor
   - Weight decay optimization with AdamW
   - Gradient clipping at 1.0

4. **Optimization Strategy**
   ```python
   # Separate parameter groups for weight decay
   decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   ```

## 🚀 Training

### Configuration
```
    total_batch_size = 524288 # ~0.5M tokens
    B, T = 16, 1024 # Batch size and sequence length
    max_lr = 6e-4
    min_lr = 6e-5
    warmup_steps = 10
    max_steps = 50
```
### Training Loop Features

1. **Gradient Accumulation**
   - Enables training with larger effective batch sizes
   - Synchronizes gradients across distributed processes

2. **Mixed Precision Training**
   - Uses `torch.bfloat16` for efficient computation
   - Automatic mixed precision casting

3. **Distributed Training**
   - Supports multi-GPU training via DDP
   - Proper gradient synchronization
   - Process rank-aware data loading

## 📊 Data Loading

Custom `DataLoaderLite` class features:
- Efficient token-based data loading
- Support for distributed training
- Circular buffer implementation
- GPT-2 tokenizer integration

## 🔧 Model Loading and Saving

1. **Pretrained Model Loading**
   - Support for loading GPT-2 weights
   - Proper weight transposition for Conv1D to Linear conversion
   - Careful parameter alignment and shape checking

2. **Checkpointing**
   ```python
   torch.save({
       'model_dict': model.state_dict(),
       'optimizer_dict': optimizer.state_dict()
   }, 'model.pt')
   ```

## 🎯 Text Generation

Features:
- Top-k sampling implementation
- Temperature-controlled generation
- Maximum sequence length control
- Batch generation support

## 🛠️ Technical Requirements

- PyTorch with CUDA support (recommended)
- tiktoken for GPT-2 tokenization
- NCCL backend for distributed training
- Sufficient GPU memory for model size

## ⚠️ Important Notes

1. The implementation assumes CUDA availability for DDP training
2. Proper GPU memory management is crucial for large models
3. Learning rate scheduling parameters may need tuning for specific tasks
4. Weight initialization is crucial for training stability

## 🔍 Code Statistics

- Total classes: 6 main classes
- Key hyperparameters: Configurable via ModelConfig
- Training features: DDP, gradient accumulation, mixed precision
- Architecture: Standard GPT-2 with optimizations