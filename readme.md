# MFMViT - Masked Frequency Modulation Vision Transformer

Vision Transformer-based model for image reconstruction using masked frequency modulation in the FFT domain.

## Overview

Reconstructs corrupted images by masking frequency components and using a ViT encoder-decoder to recover masked regions.

## Architecture

- **Encoder**: ViT Base (patch_size=16, embed_dim=768)
- **Decoder**: Multi-layer MLP (3 layers with GELU activation)
- **Input/Output**: 224×224 RGB images
