# PaliGemma from Scratch

A PyTorch implementation of Google's PaliGemma vision-language model with custom segmentation output processing.

## Overview

This project implements the PaliGemma architecture for vision-language tasks including object detection and instance segmentation. The implementation includes:

- Full model architecture (SigLIP vision encoder + Gemma language decoder)
- Custom tokenizer and image preprocessing pipeline
- KV-cache implementation for efficient autoregressive generation
- **PyTorch segmentation parser** for processing model outputs with VQ-VAE decoder
- Visualization tools for detection and segmentation results

## Acknowledgments

This implementation follows [Umar Jamil's PaliGemma tutorial](https://github.com/hkproj/pytorch-paligemma) and his [detailed video walkthrough](https://www.youtube.com/watch?v=vAmKB7iPkWw). The model architecture (`paligemma.py`, `siglip.py`, `gemma.py`) and input processing pipeline were implemented by coding along with his lecture.

## My Contributions

Beyond following the tutorial, I added:

- **Segmentation Output Parser** (`output_processor.py`): Converted Big Vision's JAX/Flax VQ-VAE decoder to PyTorch for processing segmentation tokens, including:
  - Weight loading from `.npz` checkpoints with proper format handling
  - VQ-VAE decoder architecture (ResBlocks + transposed convolutions)
  - Mask reconstruction from 16 codebook indices to 64×64 segmentation masks
  
- **Unified Inference Pipeline**: Added automatic task detection (detect vs segment) and visualization
  
- **Code Documentation**: Added explanatory comments throughout, particularly in the sampling and output processing logic

## Architecture Details

### Model Components

**Vision Encoder (SigLIP)**
- Processes 224×224 images into 256 patch tokens
- Uses vision transformer architecture with learned positional embeddings

**Language Decoder (Gemma)**
- Transformer decoder with RoPE (Rotary Position Embeddings)
- Generates text and special tokens for structured outputs

**Segmentation Decoder (VQ-VAE - Referring Expression Segmentation)**

The VQ-VAE decoder is trained to convert discrete "segmentation tokens" into pixel-level masks. Here's how it works:

1. **Tokenization**: Each segmentation mask is encoded as 16 integers (0-127), representing a 4×4 spatial grid of codebook indices
2. **Codebook Lookup**: Each index maps to a 512-dimensional learned embedding vector
3. **Spatial Arrangement**: The 16 embeddings are reshaped into a 4×4×512 feature map
4. **Upsampling**: The decoder progressively upsamples through 4 stages:
   - 4×4 → 8×8 → 16×16 → 32×32 → 64×64
   - Uses transposed convolutions with ResNet-style skip connections
   - Channel dimensions: 512 → 128 → 64 → 32 → 16 → 1
5. **Output**: A 64×64 binary mask is produced, then resized to fit the detected bounding box

This approach allows the model to compress high-resolution segmentation masks into a compact token sequence that fits within the language model's vocabulary.

### Special Token Format

PaliGemma outputs structured predictions using special tokens:

```
Detection:  <loc0123><loc0456><loc0789><loc0234> object_name
Segmentation: <loc0123><loc0456><loc0789><loc0234> <seg001><seg023>...<seg127> object_name
```

- `<loc####>`: Bounding box coordinates (y1, x1, y2, x2) normalized to [0, 1023]
- `<seg###>`: 16 codebook indices (each 0-127) encoding the segmentation mask
- Coordinates map to positions in a 1024×1024 grid regardless of actual image size

## Setup

Tested on Windows 11 with NVIDIA RTX 4060 (8GB VRAM).

### Installation

```
# Clone repository
git clone https://github.com/PrudhviGudla/paligemma-from-scratch.git
cd paligemma-from-scratch

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Linux/Mac: source venv/bin/activate

# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

### Model Weights

PaliGemma weights require permission from Google:

1. Request access: [google/paligemma-3b-mix-224](https://huggingface.co/google/paligemma-3b-mix-224)
2. Create HuggingFace token with read permissions: [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Login via CLI:
   ```
   huggingface-cli login
   ```

Download model files:

```
# Detection and segmentation (recommended)
huggingface-cli download google/paligemma-3b-mix-224 --local-dir ./models/paligemma-3b-mix-224

# VQ-VAE segmentation decoder
huggingface-cli download big-vision/paligemma --include "vae-oid.npz" --local-dir ./models/
```

**Note**: Use `paligemma-3b-mix-*` models for detection/segmentation. The `pt` (pretrained) models only support text tasks like captioning and VQA.

## Usage

```
# Segmentation
python src/inference.py \
    --model_path "./models/paligemma-3b-mix-224" \
    --prompt "segment car\n" \
    --image_file_path "./test_images/street.jpg" \
    --vae_checkpoint "./models/vae-oid.npz" \
    --output_path "./output_results/segmentation.png"
```

or

Edit paths in `run_inference.bat` and run:
```
run_inference.bat
```

## Prompt Format

PaliGemma requires specific prompt formats (case-sensitive, newline-terminated):

| Task | Format | Example |
|------|--------|---------|
| Detection | `detect {class}\n` | `detect dog ; cat\n` |
| Segmentation | `segment {class}\n` | `segment person\n` |
| Captioning | `caption {lang}\n` | `caption en\n` |
| VQA | `answer {lang} {question}\n` | `answer en What is shown?\n` |
| OCR | `ocr\n` | `ocr\n` |

Multiple objects are separated by semicolons with spaces: `detect car ; person ; dog\n`

## Project Structure

```
paligemma-from-scratch/
├── src/
│   ├── paligemma.py           # Main model architecture
│   ├── siglip.py              # Vision encoder
│   ├── gemma.py               # Language decoder
│   ├── input_processor.py     # Image preprocessing and tokenization
│   ├── output_processor.py    # Detection/segmentation parsing
│   └── inference.py           # Inference pipeline
├── .gitignore                
├── requirements.txt
├── run_inference.bat
└── README.md
```

## Citation

If you use this implementation, please cite:

**Original PaliGemma Paper:**
```
@article{beyer2024paligemma,
  title={PaliGemma: A versatile 3B VLM for transfer},
  author={Beyer, Lucas and Steiner, Andreas and others},
  journal={arXiv preprint arXiv:2407.07726},
  year={2024}
}
```

**Tutorial Reference:**
```
Jamil, U. (2024). "Coding PaliGemma from scratch in PyTorch"
https://github.com/hkproj/pytorch-paligemma
```

**Disclaimer**: This is an independent educational project and is not affiliated with Google Research.
