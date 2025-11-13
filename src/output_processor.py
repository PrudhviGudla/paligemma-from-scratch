import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib import patches
import os

# Decoder of VQVAE for segmentation masks
class ResBlock(nn.Module):
    """Residual block with 3 convolutions"""
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(features, features, kernel_size=1, padding=0)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x + residual


class MaskDecoder(nn.Module):
    """VQ-VAE decoder that reconstructs 64x64 masks from 4x4 quantized codes"""
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        
        # Initial conv 
        self.conv0 = nn.Conv2d(embedding_dim, 128, kernel_size=1, padding=0)
        
        # ResBlocks
        self.resblock0 = ResBlock(128)
        self.resblock1 = ResBlock(128)
        
        # Upsampling layers: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        # Based on checkpoint: 128->128->64->32->16
        self.deconv0 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        
        # Final conv
        self.conv_final = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        
    def forward(self, x):
        """
        Args:
            x: [B, 4, 4, C] quantized vectors
        Returns:
            masks: [B, 1, 64, 64] reconstructed masks
        """
        # Convert from (B, H, W, C) to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv0(x))
        x = self.resblock0(x)
        x = self.resblock1(x)
        
        x = F.relu(self.deconv0(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        
        x = self.conv_final(x)
        return x


def load_vqvae_weights(checkpoint_path: str) -> Tuple[torch.Tensor, Dict]:
    """
    Load VQ-VAE weights (already in PyTorch format)
    
    Args:
        checkpoint_path: Path to vae-oid.npz
        
    Returns:
        embeddings: [num_codes, embedding_dim] codebook tensor
        decoder_params: Dictionary of decoder parameters
    """
    checkpoint = dict(np.load(checkpoint_path))
    
    def conv(name):
        """Extract conv layer weights"""
        weight_key = f'{name}.weight'
        bias_key = f'{name}.bias'
        
        if weight_key not in checkpoint:
            return None
        
        weight = checkpoint[weight_key]
        bias = checkpoint[bias_key]
        
        return {
            'weight': torch.from_numpy(weight).float(),
            'bias': torch.from_numpy(bias).float()
        }
    
    def resblock(name):
        """Extract ResBlock weights (3 conv layers)"""
        return {
            'conv1': conv(f'{name}.0'),
            'conv2': conv(f'{name}.2'),
            'conv3': conv(f'{name}.4'),
        }
    
    # Extract embeddings (codebook)
    # [num_codes, embedding_dim] = [128, 512]
    embeddings = torch.from_numpy(checkpoint['_vq_vae._embedding']).float()
    # Extract decoder parameters 
    decoder_params = {
        'conv0': conv('decoder.0'),
        'resblock0': resblock('decoder.2.net'),
        'resblock1': resblock('decoder.3.net'),
        'deconv0': conv('decoder.4'),
        'deconv1': conv('decoder.6'),
        'deconv2': conv('decoder.8'),
        'deconv3': conv('decoder.10'),
        'conv_final': conv('decoder.12'),
    }   
    return embeddings, decoder_params


def load_weights_into_model(model: MaskDecoder, decoder_params: Dict):
    """
    Load decoder parameters into PyTorch model
    
    Args:
        model: MaskDecoder instance
        decoder_params: Dictionary from load_vqvae_weights
    """
    state_dict = {}
    
    # Map parameters to model state dict
    for param_name, param_dict in decoder_params.items():
        if param_dict is None:
            continue
            
        if 'resblock' in param_name:
            # ResBlock has nested structure
            for conv_name, conv_params in param_dict.items():
                if conv_params:
                    state_dict[f'{param_name}.{conv_name}.weight'] = conv_params['weight']
                    state_dict[f'{param_name}.{conv_name}.bias'] = conv_params['bias']
        else:
            # Regular conv/deconv layer
            state_dict[f'{param_name}.weight'] = param_dict['weight']
            state_dict[f'{param_name}.bias'] = param_dict['bias']
    
    # Load into model
    missing, unexpected = model.load_state_dict(state_dict, strict=True)


class PaliGemmaDetectionParser:
    """Parse PaliGemma detect outputs (bounding boxes only)"""
    
    DETECT_REGEX = re.compile(
        r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([^;<>\n]+)?'
    )
    
    def extract_objects(self, text: str, width: int, height: int) -> List[Dict]:
        """
        Extract bounding boxes from detect output
        
        Returns:
            List of dicts with keys: xyxy, name
        """
        objects = []
        
        for match in self.DETECT_REGEX.finditer(text):
            y1, x1, y2, x2 = [int(match.group(i)) / 1024 for i in range(1, 5)]
            label = match.group(5)
            
            y1 = int(round(y1 * height))
            x1 = int(round(x1 * width))
            y2 = int(round(y2 * height))
            x2 = int(round(x2 * width))
            
            objects.append({
                'xyxy': (x1, y1, x2, y2),
                'name': label.strip() if label else 'object'
            })     
        return objects


class PaliGemmaSegmentationParser:
    """Parse PaliGemma output and extract segmentation masks"""
    
    SEGMENT_REGEX = re.compile(
        r'(.*?)'  # Any text before
        # 4 location tokens - each needs its own capture group
        r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*'
        # 16 optional segmentation tokens - each needs its own capture group
        r'(?:'
        r'<seg(\d{3})><seg(\d{3})><seg(\d{3})><seg(\d{3})>'
        r'<seg(\d{3})><seg(\d{3})><seg(\d{3})><seg(\d{3})>'
        r'<seg(\d{3})><seg(\d{3})><seg(\d{3})><seg(\d{3})>'
        r'<seg(\d{3})><seg(\d{3})><seg(\d{3})><seg(\d{3})>'
        r')?\s*'
        r'([^;<>]+)? ?(?:; )?'  # Label text
    )

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to vae-oid.npz
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load weights
        print(f"Loading VQ-VAE weights from {checkpoint_path}...")
        self.embeddings, decoder_params = load_vqvae_weights(checkpoint_path)
        self.embeddings = self.embeddings.to(self.device)
        
        # Initialize decoder with correct embedding_dim
        embedding_dim = self.embeddings.shape[1]
        self.decoder = MaskDecoder(embedding_dim=embedding_dim)
        load_weights_into_model(self.decoder, decoder_params)
        self.decoder.to(self.device)
        self.decoder.eval()
    
    @torch.no_grad()
    def reconstruct_masks(self, seg_indices: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct 64x64 masks from 16 segmentation token indices
        
        Args:
            seg_indices: [B, 16] tensor of integers in [0, 127]
            
        Returns:
            masks: [B, 64, 64] tensor of floats in [0, 1]
        """
        batch_size = seg_indices.shape[0]
        
        # Look up embeddings: [B*16] -> [B*16, embedding_dim]
        indices_flat = seg_indices.reshape(-1)
        encodings = self.embeddings[indices_flat]
        
        # Reshape to spatial: [B, 4, 4, embedding_dim]
        encodings = encodings.reshape(batch_size, 4, 4, -1)
        
        # Decode to 64x64 masks: [B, 1, 64, 64]
        masks = self.decoder(encodings)
        masks = masks.squeeze(1)  # [B, 64, 64]
        
        # Normalize to [0, 1] range
        masks = torch.clamp(masks * 0.5 + 0.5, 0, 1) # to bound the range to [0,1]
        return masks
    
    def extract_objects(
        self, 
        text: str, 
        width: int, 
        height: int,
        unique_labels: bool = False
    ) -> List[Dict]:
        """
        Extract objects with bounding boxes and segmentation masks
        
        Args:
            text: PaliGemma output string with <loc> and <seg> tokens
            width: Image width in pixels
            height: Image height in pixels
            unique_labels: If True, append ' to duplicate labels
            
        Returns:
            List of dictionaries with keys:
                - content: Full token string
                - xyxy: (x1, y1, x2, y2) bounding box
                - mask: [H, W] numpy array segmentation mask (or None)
                - name: Object label
        """
        objects = []
        seen_labels = set()
        
        while text:
            match = self.SEGMENT_REGEX.match(text)
            if not match:
                break
            
            groups = list(match.groups())
            before_text = groups.pop(0)
            label = groups.pop()
            
            # Parse bounding box coordinates (normalized to 1024)
            y1, x1, y2, x2 = [int(g) / 1024 for g in groups[:4]]
            y1 = int(round(y1 * height))
            x1 = int(round(x1 * width))
            y2 = int(round(y2 * height))
            x2 = int(round(x2 * width))
            
            # Parse segmentation tokens
            seg_tokens = groups[4:20]
            
            if seg_tokens[0] is not None:
                # Convert to tensor and reconstruct mask
                seg_indices = torch.tensor(
                    [[int(t) for t in seg_tokens]], 
                    dtype=torch.long,
                    device=self.device
                )
                
                # Get 64x64 mask
                mask_64 = self.reconstruct_masks(seg_indices)[0]  # [64, 64]
                mask_64_np = mask_64.cpu().numpy()
                
                # Resize mask to bounding box and place in full image
                mask = np.zeros((height, width), dtype=np.float32)
                
                if y2 > y1 and x2 > x1:
                    # Resize 64x64 mask to bbox size
                    mask_pil = Image.fromarray((mask_64_np * 255).astype('uint8'))
                    mask_resized = mask_pil.resize((x2 - x1, y2 - y1), Image.BILINEAR)
                    mask[y1:y2, x1:x2] = np.array(mask_resized) / 255.0
            else:
                mask = None
            
            # Add text before object if exists
            content = match.group()
            if before_text:
                objects.append({'content': before_text})
                content = content[len(before_text):]
            
            # Handle unique labels
            if unique_labels:
                while label in seen_labels:
                    label = (label or '') + "'"
                seen_labels.add(label)
            
            # Add object
            objects.append({
                'content': content,
                'xyxy': (x1, y1, x2, y2),
                'mask': mask,
                'name': label
            })
            
            # Move to next match
            text = text[len(match.group()):]
        
        # Add remaining text
        if text:
            objects.append({'content': text})
        
        return objects
    

def visualize_detections(image: Image.Image, objects: List[Dict], save_path: str = None):
    """Visualize bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]
    
    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = obj['xyxy']
        color = colors[i % len(colors)]
        
        # Draw bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = obj['name']
        bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 25), label, fill='white', font=font)
    
    if save_path:
        image.save(save_path)
        print(f"Saved detection visualization to {save_path}")
    
    return image


def visualize_segmentation(
    image: np.ndarray,
    objects: List[Dict],
    alpha: float = 0.5,
    save_path: Optional[str] = None
):
    """
    Visualize segmentation masks on image
    
    Args:
        image: [H, W, 3] RGB image (0-255)
        objects: Output from extract_objects
        alpha: Mask transparency (0=transparent, 1=opaque)
        save_path: Optional path to save visualization
    """
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    obj_count = 0
    for obj in objects:
        if 'mask' not in obj or obj['mask'] is None:
            continue
        
        mask = obj['mask']
        x1, y1, x2, y2 = obj['xyxy']
        label = obj.get('name', 'object')
        color = colors[obj_count % len(colors)]
        obj_count += 1
        
        # Draw mask
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0.5] = [*color[:3], alpha]
        ax.imshow(colored_mask)
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            x1, max(0, y1-5), label,
            color='white', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved segmentation visualization to {save_path}")
    
    return fig


class PaliGemmaOutputProcessor:
    """Unified processor for detection and segmentation outputs"""
    
    def __init__(self, vae_checkpoint_path: str = None, device: str = 'cuda'):
        self.device = device
        self.detection_parser = PaliGemmaDetectionParser()
        
        # Only load segmentation parser if checkpoint exists
        self.segmentation_parser = None
        self.vae_checkpoint_path = vae_checkpoint_path
    
    def process_output(
        self, 
        prompt: str, 
        output_text: str, 
        image: Image.Image,
        save_path: str = None
    ):
        """
        Process PaliGemma output based on prompt type
        
        Args:
            prompt: Input prompt (used to determine task)
            output_text: Model output text with special tokens
            image: Input PIL Image
            save_path: Optional path to save visualization
        """
        width, height = image.size
        
        # Determine task from prompt
        if 'segment' in prompt.lower():
            print("\nTask: SEGMENTATION")
            if self.vae_checkpoint_path and os.path.exists(self.vae_checkpoint_path):
                self.segmentation_parser = PaliGemmaSegmentationParser(
                    self.vae_checkpoint_path, self.device
                )

            if self.segmentation_parser is None:
                print("Warning: VQ-VAE checkpoint not loaded, falling back to detection")
                return self._process_detection(output_text, image, width, height, save_path)
            
            return self._process_segmentation(output_text, image, width, height, save_path)
        
        elif 'detect' in prompt.lower():
            print("\nTask: DETECTION")
            return self._process_detection(output_text, image, width, height, save_path)
        
        else:
            print("\nTask: TEXT GENERATION (no visualization)")
            return None
    
    def _process_detection(self, text, image, width, height, save_path):
        """Process detection output"""
        objects = self.detection_parser.extract_objects(text, width, height)
        
        print(f"Detected {len(objects)} objects:")
        for i, obj in enumerate(objects):
            print(f"  [{i}] {obj['name']}: bbox={obj['xyxy']}")
        
        if save_path is None:
            save_path = "detection_output.png"
        
        vis_image = image.copy()
        visualize_detections(vis_image, objects, save_path)
        return objects
    
    def _process_segmentation(self, text, image, width, height, save_path):
        """Process segmentation output"""
        objects = self.segmentation_parser.extract_objects(text, width, height)
        
        print(f"Segmented {len(objects)} objects:")
        for i, obj in enumerate(objects):
            if obj.get('mask') is not None:
                coverage = obj['mask'].sum() / (width * height) * 100
                print(f"  [{i}] {obj['name']}: bbox={obj['xyxy']}, "
                      f"mask_coverage={coverage:.2f}%")
        
        if save_path is None:
            save_path = "segmentation_output.png"
        
        image_np = np.array(image)
        visualize_segmentation(image_np, objects, 0.5, save_path)
        plt.close()
        return objects
    


# Usage Example
if __name__ == '__main__':
    # Initialize parser
    parser = PaliGemmaSegmentationParser(
        checkpoint_path='vae-oid.npz',
        device='cuda'
    )
    
    # EXAMPLE_STRING
    paligemma_output = (
        '<loc0000><loc0000><loc0930><loc1012> '
        '<seg114><seg074><seg106><seg044><seg030><seg027><seg119><seg119>'
        '<seg120><seg117><seg082><seg082><seg051><seg005><seg125><seg097> '
        'wall ; '
        '<loc0722><loc0047><loc0895><loc0378> '
        '<seg068><seg114><seg014><seg037><seg029><seg063><seg048><seg104>'
        '<seg010><seg056><seg021><seg056><seg019><seg017><seg102><seg121> '
        'car ; '
        '<loc0180><loc0596><loc0782><loc0961> '
        '<seg026><seg028><seg028><seg026><seg104><seg026><seg029><seg022>'
        '<seg000><seg068><seg092><seg125><seg003><seg127><seg121><seg043> '
        'david bowie'
    )
    
    pil_image = Image.open('./test_images/bowie.jpg')
    
    # Image dimensions
    img_width, img_height = pil_image.size
    image = np.array(pil_image)
    
    # Extract objects
    print("\n" + "="*60)
    print("EXTRACTING OBJECTS")
    print("="*60)
    objects = parser.extract_objects(
        paligemma_output,
        width=img_width,
        height=img_height,
        unique_labels=True
    )
    
    # Print results
    print(f"\nExtracted {len(objects)} objects:")
    for i, obj in enumerate(objects):
        if 'mask' in obj and obj['mask'] is not None:
            mask_coverage = obj['mask'].sum() / (img_width * img_height) * 100
            print(f"  [{i}] {obj['name']}: "
                  f"bbox={obj['xyxy']}, "
                  f"mask_coverage={mask_coverage:.2f}%")

    # Visualize
    fig = visualize_segmentation(image, objects)
    plt.savefig('segmentation_result.png')
    plt.show()
