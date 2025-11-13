from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    """Wraps the creation of patch embeddings from image"""
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.embed_dim = config.hidden_size

        # non overlapping 2D convolution on image with kernel of the size as the patch size and further shape transformations will result in an efficient way to create patch embeddings
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size, # since we dont want overlap for patch embeddings
            padding="valid" # means no padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        # Creates a learnable lookup table for position embeddings. It's essentially a weight matrix where you look up the position index to get its embedding vector
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        # Creates a non-learnable tensor that stores position indices [0, 1, 2, ..., num_positions-1]
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)), # to create a positional index tensor that can be easily broadcast across a batch of sequences.
            persistent=False
        ) # position_ids automatically moves to GPU when registering is used

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        _, _, height, width = pixel_values.shape

        # [batch_size, num_channels, height, width] -> [batch_size, embed_dim, height // patch_size, width // patch_size]
        patch_embeds = self.patch_embedding(pixel_values)

        # flatten the last two dimensions num_patches_h x num_patches_w -> [batch_size, embed_dim, num_patches]
        # and transpose the last two dimensions to [batch_size, num_patches, embed_dim]
        embeddings = patch_embeds.flatten(2).transpose(1,2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    """Wrpas the Self Attention in Encoder layer"""
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads # dk/dhead
        self.scale = self.head_dim**(-0.5) # for attention calculation 1/sqrt(dk)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)  # [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]

        # [batch_size, num_heads, num_patches, head_dim],  for multi head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # [batch_size, num_heads, num_patches, num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale) # (K.Q^T)/sqrt(dhead)
        # row wise softmax on the attention weights calculated
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # [batch_size, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)
        
        # [batch_size, num_patches, num_heads,  head_dim]
        attn_output = attn_output.transpose(1,2).contiguous() # the reason for using contiguous is based on how pytorch stores tensors and how shape modification is done
        # contiguous() makes sure the attention outputs are stored in a continous memory block and further shape changes only change how the elements of the stored tensor are indexed/read

        #  [batch_size, num_patches, embed_dim] hence concatenation
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # no dim change 
        attn_output = self.out_proj(attn_output) # to kind of mix the cocnatenated attention outputs
        return attn_output


class SiglipMLP(nn.Module):
    """Wraps the MLP layers in the encoder layer"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh") # heuristically found out to be best for the model
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipVisionEncoderLayer(nn.Module):
    """Wrpas all the layers in a single encoder layer"""
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states # [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states) # no dim change# no dim chnage
        hidden_states = self.self_attn(hidden_states=hidden_states) # no dim change
        hidden_states = hidden_states + residual # no dim change
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states) # no dim change
        hidden_states = self.mlp(hidden_states) # no dim change
        hidden_states = hidden_states + residual
        return hidden_states


class SiglipVisionEncoder(nn.Module):
    """Wraps the encoder layers"""
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds # [batch_size, num_image_tokens, hidden_size]

        for encoder_layer in self.layers:
            # [batch_size, num_image_tokens, hidden_size] to [batch_size, num_image_tokens, hidden_size]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """Wrpas the embeddings layer, encoder layers, final layer norm layer"""
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        hidden_states = self.embeddings(pixel_values) # [batch_size, num_image_tokens/num_patches, hidden_size]
        last_hidden_state = self.encoder(hidden_states) # [batch_size, num_image_tokens, hidden_size]
        last_hidden_state = self.post_layernorm(last_hidden_state) # [batch_size, num_image_tokens, hidden_size]
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    """Wraps the ViT"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # [batch_size, num_channels, image_size, image_size] to [batch_size, num_image_tokens, hidden_size]
        return self.vision_model(pixel_values=pixel_values)