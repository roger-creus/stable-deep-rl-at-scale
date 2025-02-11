import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import get_act_fn_clss, get_act_fn_functional

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AtariCNN(nn.Module):
    def __init__(
        self,
        cnn_channels,
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super().__init__()
        out_channels = cnn_channels
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]
        ln_sizes = [20, 9, 7]
        in_channels = 4
        input_size = 84
        act_ = get_act_fn_clss(activation_fn)
        cnn = []
        for out_channel, kernel_size, stride, ln_size in zip(out_channels, kernel_sizes, strides, ln_sizes):
            # Add a convolutional layer
            cnn.append(
                layer_init(
                    nn.Conv2d(in_channels, out_channel, kernel_size, stride=stride, device=device),
                )
            )
                        
            # Add a layer normalization layer if needed
            if use_ln:
                cnn.append(
                    nn.LayerNorm(
                        [out_channel, ln_size, ln_size], device=device
                    )
                )
                
            # Add an activation
            cnn.append(
                act_()
            )
            
            output_size = (input_size - kernel_size) / stride + 1
            input_size = output_size
            in_channels = out_channel
            
        # Add a flattening layer
        cnn.append(
            nn.Flatten()
        )
        self.cnn = nn.Sequential(*cnn)
        
    def forward(self, x):
        return self.cnn(x)
    
    
################################### IMPALA CNN ###################################

class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        input_shape,
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super().__init__()
        self.use_ln = use_ln
        self.act_ = get_act_fn_functional(activation_fn)
        
        self.conv0 = layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, device=device))
        if self.use_ln:
            self.ln0 = nn.LayerNorm([channels, input_shape, input_shape], device=device)

        self.conv1 = layer_init(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, device=device))
        if self.use_ln:
            self.ln1 = nn.LayerNorm([channels, input_shape, input_shape], device=device)

    def forward(self, x):
        inputs = x
        x = self.act_(x)
        x = self.conv0(x)
        if self.use_ln:
            x = self.ln0(x)
        x = self.act_(x)
        x = self.conv1(x)
        if self.use_ln:
            x = self.ln1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(
        self,
        input_shape,
        out_channels,
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super().__init__()
        self.use_ln = use_ln
        
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = layer_init(nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1, device=device))
        
        if use_ln:
            self.ln = nn.LayerNorm([out_channels, input_shape[1], input_shape[2]], device=device)

        self.res_block0 = ResidualBlock(self._out_channels, (self._input_shape[1] + 1) // 2, use_ln=use_ln, activation_fn=activation_fn, device=device)
        self.res_block1 = ResidualBlock(self._out_channels, (self._input_shape[1] + 1) // 2, use_ln=use_ln, activation_fn=activation_fn, device=device)

    def forward(self, x):
        x = self.conv(x)
        if self.use_ln:
            x = self.ln(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
    
class ImpalaCNN(nn.Module):
    def __init__(
        self,
        cnn_channels,
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super().__init__()
        shape = (4, 84, 84)
        conv_seqs = []
        for out_channels in cnn_channels:
            conv_seq = ConvSequence(
                shape,
                out_channels,
                use_ln=use_ln,
                activation_fn=activation_fn,
                device=device
            )
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        
        activation = get_act_fn_clss(activation_fn)()
        
        conv_seqs += [
            activation,
            nn.Flatten(),
        ]
        
        self.cnn = nn.Sequential(*conv_seqs)
        
    def forward(self, x):
        return self.cnn(x)
    
################################ Dense Residual Nature CNN ################################
class DenseResidualConvSequence(nn.Module):
    def __init__(
        self,
        input_shape,   # tuple like (channels, height, width)
        out_channels,
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super().__init__()
        self.use_ln = use_ln
        self.activation = get_act_fn_clss(activation_fn)()
        self._input_shape = input_shape
        self._out_channels = out_channels

        # Initial convolution: projects input to out_channels.
        self.conv = layer_init(nn.Conv2d(in_channels=input_shape[0],
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         padding=1,
                                         device=device))
        if use_ln:
            self.ln = nn.LayerNorm([out_channels, input_shape[1], input_shape[2]], device=device)
        
        # Downsample spatially (as in your ConvSequence).
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # After pooling, the spatial dims shrink.
        # Use a residual block on the resulting feature map.
        # (Assume new spatial size is (h+1)//2; we pass that as input_shape to the ResidualBlock.)
        pooled_h = (input_shape[1] + 1) // 2
        self.res_block1 = ResidualBlock(channels=out_channels,
                                        input_shape=pooled_h,
                                        use_ln=use_ln,
                                        activation_fn=activation_fn,
                                        device=device)
        # Dense connection: after concatenating the pool output and first residual output,
        # we reduce the concatenated channels (2*out_channels) back to out_channels.
        self.reduce_conv = layer_init(nn.Conv2d(in_channels=2 * out_channels,
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                device=device))
        # A second residual block processes the reduced features.
        self.res_block2 = ResidualBlock(channels=out_channels,
                                        input_shape=pooled_h,
                                        use_ln=use_ln,
                                        activation_fn=activation_fn,
                                        device=device)

    def forward(self, x):
        # Initial conv and optional layer norm.
        x = self.conv(x)
        if self.use_ln:
            x = self.ln(x)
        # Downsample spatially.
        x = self.pool(x)
        # First residual branch.
        out1 = self.res_block1(x)
        # Dense connection: concatenate the pooled input and its processed version.
        x_cat = torch.cat([x, out1], dim=1)  # shape: (2*out_channels, H, W)
        # Reduce channels back to out_channels.
        x_reduced = self.activation(self.reduce_conv(x_cat))
        # Second residual branch.
        out2 = self.res_block2(x_reduced)
        # Final dense fusion: concatenate the reduced input and the second branch’s output.
        out = torch.cat([x_reduced, out2], dim=1)  # Final channels: 2*out_channels.
        return out

    def get_output_shape(self):
        # Given input_shape = (C, H, W), after max pooling the spatial dims become (H+1)//2 and (W+1)//2.
        _, h, w = self._input_shape
        new_h = (h + 1) // 2
        new_w = (w + 1) // 2
        return (2 * self._out_channels, new_h, new_w)

class DenseResidualCNN(nn.Module):
    def __init__(
        self,
        cnn_channels,   # e.g. a list like [32, 64, 64]
        use_ln=False,
        activation_fn="relu",
        device='cpu'
    ):
        super().__init__()
        # Atari: 4 channels, 84x84 images.
        shape = (4, 84, 84)
        blocks = []
        for out_channels in cnn_channels:
            block = DenseResidualConvSequence(
                input_shape=shape,
                out_channels=out_channels,
                use_ln=use_ln,
                activation_fn=activation_fn,
                device=device
            )
            blocks.append(block)
            # Update shape for the next block.
            shape = block.get_output_shape()
        
        blocks.append(nn.Flatten())
        blocks.append(get_act_fn_clss(activation_fn)())
        self.cnn = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.cnn(x)


################################### Vision Transformer ###################################
class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio, dropout=0.0, activation_fn="relu", device='cpu'):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim, device=device)
        # Use batch_first=True so that inputs are (B, seq_len, emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim, device=device)
        hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(emb_dim, hidden_dim, device=device)),
            get_act_fn_clss(activation_fn)(),  # GELU is a common choice in transformer literature
            nn.Dropout(dropout),
            layer_init(nn.Linear(hidden_dim, emb_dim, device=device)),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: (B, seq_len, emb_dim)
        # Attention sub-block with residual connection.
        x_attn = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + x_attn
        # MLP sub-block with residual connection.
        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp
        return x

# ---------------------------
# Vision Transformer Encoder for Atari
# ---------------------------
class VisionTransformerEncoder(nn.Module):
    def __init__(self,
                 image_size=84,
                 patch_size=14,
                 in_channels=4,
                 emb_dim=256,
                 depth=6,
                 activation_fn="relu",
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.0,
                 device='cpu'):
        """
        Args:
            image_size (int): Size of the (square) input image (84 for Atari).
            patch_size (int): Size of each patch. Must divide image_size evenly.
            in_channels (int): Number of input channels (e.g., 4 for frame stacking).
            emb_dim (int): Embedding dimension (output of the patch embedding).
            depth (int): Number of Transformer encoder blocks.
            num_heads (int): Number of attention heads.
            mlp_ratio (int or float): Ratio to compute hidden dimension of the MLP sub–block.
            dropout (float): Dropout rate.
            device (str): Device (e.g. 'cpu' or 'cuda').
        """
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.emb_dim = emb_dim

        # Patch embedding: using a conv layer with kernel and stride equal to patch_size.
        self.patch_embed = layer_init(
            nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size, device=device)
        )

        # Positional embeddings for each patch.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim, device=device))
        self.pos_drop = nn.Dropout(dropout)

        # Stack of Transformer encoder blocks.
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, mlp_ratio, dropout=dropout, device=device, activation_fn=activation_fn)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim, device=device)

        # Initialize the positional embeddings.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: Input images of shape (B, in_channels, image_size, image_size) e.g. (B, 4, 84, 84)
        Returns:
            A tensor of shape (B, emb_dim) representing the encoded image features.
        """
        B = x.shape[0]
        # Patch embedding: output shape (B, emb_dim, H, W), where H = W = image_size // patch_size.
        x = self.patch_embed(x)
        # Flatten spatial dimensions: (B, emb_dim, num_patches) and then transpose to (B, num_patches, emb_dim)
        x = x.flatten(2).transpose(1, 2)
        # Add positional embeddings.
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer encoder blocks.
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Instead of a class token, average pool the tokens to obtain a global representation.
        x = x.mean(dim=1)  # Resulting shape: (B, emb_dim)
        return x