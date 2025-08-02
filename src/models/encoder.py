import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
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
        kernel_sizes = [8, 4, 3],
        ln_sizes = [20, 9, 7],
        strides = [4, 2, 1],
        in_channels = 4,
        input_size = 84,
        device='cpu'
    ):
        super().__init__()
        act_ = get_act_fn_clss(activation_fn)
        cnn = []
        for out_channel, kernel_size, stride, ln_size in zip(cnn_channels, kernel_sizes, strides, ln_sizes):
            cnn.append(
                layer_init(
                    nn.Conv2d(in_channels, out_channel, kernel_size, stride=stride, device=device),
                )
            )
                        
            if use_ln:
                cnn.append(
                    nn.LayerNorm(
                        [out_channel, ln_size, ln_size], device=device
                    )
                )
                
            cnn.append(
                act_()
            )
            
            output_size = (input_size - kernel_size) / stride + 1
            input_size = output_size
            in_channels = out_channel
            
        cnn.append(
            nn.Flatten()
        )
        self.cnn = nn.Sequential(*cnn)
        
        self.output_size = out_channel * output_size * output_size
        
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
        shape=(4, 84, 84),
        device='cpu'
    ):
        super().__init__()
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
        self.output_size = np.prod(shape)
        
    def forward(self, x):
        return self.cnn(x)
    
################################### HADA-MAX CNN ###################################

class _HadaMaxBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, ln_size, pk, ps, pp, use_ln, activation_fn, device):
        super().__init__()
        self.use_ln = use_ln
        self.act_ = get_act_fn_functional(activation_fn)
        # two parallel convs
        self.conv_a = layer_init(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, device=device)
        )
        self.conv_b = layer_init(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, device=device)
        )

        norm_shape = [out_ch, ln_size, ln_size]
        self.ln_a = nn.LayerNorm(norm_shape, device=device)
        self.ln_b = nn.LayerNorm(norm_shape, device=device)
        
        # pooling
        self.pool = nn.MaxPool2d(kernel_size=pk, stride=ps, padding=pp if pp is not None else 0)

    def forward(self, x):
        a = self.act_(self.ln_a(self.conv_a(x)))
        b = self.act_(self.ln_b(self.conv_b(x)))
        return self.pool(a * b)

class HadaMaxCNN(nn.Module):
    def __init__(
        self,
        cnn_channels,
        use_ln=True,
        activation_fn="gelu",
        kernel_sizes = [8, 4, 3],
        ln_sizes     = [85, 22, 11],
        strides      = [1, 1, 1],
        paddings     = [4, 2, 1],
        pool_kernels = [4, 2, 3],
        pool_strides = [4, 2, 1],
        pool_paddings = [None, None, 1],
        in_channels  = 4,
        input_size   = 84,
        device='cpu'
    ):
        super().__init__()
        blocks = []
        size = input_size
        ic   = in_channels

        # build each HadaMax block
        for i, oc in enumerate(cnn_channels):
            k  = kernel_sizes[i]
            s  = strides[i]
            p  = paddings[i]
            ln = ln_sizes[i]
            pk = pool_kernels[i]
            ps = pool_strides[i]
            pp = pool_paddings[i]

            blocks.append(_HadaMaxBlock(ic, oc, k, s, p, ln, pk, ps, pp, use_ln, activation_fn, device))

            # update spatial size: conv then pool
            size = (size + 2*p - k) // s + 1
            pad  = (pk - ps) // 2
            size = (size + 2*pad - pk) // ps + 1
            ic = oc

        # flatten at end
        blocks.append(nn.Flatten())

        # the Sequential CNN
        self.cnn = nn.Sequential(*blocks)
        # final flat size
        self.output_size = ic * size * size

    def forward(self, x):
        return self.cnn(x)