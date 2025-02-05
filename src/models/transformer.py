import torch
from torch import nn
from einops import rearrange
from IPython import embed

class PositionalEncoding(nn.Module):
    def __init__(self, dim, min_timescale=2.0, max_timescale=1e4, device="cuda"):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale, device=device)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)
        self.device = device

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.0, device=self.device)
        sinusoidal_inp = rearrange(seq, "n -> n ()") * rearrange(self.inv_freqs, "d -> () d")
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device="cuda"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert self.head_size * num_heads == embed_dim, "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False, device=device)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False, device=device)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False, device=device)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim, device=device)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Dot-product
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))  # -inf causes NaN

        # Normalize energy values and apply softmax to retrieve the attention scores
        attention = torch.softmax(
            energy / (self.embed_dim ** (1 / 2)), dim=3
        )  # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.num_heads * self.head_size)

        return self.fc_out(out), attention


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, device="cuda"):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads, device)
        self.layer_norm_q = nn.LayerNorm(dim, device=device)
        self.norm_kv = nn.LayerNorm(dim, device=device)
        self.layer_norm_attn = nn.LayerNorm(dim, device=device)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU()).to(device)

    def forward(self, value, key, query, mask):
        # Pre-layer normalization (post-layer normalization is usually less effective)
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value  # K = V -> self-attention
        attention, attention_weights = self.attention(value, key, query_, mask)  # MHA
        x = attention + query  # Skip connection
        x_ = self.layer_norm_attn(x)  # Pre-layer normalization
        forward = self.fc_projection(x_)  # Forward projection
        out = forward + x  # Skip connection
        return out, attention_weights


class Transformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, max_episode_steps, positional_encoding, device="cuda"):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.positional_encoding = positional_encoding
        if positional_encoding == "absolute":
            self.pos_embedding = PositionalEncoding(dim, device=device)
        elif positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(max_episode_steps, dim), device=device)
        self.transformer_layers = nn.ModuleList([TransformerLayer(dim, num_heads, device) for _ in range(num_layers)])

    def forward(self, x, memories, mask, memory_indices):
        # Add positional encoding to every transformer layer input
        if self.positional_encoding == "absolute":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
        elif self.positional_encoding == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)

        # Forward transformer layers and return new memories (i.e. hidden states)
        out_memories = []
        for i, layer in enumerate(self.transformer_layers):
            out_memories.append(x.detach())
            x, attention_weights = layer(
                memories[:, :, i], memories[:, :, i], x.unsqueeze(1), mask
            )  # args: value, key, query, mask
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x, torch.stack(out_memories, dim=1)