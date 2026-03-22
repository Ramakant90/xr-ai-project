
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # 🔥 Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # 🔥 Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # 🔥 Final layer norm
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # 🔥 Output head
        self.head = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x):
        B, T = x.size()

        # 🔥 position ids
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)

        # 🔥 embeddings
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb

        # 🔥 transformer layers
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits


# ================================
# 🔥 Transformer Block
# ================================

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadAttention(config)

        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ================================
# 🔥 Multi-Head Attention
# ================================

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.embed_dim % config.n_heads == 0, "embed_dim must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.out = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # reshape for multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 🔥 causal mask (important for GPT)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        out = attn @ v

        # reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)


# ================================
# 🔥 Feed Forward
# ================================

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.ReLU(),
            nn.Linear(config.ff_dim, config.embed_dim)
        )

    def forward(self, x):
        return self.net(x)
