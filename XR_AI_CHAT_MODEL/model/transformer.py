
import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_size,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        self.fc = nn.Linear(config.embed_size, config.vocab_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T = x.shape

        pos = torch.arange(0, T, device=x.device).unsqueeze(0)

        x = self.token_embedding(x) + self.pos_embedding(pos)
        x = self.dropout(x)

        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        x = self.transformer(x, mask=mask)

        logits = self.fc(x)

        return logits
