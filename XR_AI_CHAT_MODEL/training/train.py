
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from model.transformer import MiniGPT
from model.config import Config
from training.dataset import ChatDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

dataset = ChatDataset("XR_AI_CHAT_MODEL/data/processed/train.txt")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = MiniGPT(config).to(device)

# 🔥 LOWER LR (important)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 🔥 Ignore padding
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.pad_id)

EPOCHS = 10

for epoch in range(EPOCHS):
    total_loss = 0
    steps = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # 🔥 SAFETY CHECK (token bounds)
        if x.min() < 0 or x.max() >= config.vocab_size:
            print("❌ Invalid token detected, skipping batch")
            continue

        logits = model(x)

        loss = loss_fn(
            logits.view(-1, config.vocab_size),
            y.view(-1)
        )

        # 🔥 NaN guard
        if torch.isnan(loss):
            print("❌ NaN loss detected, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()

        # 🔥 Gradient clipping (VERY IMPORTANT)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        steps += 1

    if steps > 0:
        avg_loss = total_loss / steps
    else:
        avg_loss = float("nan")

    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")

os.makedirs("XR_AI_CHAT_MODEL/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "XR_AI_CHAT_MODEL/checkpoints/model.pt")

print("✅ Training Complete")
