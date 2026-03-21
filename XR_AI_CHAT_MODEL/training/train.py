
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

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)

        loss = loss_fn(
            logits.view(-1, config.vocab_size),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")

torch.save(model.state_dict(), "XR_AI_CHAT_MODEL/checkpoints/model.pt")
