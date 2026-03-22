
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import sentencepiece as spm

from model.transformer import MiniGPT
from model.config import Config

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("XR_AI_CHAT_MODEL/tokenizer/tokenizer.model")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
config = Config()
model = MiniGPT(config).to(device)

model.load_state_dict(torch.load("XR_AI_CHAT_MODEL/checkpoints/model.pt", map_location=device))
model.eval()


def generate(prompt, max_new_tokens=50, temperature=0.7, top_k=40):
    tokens = sp.encode(prompt)
    tokens = tokens[-config.max_seq_len:]

    for _ in range(max_new_tokens):
        x = torch.tensor(tokens).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)

        logits = logits[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)

        # 🔥 Top-K sampling
        values, indices = torch.topk(probs, top_k)
        probs = values / values.sum()

        next_token = indices[torch.multinomial(probs, 1)].item()
        tokens.append(next_token)

        if next_token == config.eos_id:
            break

    return sp.decode(tokens)


# 🔥 TEST CASES (manual input Kaggle friendly)
tests = [
    "hello",
    "tum kaun ho?",
    "namaste",
    "भारत की राजधानी क्या है?",
    "what is AI?"
]

print("\n🤖 XR AI Chatbot Test\n")

for user_input in tests:
    prompt = f"User: {user_input}\nAssistant:"
    response = generate(prompt)

    if "Assistant:" in response:
        response = response.split("Assistant:")[-1]

    print("User:", user_input)
    print("AI:", response.strip(), "\n")
