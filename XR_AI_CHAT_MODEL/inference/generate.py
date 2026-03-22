
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


# 🔥 Generate function (improved)
def generate(prompt, max_new_tokens=50, temperature=0.8):
    tokens = sp.encode(prompt)
    tokens = tokens[-config.max_seq_len:]

    for _ in range(max_new_tokens):
        x = torch.tensor(tokens).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)

        logits = logits[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).item()

        tokens.append(next_token)

        # 🔥 stop condition (important)
        if next_token == 2:  # EOS token
            break

    return sp.decode(tokens)


# 🔥 Chat loop
print("🤖 XR AI Chatbot Ready (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    prompt = f"User: {user_input}\nAssistant:"

    response = generate(prompt, max_new_tokens=60, temperature=0.8)

    # 🔥 clean output
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1]

    print("AI:", response.strip(), "\n")
