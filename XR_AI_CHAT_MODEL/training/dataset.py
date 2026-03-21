
import torch
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("XR_AI_CHAT_MODEL/tokenizer/tokenizer.model")

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, file, max_len=128):
        with open(file, "r", encoding="utf-8") as f:
            self.lines = f.readlines()

        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]

        tokens = sp.encode(text)

        tokens = tokens[:self.max_len]

        x = torch.tensor(tokens[:-1])
        y = torch.tensor(tokens[1:])

        return x, y
