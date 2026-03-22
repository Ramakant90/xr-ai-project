
import sentencepiece as spm

class Config:
    def __init__(self):
        # 🔥 Load tokenizer to get vocab dynamically
        sp = spm.SentencePieceProcessor()
        sp.load("XR_AI_CHAT_MODEL/tokenizer/tokenizer.model")

        # 🔥 IMPORTANT: auto-sync vocab
        self.vocab_size = sp.get_piece_size()

        # Model size (small but stable)
        self.n_layers = 4
        self.n_heads = 4
        self.embed_dim = 256
        self.ff_dim = 512

        # Sequence
        self.max_seq_len = 128

        # 🔥 Special tokens (must be >= 0)
        self.pad_id = 0      # safe pad
        self.bos_id = 1
        self.eos_id = 2
