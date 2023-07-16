from pathlib import Path

import torch
from torch.optim import AdamW

from poushkine.dataloader import Dataloader
from poushkine.model import BigramModel
from poushkine.tokenizer import Text2Vec
from poushkine.trainer import Trainer


if __name__ == "__main__":
    text = Path("poems.txt").read_text()

    # parameters
    train_size = 0.9
    batch_size = 4
    ctx_length = 8
    num_embed = 36
    max_iter = 10000
    lr = 1e-3
    dropout = 0.2
    num_blocks = 6
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    attention_num_heads = 6

    t2v = Text2Vec()
    t2v.fit(text)

    dl = Dataloader(
        t2v.encode(text),
        train_size=train_size,
        batch_size=batch_size,
        context_length=ctx_length,
    )
    xb, yb = dl.get_batch()

    m = BigramModel(
        vocab_size=t2v.vocab_size,
        block_size=ctx_length,
        num_embed=num_embed,
        attention_num_heads=attention_num_heads,
        dropout=dropout,
        num_blocks=num_blocks,
        device=device,
    )
    _, loss = m(xb, yb)

    t = Trainer(m, dl, AdamW, max_iterations=max_iter, lr=lr)
    t.train()

    _, loss = m(xb, yb)
    print(t2v.decode(m.generate(1000).tolist()))
