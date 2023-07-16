from pathlib import Path

import torch
from torch.optim import AdamW

from poushkine.dataloader import Dataloader
from poushkine.model import BigramModel
from poushkine.tokenizer import Text2Vec
from poushkine.trainer import Trainer


if __name__ == "__main__":
    text = Path("poems.txt").read_text(encoding="utf-8")

    # parameters
    train_size = 0.9
    batch_size = 64
    ctx_length = 256
    num_embed = 384
    max_iter = 5000
    eval_interval = 500
    lr = 3e-4
    dropout = 0.2
    num_blocks = 6
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    attention_num_heads = 6

    print(f"Using {device} device.")

    t2v = Text2Vec()
    t2v.fit(text)

    dl = Dataloader(
        t2v.encode(text),
        train_size=train_size,
        batch_size=batch_size,
        context_length=ctx_length,
    )

    m = BigramModel(
        vocab_size=t2v.vocab_size,
        block_size=ctx_length,
        num_embed=num_embed,
        attention_num_heads=attention_num_heads,
        dropout=dropout,
        num_blocks=num_blocks,
        device=device,
    ).to(device)

    xb, yb = dl.get_batch(device=device)
    _, loss = m(xb, yb)

    t = Trainer(m, dl, AdamW, max_iterations=max_iter, eval_interval=eval_interval, lr=lr)
    t.train()

    torch.save(m.state_dict(), "./model.pt")

    _, loss = m(xb, yb)
    print(t2v.decode(m.generate(1000).tolist()))
