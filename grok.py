import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from data import Tokenizer
from data import BinaryDivisionModDataset
from data import generate_data
from model import TransformerModel

seed = 42
log_wandb = True

try:
    import wandb
    wandb.init(project="grokking")
except ImportError:
    log_wandb = False


def train(model, train_loader, optimizer, loss_f, device):
    model.train()
    total_loss = 0
    total_acc = 0

    for batch, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        # output is (batch, seq_len, vocab_size), y is (batch, seq_len)
        # in the paper - "calculated loss and accuracy only on the answer part of the equation"
        # answer idx is 5.
        # the mask at idx 4 is [0, 0, 0, 0, 
        answer_pred = output[:, 4, :]
        answer_targets = y[:, 5]

        optimizer.zero_grad()
        loss = loss_f(answer_pred, answer_targets)
        loss.backward()
        optimizer.step()
        final_ans = answer_pred.argmax(dim=1)

        total_loss += loss.item()
        total_acc += (final_ans == answer_targets).sum().item() / y.size(0)
    
    return total_loss / len(train_loader), total_acc / len(train_loader)

def validate(model, val_loader, loss_f, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            answer_pred = output[:, 4, :]
            answer_targets = y[:, 5]
            loss = loss_f(answer_pred, answer_targets)
            total_loss += loss.item()
            final_ans = answer_pred.argmax(dim=1)
            total_acc += (final_ans == answer_targets).sum().item() / y.size(0)

    return total_loss / len(val_loader), total_acc / len(val_loader)


def main():

    # set seed for all
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer()
    full_dataset = generate_data(tokenizer)

    np.random.shuffle(full_dataset)
    split_idx = len(full_dataset) // 2

    train_dataset = BinaryDivisionModDataset(tokenizer, full_dataset[:split_idx])
    val_dataset = BinaryDivisionModDataset(tokenizer, full_dataset[split_idx:])
    
    bs = min(512, len(train_dataset) // 2) # A.1.2 from the paper
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    vocab_size = len(tokenizer.symbol_to_id)
    model = TransformerModel(vocab_size=vocab_size).to(device)

    count_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # in the paper they mention "about 4x10e5"
    # Seems critical, since we are in the game of overfitting
    print(f"Model has {count_params} trainable parameters")

    # A.1.2 from the paper, hopefully got it right
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1, betas=(0.9, 0.98))
    loss = nn.CrossEntropyLoss()
    
    n_epochs = 100000
    model_metrics = []

    for epoch in range(n_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, loss, device)
        val_loss, val_accuracy = validate(model, val_loader, loss, device)
        metrics = {
            # there are len(train_loader) batches per epoch (opt steps)
            "opt_step": (epoch + 1) * len(train_loader),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_accuracy
        }
        model_metrics.append(metrics)

        if log_wandb:
            wandb.log(metrics)

        if (epoch + 1) % 10 == 0:
            print(
                f"""
                Opt step {(epoch + 1) * len(train_loader)}, 
                Train Loss: {train_loss}, 
                Train Acc: {train_acc * 100}, 
                Val Loss: {val_loss}, 
                Val Acc: {val_accuracy * 100}
                """
            )
    
    return pd.DataFrame(model_metrics)


if __name__ == "__main__":
    model_metrics = main()
    model_metrics.to_csv("model_metrics.csv", index=False)

