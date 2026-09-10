"""Entrena y evalúa la GRU global a partir del corpus de IDs."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from neural_model import GRUPictogramPredictor

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "dataset_frases_ids.txt"
CATALOG_PATH = ROOT / "pictogram_catalog.json"
ARTIFACT_PATH = ROOT / "gru_model"
SEED = 20260910


class PrefixDataset(Dataset):
    def __init__(self, phrases, token_by_id, start_token, end_id, max_context):
        self.samples = []
        for phrase in phrases:
            encoded = [start_token] + [token_by_id[value] for value in phrase]
            for position, target in enumerate(phrase + [end_id]):
                self.samples.append((encoded[:position + 1][-max_context:], target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def make_batch(rows, output_index):
    sequences = [torch.tensor(row[0], dtype=torch.long) for row in rows]
    return (pad_sequence(sequences, batch_first=True, padding_value=0),
            torch.tensor([len(row) for row in sequences], dtype=torch.long),
            torch.tensor([output_index[row[1]] for row in rows], dtype=torch.long))


def evaluate(model, loader, device, end_index):
    model.eval()
    correct = {1: 0, 3: 0, 5: 0}
    total = 0
    loss_samples = 0
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, lengths, targets in loader:
            inputs, lengths, targets = inputs.to(device), lengths.to(device), targets.to(device)
            logits = model(inputs, lengths)
            loss_sum += nn.functional.cross_entropy(logits, targets).item() * len(targets)
            loss_samples += len(targets)
            top = logits.topk(5, dim=1).indices
            visible = targets != end_index
            for size in correct:
                correct[size] += (top[visible, :size] == targets[visible, None]).any(dim=1).sum().item()
            total += visible.sum().item()
    return {"loss": round(loss_sum / loss_samples, 6),
            "top1": round(correct[1] / total, 6),
            "top3": round(correct[3] / total, 6),
            "top5": round(correct[5] / total, 6), "samples": total}


def main(epochs: int, batch_size: int):
    random.seed(SEED)
    torch.manual_seed(SEED)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["pictograms"]
    pictogram_ids = sorted(int(item["id"]) for item in catalog)
    phrases = [list(map(int, line.split())) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    random.shuffle(phrases)
    train_end, validation_end = int(len(phrases) * .8), int(len(phrases) * .9)
    splits = {"train": phrases[:train_end], "validation": phrases[train_end:validation_end],
              "test": phrases[validation_end:]}

    token_by_id = {identifier: index + 1 for index, identifier in enumerate(pictogram_ids)}
    start_token, end_id, max_context = len(token_by_id) + 1, -1, 8
    output_ids = pictogram_ids + [end_id]
    output_index = {identifier: index for index, identifier in enumerate(output_ids)}
    datasets = {key: PrefixDataset(value, token_by_id, start_token, end_id, max_context)
                for key, value in splits.items()}
    loaders = {key: DataLoader(value, batch_size=batch_size, shuffle=key == "train", num_workers=0,
                               collate_fn=lambda rows: make_batch(rows, output_index))
               for key, value in datasets.items()}

    model = GRUPictogramPredictor(len(token_by_id) + 2, len(output_ids))
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=.03)
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, lengths, targets in loaders["train"]:
            inputs, lengths, targets = inputs.to(device), lengths.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs, lengths), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, loaders["validation"], device, output_index[end_id])
        print(json.dumps({"epoch": epoch, "validation": metrics}), flush=True)
        if metrics["loss"] < best_loss:
            best_loss, stale = metrics["loss"], 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 3:
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], device, output_index[end_id])
    ARTIFACT_PATH.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_PATH / "model.pt")
    config = {"version": 1, "architecture": "embedding-gru-softmax", "embedding_dim": 64,
              "hidden_dim": 128, "layers": 1, "dropout": .15, "max_context": max_context,
              "vocabulary_size": len(token_by_id) + 2, "token_by_id": token_by_id,
              "start_token": start_token, "end_id": end_id, "output_ids": output_ids,
              "split": {f"{key}_phrases": len(value) for key, value in splits.items()},
              "target_coverage": len(Counter(target for _, target in datasets["train"].samples)),
              "metrics": test_metrics, "seed": SEED}
    (ARTIFACT_PATH / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_metrics, "artifact": str(ARTIFACT_PATH)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    arguments = parser.parse_args()
    main(arguments.epochs, arguments.batch_size)
