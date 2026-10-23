"""CNN de texte : convolutions de largeur 3, 4 et 5, puis maximum temporel."""

import os

import numpy as np
import torch
from torch import nn


class TextNetwork(nn.Module):
    def __init__(self, name, input_size, width=64, dropout=0.3):
        super().__init__()
        if name != "cnn":
            raise ValueError("Seule l'architecture CNN est prise en charge")
        self.name = name
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, width, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(width, 32, kernel_size=k) for k in (3, 4, 5)])
        self.output = nn.Linear(96, 8)

    def forward(self, x):
        lengths = x.ne(0).sum(1)
        embeddings = self.embedding(x).transpose(1, 2)
        features = []
        for conv in self.convs:
            activations = torch.relu(conv(embeddings))
            valid = (lengths - conv.kernel_size[0] + 1).clamp(min=1)
            positions = torch.arange(activations.shape[2], device=x.device)
            activations = activations.masked_fill(
                positions[None, None, :] >= valid[:, None, None], float("-inf")
            )
            features.append(activations.amax(2))
        return self.output(self.dropout(torch.cat(features, dim=1)))


class NeuralModel:
    kind = "probability"

    def __init__(self, network, batch_size=32):
        self.network = network.cpu().eval()
        self.batch_size = batch_size

    def scores(self, x):
        self.network.eval()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(x), self.batch_size):
                ids = torch.as_tensor(x[start : start + self.batch_size], dtype=torch.long)
                batches.append(self.network(ids).sigmoid().numpy())
        return np.concatenate(batches)


def fit_cnn(encoder, rows, y, config, device="cpu"):
    """Budget fixe d'époques : aucune validation ni donnée test ne règle le réseau."""
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA indisponible dans cet environnement PyTorch")
    if config.get("max_tokens", 512) < 5:
        raise ValueError("Le CNN nécessite au moins cinq positions")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("jobs", 2))
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    width = config.get("width", 64)
    network = TextNetwork("cnn", len(encoder.vocabulary_) + 2, width, config.get("dropout", 0.3)).to(device)
    initialization = config.get("initialization", "random")
    if initialization != "random":
        matrix = encoder.initialize(rows, width, initialization, seed)
        with torch.no_grad():
            network.embedding.weight.copy_(torch.as_tensor(matrix, device=device))
    positive = y.sum(0)
    if (positive == 0).any() or (positive == len(y)).any():
        raise ValueError("Chaque tag doit avoir des positifs et des négatifs")
    weights = torch.as_tensor((len(y) - positive) / positive, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.get("lr", 0.001), weight_decay=config.get("weight_decay", 0.0001)
    )
    x = encoder.transform(rows)
    batch_size = config.get("batch_size", 32)
    rng, history = np.random.default_rng(seed), []
    for epoch in range(1, config["epochs"] + 1):
        network.train()
        total_loss = 0.0
        indices = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = indices[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            ids = torch.as_tensor(x[batch], dtype=torch.long, device=device)
            targets = torch.as_tensor(y[batch], dtype=torch.float32, device=device)
            loss = criterion(network(ids), targets)
            if not torch.isfinite(loss):
                raise ValueError("Loss non finie")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        history.append({"epoch": epoch, "train_loss": total_loss / len(y)})
        print(f"  époque {epoch}/{config['epochs']} : loss={total_loss / len(y):.4f}", flush=True)
    return NeuralModel(network, batch_size), history
