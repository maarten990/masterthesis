"""
Train the neural networks.

Usage:
train.py <paramfile> <files>... [options]
train.py (-h | --help)

Options:
    -h --help                      Show this screen
    --with_labels                  Include computed cluster labels.
    -b <size> --batch_size=<size>  The batch size used for training.
"""


from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from docopt import docopt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import yaml

from data import get_iterator, to_cpu, to_gpu, to_tensors
from models import LSTMClassifier, CNNClassifier
from evaluate import get_scores


class CNNParams:
    """Parameters for a CNNClassifier."""

    def __init__(
        self,
        embed_size: int,
        dropout: float,
        epochs: int,
        filters: List[Tuple[int, int]],
        num_layers: int,
        max_norm: float,
    ) -> None:
        self.embed_size = embed_size
        self.dropout = dropout
        self.epochs = epochs
        self.filters = filters
        self.num_layers = num_layers
        self.max_norm = max_norm


class RNNParams:
    """Parameters for an LSTMClassifier."""

    def __init__(
        self,
        embed_size: int,
        dropout: float,
        epochs: int,
        num_layers: int,
        hidden_size: int,
        max_norm: float,
    ) -> None:
        self.embed_size = embed_size
        self.dropout = dropout
        self.epochs = epochs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_norm = max_norm


def clip_norms(model: nn.Module, max_val: float, eps: float = 1e-8) -> None:
    """
    Clip the L2 norm of each parameter in a model to a maximum value.
    :param model: The pytorch model to clip.
    :param max_val: The maximum value for the L2 norm of the parameters.
    :param eps: An additional term added to prevent division by zero.
    """
    for name, param in model.named_parameters():
        if "bias" not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    epochs: int = 100,
    gpu: bool = True,
    early_stopping: int = 0,
    progbar: bool = False,
    max_norm: float = 0,
    validation_set: Optional[Tuple[Dataset, List[int]]] = None,
) -> List[float]:
    """Train a Pytorch model.

    :param model: A Pytorch model.
    :param optimizer: A Pytorch optimizer for the model.
    :param dataloader: An iterator returning batches.
    :param epochs: The number of epochs to train for.
    :param gpu: If true, train on the gpu. Otherwise use the cpu.
    :param early_stopping: If 0, don't use early stopping. If positive, stop
        after that many epochs have yielded no improvement in loss.
    :param progbar: Display a progress bar or not.
    :param max_norm: Value to clip each weight vector's L2 norm at. If 0, no
        clipping is done.
    :param validation_set: Optional verification set to use for early stopping.
    :returns: The value of the model's loss function at every epoch.
    """
    model.train()
    model.cuda() if gpu else model.cpu()

    epoch_losses: List[float] = []
    best_params: Dict[str, Any] = {}
    best_loss = 99999

    f1_scores: List[float] = []
    best_f1 = 0.0

    stopping_counter = 0
    if progbar:
        t = trange(epochs, desc="Training")
    else:
        t = range(epochs)
    for _ in t:
        epoch_loss = 0.0

        for batch in dataloader:
            data = to_tensors(batch)
            if gpu:
                data = to_gpu(data)

            for _, d in data.items():
                X = d["data"]
                c = d["cluster_data"]
                y = d["label"]

                y_pred = model(X, c)
                loss = model.loss(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if max_norm > 0:
                    clip_norms(model, max_norm)

                epoch_loss += loss.item()

            to_cpu(data)

        loss = epoch_loss / len(dataloader)
        epoch_losses.append(loss)

        # check if the model is the best yet
        if validation_set:
            ds, buckets = validation_set
            f1 = get_scores(model, buckets, ds, gpu)["F1"]
            f1_scores.append(f1)
            if f1 > best_f1:
                best_f1 = f1
                best_params = model.state_dict()
                stopping_counter = 0
            else:
                stopping_counter += 1

        else:
            if loss < best_loss:
                best_loss = loss
                best_params = model.state_dict()
                stopping_counter = 0
            else:
                stopping_counter += 1

        if early_stopping:
            if stopping_counter >= early_stopping:
                break

        # update the progress bar
        if progbar:
            loss_delta = (
                epoch_losses[-1] - epoch_losses[-2] if len(epoch_losses) > 1 else 0
            )
            t.set_postfix({"loss": loss, "Δloss": loss_delta})

    if progbar:
        t.close()

    model.load_state_dict(best_params)
    model.eval()
    return epoch_losses if not validation_set else f1_scores


def train_BoW(
    dataset: Dataset, vocab: Dict[str, int], ngram_range: Tuple[int, int] = (1, 1)
) -> Tuple[SVC, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        vocabulary=vocab, token_pattern=r"\w+|[^\w\s]", ngram_range=ngram_range
    )
    model = SVC(probability=True)

    samples = [list(entry.values())[0]["data"] for entry in dataset]
    labels = [list(entry.values())[0]["label"] for entry in dataset]
    print(samples[0])
    X = vectorizer.fit_transform(samples)

    model.fit(X, labels)
    return model, vectorizer


def setup_and_train(
    params: Union[CNNParams, RNNParams],
    model_fn: Callable[[nn.Module], nn.Module],
    optim_fn: Callable[[Any], torch.optim.Optimizer],
    dataset: Dataset,
    epochs: int = 100,
    batch_size: int = 32,
    gpu: bool = True,
    early_stopping: int = 0,
    progbar: bool = True,
    max_norm: float = 0,
    validation_set: Optional[Dataset] = None,
) -> Tuple[nn.Module, List[float], List[int]]:
    """Create a neural network model and train it."""
    recurrent_model: nn.Module
    argdict: Dict[str, Any]
    if isinstance(params, RNNParams):
        buckets = [5, 10, 25, 50, 75, 100]
        argdict = {
            "input_size": len(dataset.vocab.token_to_idx) + 1,
            "embed_size": params.embed_size,
            "hidden_size": params.hidden_size,
            "num_layers": params.num_layers,
            "dropout": params.dropout,
        }
        recurrent_model = LSTMClassifier(**argdict)
        data, _ = get_iterator(dataset, buckets=buckets, batch_size=batch_size)
    elif isinstance(params, CNNParams):
        buckets = None
        data, buckets = get_iterator(dataset, buckets=buckets, batch_size=batch_size)
        argdict = {
            "input_size": len(dataset.vocab.token_to_idx) + 1,
            "seq_len": buckets[0],
            "embed_size": params.embed_size,
            "filters": params.filters,
            "dropout": params.dropout,
            "num_layers": params.num_layers,
        }
        recurrent_model = CNNClassifier(**argdict)

    model = model_fn(recurrent_model)
    optimizer = optim_fn(model.parameters())
    losses = train(
        model, optimizer, data, epochs, gpu, early_stopping, progbar, max_norm, (validation_set, buckets)
    )

    return model, losses, buckets


def parse_params(params: Dict[str, Any]) -> Union[CNNParams, RNNParams, None]:
    """Parse a parameter dictinary and ensure it's valid.

    :param params: The parameter dict.
    :returns: The parsed parameters.
    """
    tp = params["type"]
    del params["type"]

    constructor: Union[Type[CNNParams], Type[RNNParams]]
    if tp == "cnn":
        keys = ["embed_size", "filters", "dropout", "epochs", "num_layers"]
        constructor = CNNParams
    elif tp == "rnn":
        keys = ["embed_size", "hidden_size", "num_layers", "dropout", "epochs"]
        constructor = RNNParams
    else:
        print("Error: only cnn or rnn allowed as type.")
        return None

    for key in keys:
        if key not in params.keys():
            print(f"Error: missing key {key}")
            return None

    return constructor(**params)


def main():
    args = docopt(__doc__)
    paramfile = args["<paramfile>"]
    files = args["<files>"]
    with_labels = args["--with_labels"]
    batch_size = args["--batch_size"]

    with open(paramfile, "r") as f:
        params = yaml.load(f)
        p = parse_params(params)

        losses = setup_and_train(p, with_labels, files, 100, int(batch_size))
        print(losses)


if __name__ == "__main__":
    main()
