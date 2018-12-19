from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from data import get_iterator
from models import CharCNN, CNNClassifier
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


class CharCNNParams:
    """Parameters for a CharCNN classifier."""

    def __init__(self, dropout: float, epochs: int, max_norm: float) -> None:
        self.dropout = dropout
        self.epochs = epochs
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
    progbar: int = -1,
    max_norm: float = 0,
    validation_set: Optional[Tuple[Dataset, int, int]] = None,
    use_dist: bool = False,
    use_chars: bool = False,
) -> List[float]:
    """Train a Pytorch model.

    :param model: A Pytorch model.
    :param optimizer: A Pytorch optimizer for the model.
    :param dataloader: An iterator returning batches.
    :param epochs: The number of epochs to train for.
    :param gpu: If true, train on the gpu. Otherwise use the cpu.
    :param early_stopping: If 0, don't use early stopping. If positive, stop
        after that many epochs have yielded no improvement in loss.
    :param progbar: Display a progress at the given position, or none if < 0.
    :param max_norm: Value to clip each weight vector's L2 norm at. If 0, no
        clipping is done.
    :param validation_set: Optional verification set to use for early stopping.
    :param use_dist: Only use the indices of the clusterlabels rather than a
        categorical vector.
    :param use_chars: Only the CharCNN dataset.
    :returns: The value of the model's loss function at every epoch.
    """
    model.cuda() if gpu else model.cpu()
    model.train()

    epoch_losses: List[float] = []
    best_params: Dict[str, Any] = {}
    best_loss = 99999

    f1_scores: List[float] = []
    best_f1 = 0.0

    stopping_counter = 0
    if progbar >= 0:
        t = trange(epochs, desc="Training", position=progbar)
    else:
        t = range(epochs)
    for i in t:
        epoch_loss = 0.0

        for batch in dataloader:
            batch.to_tensors()
            if gpu:
                batch.to_gpu()

            X = batch.X_chars if use_chars else batch.X_words
            c = batch.clusters_gmm if use_dist else batch.clusters_kmeans
            y = batch.label

            model.train()
            y_pred = model(X, c)
            loss = model.loss(y_pred.view(-1), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if max_norm > 0:
                clip_norms(model, max_norm)

            epoch_loss += loss.item()
            batch.to_cpu()

        loss = epoch_loss / len(dataloader)
        epoch_losses.append(loss)

        # check if the model is the best yet
        if validation_set:
            if i % 5 == 0:
                ds, wordlen, charlen = validation_set
                f1 = get_scores(model, wordlen, charlen, ds, gpu, use_dist, use_chars)["F1"]
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

        if early_stopping > 0:
            if stopping_counter >= early_stopping:
                break

        # update the progress bar
        if progbar >= 0:
            if validation_set:
                f1_delta = (
                    f1_scores[-1] - f1_scores[-2] if len(f1_scores) > 1 else 0
                )
                t.set_postfix({"f1": f1_scores[-1], "Δf1": f1_delta})
            else:
                loss_delta = (
                    epoch_losses[-1] - epoch_losses[-2] if len(epoch_losses) > 1 else 0
                )
                t.set_postfix({"loss": loss, "Δloss": loss_delta})
    else:
        # warn if the for-loop didn't break
        print("Warning: did not stop early")

    if progbar >= 0:
        t.close()

    model.load_state_dict(best_params)
    model.eval()
    return epoch_losses if not validation_set else f1_scores


def train_BoW(
    dataset: Dataset, vocab: Dict[str, int], ngram_range: Tuple[int, int] = (1, 1)
) -> Tuple[SVC, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        token_pattern=r"\w+|[^\w\s]", ngram_range=ngram_range
    )
    model = SVC(probability=True)

    samples = [
        " ".join(
            [vocab.idx_to_token.get(idx, "NULL") for row in entry.X_words for idx in row]
        )
        for entry in dataset
    ]
    labels = [entry.label for entry in dataset]
    X = vectorizer.fit_transform(samples)

    model.fit(X, labels)
    return model, vectorizer


def setup_and_train(
    params: Union[CNNParams, CharCNNParams],
    model_fn: Callable[[nn.Module], nn.Module],
    optim_fn: Callable[[Any], torch.optim.Optimizer],
    dataset: Dataset,
    epochs: int = 100,
    batch_size: int = 32,
    gpu: bool = True,
    early_stopping: int = 0,
    progbar: int = -1,
    max_norm: float = 0,
    validation_set: Optional[Dataset] = None,
    use_dist: bool = False,
) -> Tuple[nn.Module, List[float], int, int, bool]:
    """Create a neural network model and train it."""
    recurrent_model: nn.Module
    argdict: Dict[str, Any]
    use_chars: bool
    seqlen: int
    data, wordlen, charlen = get_iterator(dataset, None, None, batch_size=batch_size)
    if isinstance(params, CNNParams):
        seqlen = wordlen
        argdict = {
            "input_size": len(dataset.word_vocab.token_to_idx) + 1,
            "seq_len": seqlen,
            "embed_size": params.embed_size,
            "filters": params.filters,
        }
        recurrent_model = CNNClassifier(**argdict)
        use_chars = False
    elif isinstance(params, CharCNNParams):
        seqlen = charlen
        argdict = {
            "input_size": len(dataset.char_vocab.token_to_idx) + 1,
            "seq_len": seqlen,
        }
        recurrent_model = CharCNN(**argdict)
        use_chars = True

    model = model_fn(recurrent_model)
    optimizer = optim_fn(model.parameters())
    valid = (validation_set, wordlen, charlen) if validation_set else None
    losses = train(
        model,
        optimizer,
        data,
        epochs,
        gpu,
        early_stopping,
        progbar,
        max_norm,
        valid,
        use_dist,
        use_chars,
    )

    return model, losses, wordlen, charlen, use_chars
