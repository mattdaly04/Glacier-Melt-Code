"""
glacier_melt.train
==================
Training loops, tensor construction, and checkpoint utilities.

All PyTorch models are trained with ``BCEWithLogitsLoss`` (which combines
sigmoid + BCE in a numerically stable way) and the Adam optimiser with a
``ReduceLROnPlateau`` scheduler. A custom ``GPUDataLoader`` keeps all tensors
on GPU memory throughout training, avoiding CPU↔GPU transfer overhead on
each batch — critical for the 12.2M-pixel training set.

Random Forest models are trained via ``train_rf``, using a stratified
2M-pixel subsample due to CPU memory constraints of the sklearn RF
implementation.

The neighbour table (for CNN patch construction) is built using a cKDTree
over pixel coordinates, using the median pixel spacing to determine grid
alignment and filter out spurious neighbours.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from glacier_melt.sampling import AE_COLS, EASD_COLS, LABEL_COL


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Uses 300 trees, exceeding the ~128-tree convergence threshold identified
    by Oshiro et al. (2012). OOB scoring is enabled for in-bag error
    estimation without a separate validation set.

    Parameters
    ----------
    X_train : np.ndarray, shape (N, n_features)
        Feature matrix (EASD or AE64 embeddings).
    y_train : np.ndarray, shape (N,)
        Binary melt labels (0 = ice, 1 = melt).
    n_estimators : int
        Number of trees. Default 300.
    n_jobs : int
        Parallel jobs. ``-1`` uses all cores.
    random_state : int

    Returns
    -------
    RandomForestClassifier
        Fitted classifier with ``oob_score_`` available.
    """
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    print(f"RF trained in {time.time()-t0:.1f}s, OOB: {rf.oob_score_:.4f}")
    return rf


def sample_for_rf(
    df: pd.DataFrame,
    n_samples: int = 2_000_000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw a stratified subsample for RF training.

    The full 12.2M-pixel training set exceeds practical memory limits for
    the CPU-based sklearn RF. A 2M-pixel stratified subsample preserves
    the melt/non-melt class ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Training pixel dataframe.
    n_samples : int
        Number of pixels to sample. Default 2,000,000.
    random_state : int

    Returns
    -------
    X : np.ndarray — feature matrix
    y : np.ndarray — labels
    """
    df_sample, _ = train_test_split(
        df,
        train_size=n_samples,
        stratify=df[LABEL_COL],
        random_state=random_state,
    )
    print(f"RF sample: {len(df_sample):,} pixels, "
          f"melt rate: {df_sample[LABEL_COL].mean():.3f}")
    return df_sample, _


# ---------------------------------------------------------------------------
# Neighbour table (for CNN patch construction)
# ---------------------------------------------------------------------------

def build_neighbour_table(
    df: pd.DataFrame,
    patch_size: int = 3,
    tolerance: float = 0.6,
) -> tuple[np.ndarray, float, float]:
    """
    Build a pixel → patch-neighbour index table using a cKDTree.

    For each pixel, identifies the surrounding ``patch_size × patch_size``
    neighbourhood by querying a cKDTree over (lon, lat) coordinates. The
    median pixel spacing is estimated from unique coordinate values, and
    used to determine grid alignment. Neighbours that are not on the
    expected grid (residual > ``tolerance`` × pixel spacing) are excluded,
    so boundary pixels whose patches extend outside the glacier are
    zero-padded (index = -1).

    Parameters
    ----------
    df : pd.DataFrame
        Pixel dataframe with ``lon`` and ``lat`` columns.
    patch_size : int
        Side length of the square patch. Default 3 (3×3 patches).
    tolerance : float
        Fraction of pixel spacing allowed as residual alignment error.
        Default 0.6.

    Returns
    -------
    neighbour_table : np.ndarray, shape (N, patch_size, patch_size)
        Integer indices into ``df`` giving the neighbour at each patch
        position. Entries of -1 indicate missing (boundary) pixels.
    pixel_lon : float
        Estimated pixel spacing in longitude.
    pixel_lat : float
        Estimated pixel spacing in latitude.
    """
    half = patch_size // 2
    coords = np.stack([df["lon"].values, df["lat"].values], axis=1)

    pixel_lon = np.median(np.diff(np.sort(df["lon"].unique())))
    pixel_lat = np.median(np.diff(np.sort(df["lat"].unique())))

    print(f"  Building cKDTree for {len(df):,} pixels "
          f"(pixel spacing: {pixel_lon:.6f}°, {pixel_lat:.6f}°)...")
    tree = cKDTree(coords)
    k = patch_size ** 2 + 1
    distances, indices = tree.query(coords, k=k, workers=-1)

    neighbour_table = np.full(
        (len(df), patch_size, patch_size), -1, dtype=np.int32
    )

    for slot in range(k):
        neighbour_coords = coords[indices[:, slot]]
        delta_lon = neighbour_coords[:, 0] - coords[:, 0]
        delta_lat = neighbour_coords[:, 1] - coords[:, 1]
        dc = np.round(delta_lon / pixel_lon).astype(int)
        dr = np.round(delta_lat / pixel_lat).astype(int)

        patch_row = (dr + half).clip(0, patch_size - 1)
        patch_col = (dc + half).clip(0, patch_size - 1)

        residual_lon = np.abs(delta_lon - dc * pixel_lon)
        residual_lat = np.abs(delta_lat - dr * pixel_lat)

        valid = (
            (np.abs(dc) <= half) & (np.abs(dr) <= half) &
            (residual_lon < tolerance * pixel_lon) &
            (residual_lat < tolerance * pixel_lat) &
            (patch_row >= 0) & (patch_row < patch_size) &
            (patch_col >= 0) & (patch_col < patch_size)
        )

        pixel_indices = np.arange(len(df))
        mask = valid & (
            neighbour_table[pixel_indices, patch_row, patch_col] == -1
        )
        neighbour_table[
            pixel_indices[mask], patch_row[mask], patch_col[mask]
        ] = indices[mask, slot]

    return neighbour_table, pixel_lon, pixel_lat


# ---------------------------------------------------------------------------
# Tensor construction
# ---------------------------------------------------------------------------

def build_pixel_tensors(df: pd.DataFrame) -> TensorDataset:
    """
    Build a TensorDataset of single-pixel AE embeddings for MLP training.

    All tensors are moved to GPU immediately if CUDA is available. The
    custom ``GPUDataLoader`` then batches directly from GPU memory,
    avoiding per-batch CPU→GPU transfers.

    Parameters
    ----------
    df : pd.DataFrame
        Pixel dataframe containing AE embedding columns (A00–A63) and
        melt labels.

    Returns
    -------
    TensorDataset
        Tensors on GPU (or CPU if no CUDA). Shape: (N, 64) and (N,).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = torch.from_numpy(
        df[AE_COLS].values.astype(np.float32)
    ).to(device)
    labels = torch.from_numpy(
        df[LABEL_COL].values.astype(np.float32)
    ).to(device)
    print(f"  {len(df):,} pixel tensors on {device}")
    return TensorDataset(embeddings, labels)


def build_patch_tensors(df: pd.DataFrame, patch_size: int = 3) -> TensorDataset:
    """
    Build a TensorDataset of spatial patch embeddings for CNN training.

    Constructs the neighbour table via ``build_neighbour_table``, then
    gathers the AE embeddings for each pixel's patch neighbourhood.
    Boundary pixels (index -1 in the neighbour table) are zero-padded.
    Output tensor shape: (N, 64, patch_size, patch_size).

    Parameters
    ----------
    df : pd.DataFrame
        Pixel dataframe with AE embeddings and coordinate columns.
    patch_size : int
        Side length of the square patch. Default 3.

    Returns
    -------
    TensorDataset
        Tensors on GPU (or CPU if no CUDA).
        Shapes: (N, 64, patch_size, patch_size) and (N,).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nt, _, _ = build_neighbour_table(df, patch_size)
    n = len(df)
    flat = nt.reshape(n, patch_size * patch_size)

    missing = flat == -1
    flat_safe = flat.copy()
    flat_safe[missing] = 0

    embeddings = df[AE_COLS].values.astype(np.float32)
    patches = embeddings[flat_safe]          # (N, patch_size^2, 64)
    patches[missing] = 0.0
    patches = patches.transpose(0, 2, 1).reshape(n, 64, patch_size, patch_size)

    labels = df[LABEL_COL].values.astype(np.float32)
    print(f"  {n:,} patch tensors ({patch_size}×{patch_size}) on {device}")
    return TensorDataset(
        torch.from_numpy(patches).to(device),
        torch.from_numpy(labels).to(device),
    )


def save_tensors(dataset: TensorDataset, path: str | Path) -> None:
    """
    Save a TensorDataset to disk as a ``.pt`` file.

    Caching tensors avoids rebuilding the neighbour table (which is
    expensive for large regions) on every training run.

    Parameters
    ----------
    dataset : TensorDataset
    path : str or Path
    """
    torch.save(dataset, str(path))
    print(f"Saved tensors: {path}")


def load_tensors(path: str | Path) -> TensorDataset:
    """
    Load a TensorDataset from a cached ``.pt`` file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    TensorDataset
    """
    return torch.load(str(path))


# ---------------------------------------------------------------------------
# GPU DataLoader
# ---------------------------------------------------------------------------

class GPUDataLoader:
    """
    All-on-GPU data loader for fast mini-batch iteration.

    Keeps the full dataset in GPU memory and samples batch indices directly
    on the GPU via ``torch.randperm``, avoiding the CPU→GPU transfer that
    occurs with a standard PyTorch DataLoader on each batch.

    Parameters
    ----------
    tensor_dataset : TensorDataset
        Dataset with tensors already on GPU.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        If True, shuffles the dataset on each epoch.
    """

    def __init__(
        self,
        tensor_dataset: TensorDataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        self.patches = tensor_dataset.tensors[0]
        self.labels = tensor_dataset.tensors[1]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.labels)
        self.dataset = self

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        device = self.patches.device
        idx = (
            torch.randperm(self.n, device=device)
            if self.shuffle
            else torch.arange(self.n, device=device)
        )
        for start in range(0, self.n, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield self.patches[batch_idx], self.labels[batch_idx]


def make_loaders(
    train_tensors_dict: dict[str, TensorDataset],
    val_tensors: TensorDataset,
    train_regions: list[str],
    batch_size: int = 65536,
) -> tuple[GPUDataLoader, GPUDataLoader, torch.Tensor]:
    """
    Concatenate per-region training tensors and construct loaders.

    Also computes the positive class weight for ``BCEWithLogitsLoss``,
    which corrects for the class imbalance between melt and non-melt pixels.

    Parameters
    ----------
    train_tensors_dict : dict of str → TensorDataset
        Per-region tensor datasets for training regions.
    val_tensors : TensorDataset
        Tensor dataset for the validation region (R3).
    train_regions : list of str
        Keys to use from ``train_tensors_dict``.
    batch_size : int
        Mini-batch size. Default 65536.

    Returns
    -------
    train_loader : GPUDataLoader
    val_loader : GPUDataLoader
    pos_weight : torch.Tensor
        Scalar tensor: (1 - melt_rate) / melt_rate, for BCEWithLogitsLoss.
    """
    all_patches = torch.cat(
        [train_tensors_dict[r].tensors[0] for r in train_regions]
    )
    all_labels = torch.cat(
        [train_tensors_dict[r].tensors[1] for r in train_regions]
    )
    train_combined = TensorDataset(all_patches, all_labels)

    melt_rate = all_labels.cpu().mean().item()
    pos_weight = torch.tensor((1 - melt_rate) / melt_rate)
    print(f"  Melt rate: {melt_rate:.3f}, pos_weight: {pos_weight:.2f}")

    return (
        GPUDataLoader(train_combined, batch_size=batch_size, shuffle=True),
        GPUDataLoader(val_tensors, batch_size=batch_size, shuffle=False),
        pos_weight,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: GPUDataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    train: bool = True,
) -> tuple[float, float]:
    """
    Run one pass over a data loader (train or eval).

    Parameters
    ----------
    model : nn.Module
    loader : GPUDataLoader
    criterion : nn.Module
        Loss function. Should be ``nn.BCEWithLogitsLoss``.
    optimizer : Optimizer or None
        Required when ``train=True``.
    train : bool
        If True, runs forward + backward + update. If False, eval only.

    Returns
    -------
    avg_loss : float
    auc : float
    """
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.set_grad_enabled(train):
        for patches, labels in loader:
            logits = model(patches)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            all_probs.extend(torch.sigmoid(logits).cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / loader.n
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


def train_model(
    model: nn.Module,
    train_loader: GPUDataLoader,
    val_loader: GPUDataLoader,
    pos_weight: torch.Tensor,
    n_epochs: int = 20,
    patience: int = 5,
    device: torch.device | None = None,
    save_path: str | Path | None = None,
) -> float:
    """
    Full training loop with early stopping and LR scheduling.

    Trains for up to ``n_epochs`` epochs, evaluating on validation after
    each. Saves the best checkpoint (by validation AUC) if ``save_path``
    is provided. Training stops early if validation AUC has not improved
    for ``patience`` consecutive epochs.

    The learning rate is halved when validation AUC plateaus for 2 epochs
    (``ReduceLROnPlateau`` with factor=0.5, patience=2).

    Parameters
    ----------
    model : nn.Module
    train_loader : GPUDataLoader
    val_loader : GPUDataLoader
    pos_weight : torch.Tensor
        Class weight for BCEWithLogitsLoss (from ``make_loaders``).
    n_epochs : int
        Maximum number of training epochs. Default 20.
    patience : int
        Early stopping patience. Default 5.
    device : torch.device or None
        Defaults to CUDA if available.
    save_path : str, Path, or None
        Path to save the best model state dict as a ``.pt`` file.

    Returns
    -------
    float
        Best validation AUC achieved.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_auc = 0.0
    best_epoch = 0

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train AUC':>9}  "
          f"{'Val Loss':>8}  {'Val AUC':>7}")
    print("-" * 55)

    for epoch in range(1, n_epochs + 1):
        train_loss, train_auc = run_epoch(
            model, train_loader, criterion, optimizer, train=True
        )
        val_loss, val_auc = run_epoch(
            model, val_loader, criterion, train=False
        )
        scheduler.step(val_auc)

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_epoch = epoch
            if save_path:
                torch.save(model.state_dict(), str(save_path))

        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_auc:>9.4f}  "
            f"{val_loss:>8.4f}  {val_auc:>7.4f}"
            + (" ← best" if is_best else "")
        )

        if epoch - best_epoch >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best: epoch {best_epoch}, AUC {best_val_auc:.4f})")
            break

    print(f"\nBest val AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    return best_val_auc


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model: nn.Module,
    tensor_dataset: TensorDataset,
    batch_size: int = 65536,
) -> np.ndarray:
    """
    Run inference on a TensorDataset, returning per-pixel probabilities.

    Parameters
    ----------
    model : nn.Module
        Model in eval mode (or will be set to eval).
    tensor_dataset : TensorDataset
        Dataset with input tensors on GPU.
    batch_size : int
        Inference batch size.

    Returns
    -------
    np.ndarray, shape (N,), dtype float32
        Per-pixel melt probabilities in [0, 1].
    """
    model.eval()
    loader = GPUDataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for patches, _ in loader:
            logits = model(patches)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(probs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save model state dict and optional metadata to a ``.pt`` file.

    Parameters
    ----------
    model : nn.Module
    path : str or Path
    metadata : dict or None
        Extra fields stored alongside weights, e.g.
        ``{'epoch': 12, 'val_auc': 0.9454}``.
    """
    payload: dict[str, Any] = {"state_dict": model.state_dict()}
    if metadata:
        payload.update(metadata)
    torch.save(payload, str(path))
    print(f"Checkpoint saved: {path}")
