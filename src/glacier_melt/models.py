"""
glacier_melt.models
===================
PyTorch model architecture definitions.

Three model families are implemented, all producing a per-pixel raw logit
as output. Sigmoid is applied externally (at inference time via
``torch.sigmoid``) rather than inside the model, since training uses
``nn.BCEWithLogitsLoss`` which expects raw logits for numerical stability.

MLP
    Classifies each pixel independently from its 64-dimensional AE embedding.
    Architecture: Linear(64→512) → BN → ReLU → Dropout →
                  Linear(512→256) → BN → ReLU → Dropout →
                  Linear(256→128) → BN → ReLU → Dropout →
                  Linear(128→1)
    Selected for deployment: smallest train/validation overlap gap (1.58%),
    indicating robust cross-regional generalisation.

CNN (3×3 patch)
    Uses a 3×3 spatial patch of AE embeddings as input (shape: B, 64, 3, 3).
    Two convolutional layers with kernel_size=2, reducing a 3×3 input to
    1×1 before the FC head. FC head: 256→128→32→1.
    Train/validation overlap gap: 4.69%.

CNN (5×5 patch)
    Uses a 5×5 spatial patch. Three convolutional layers. Not selected for
    deployment due to larger train/validation gap (11.05%).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Single-pixel MLP for melt classification.

    Operates on individual 64-dimensional AlphaEarth embeddings with no
    spatial context. This makes it robust to the spatial overfitting that
    affects CNN models when applied to geographically held-out regions.

    Architecture
    ------------
    Linear(64→512) → BN → ReLU → Dropout →
    Linear(512→256) → BN → ReLU → Dropout →
    Linear(256→128) → BN → ReLU → Dropout →
    Linear(128→1)

    Output is a raw logit. Apply ``torch.sigmoid`` for probabilities.

    Parameters
    ----------
    dropout : float
        Dropout probability applied after each ReLU. Default 0.5.
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 64)
            Raw AlphaEarth embedding vectors.

        Returns
        -------
        torch.Tensor, shape (batch,)
            Raw logits. Apply sigmoid for melt probabilities.
        """
        return self.net(x).squeeze(1)


# ---------------------------------------------------------------------------
# CNN (3×3 patch)
# ---------------------------------------------------------------------------

class CNN3x3(nn.Module):
    """
    Patch-based CNN with a 3×3 input patch.

    Convolutional backbone uses kernel_size=2, which reduces a 3×3 spatial
    input to 1×1 after two layers:
        3×3 → Conv(k=2) → 2×2 → Conv(k=2) → 1×1

    Architecture
    ------------
    Conv2d(64→128, k=2) → BN → ReLU →
    Conv2d(128→256, k=2) → BN → ReLU →
    Flatten → Linear(256→128) → ReLU → Dropout →
              Linear(128→32)  → ReLU → Dropout →
              Linear(32→1)

    Output is a raw logit.

    Parameters
    ----------
    dropout : float
        Dropout probability in the FC head. Default 0.5.
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 64, 3, 3)
            3×3 patch of AlphaEarth embeddings. Boundary pixels that extend
            outside the glacier mask are zero-padded.

        Returns
        -------
        torch.Tensor, shape (batch,)
            Raw logits.
        """
        return self.fc_block(self.conv_block(x)).squeeze(1)


# ---------------------------------------------------------------------------
# CNN (5×5 patch)
# ---------------------------------------------------------------------------

class CNN5x5(nn.Module):
    """
    Patch-based CNN with a 5×5 input patch.

    Three convolutional layers with kernel_size=2 reduce 5×5 → 4×4 → 3×3 → 2×2,
    followed by an adaptive average pool to 1×1 before the FC head.

    Not selected for deployment due to a train/validation overlap gap of
    11.05%, indicating stronger spatial overfitting than the 3×3 CNN (4.69%)
    or MLP (1.58%).

    Parameters
    ----------
    dropout : float
        Dropout probability in the FC head. Default 0.5.
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 64, 5, 5)
            5×5 patch of AlphaEarth embeddings.

        Returns
        -------
        torch.Tensor, shape (batch,)
            Raw logits.
        """
        return self.fc_block(self.conv_block(x)).squeeze(1)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name: str, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    model_name : str
        One of ``'mlp'``, ``'cnn3x3'``, ``'cnn5x5'``.
    **kwargs
        Passed to the model constructor (e.g. ``dropout=0.5``).

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        If ``model_name`` is not recognised.
    """
    registry = {
        "mlp":    MLP,
        "cnn3x3": CNN3x3,
        "cnn5x5": CNN5x5,
    }
    if model_name not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(registry)}."
        )
    return registry[model_name](**kwargs)


def load_model(
    checkpoint_path: str,
    model_name: str,
    device: torch.device | None = None,
    **kwargs,
) -> nn.Module:
    """
    Load a saved model from a ``.pt`` checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` file containing the model state dict.
    model_name : str
        Architecture identifier (``'mlp'``, ``'cnn3x3'``, ``'cnn5x5'``).
    device : torch.device or None
        Device to load onto. Defaults to CUDA if available, else CPU.
    **kwargs
        Architecture hyperparameters passed to ``build_model``
        (e.g. ``dropout=0.5``).

    Returns
    -------
    nn.Module
        Model in eval mode with weights loaded from checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, **kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model
