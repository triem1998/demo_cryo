"""Rotate3D — deepinv Transform wrapping icecream's cubic-symmetry rotation set.

Generates a random 90°-multiple 3D rotation (from the 24-element cubic symmetry
group, optionally extended with axis flips) following the exact same convention
used by icecream's equivariant trainer.
"""
from __future__ import annotations

import torch
import deepinv.transform as T


class Rotate3D(T.Transform):
    """Random 90°-multiple 3D rotation drawn from the cubic symmetry group.

    Uses the same ``k_set`` as icecream (40 transformations, rotations + flips)
    so the EI loss sees the same distribution of augmentations.

    :param bool n_trans: number of independent transforms to sample per input
        (passed to the deepinv Transform base class). Default 1.
    :param tuple[int,int,int] | None volume_shape: When given and not a cube,
        only rotations that preserve this shape are sampled — needed for
        tomography physics, whose astra geometry is bound to a fixed
        ``volume_shape`` (unlike the Fourier-wedge presets, which always crop
        to a cube). ``None`` (default) keeps all 40 — cubic volumes are
        shape-preserving under every entry, so this is a no-op there.
    """

    # The 40-element k_set from icecream (kx, ky, kz, flip_axis) where
    # flip_axis=-1 means no flip.
    _KSET = [
        [0, 0, 1, -1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2],
        [0, 0, 3, -1], [0, 0, 3, 1],
        [0, 1, 0, -1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 2],
        [0, 1, 1, -1], [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 2],
        [0, 1, 2, -1], [0, 1, 2, 1], [0, 1, 3, -1], [0, 1, 3, 1],
        [0, 2, 1, -1], [0, 2, 3, -1],
        [0, 3, 0, -1], [0, 3, 1, -1], [0, 3, 2, -1], [0, 3, 3, -1],
        [1, 0, 0, -1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 2],
        [1, 0, 1, -1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 1, 2],
        [1, 0, 2, -1], [1, 0, 2, 1], [1, 0, 3, -1], [1, 0, 3, 1],
        [1, 2, 0, -1], [1, 2, 1, -1], [1, 2, 2, -1], [1, 2, 3, -1],
    ]

    def __init__(
        self, n_trans: int = 1,
        volume_shape: tuple[int, int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(n_trans=n_trans, **kwargs)
        self._valid_indices = self._shape_preserving_indices(volume_shape)

    @staticmethod
    def _rotated_shape(shape: tuple[int, int, int], kx: int, ky: int, kz: int) -> tuple[int, int, int]:
        """Shape after the same (kx, ky, kz) rot90 sequence as ``transform`` —
        an odd count swaps the two axes it acts on; flips never change shape."""
        s = list(shape)
        for k, (a, b) in ((kx, (1, 2)), (ky, (0, 2)), (kz, (0, 1))):
            if k % 2:
                s[a], s[b] = s[b], s[a]
        return tuple(s)

    @classmethod
    def _shape_preserving_indices(cls, volume_shape: tuple[int, int, int] | None) -> list[int]:
        if volume_shape is None:
            return list(range(len(cls._KSET)))
        shape = tuple(int(s) for s in volume_shape)
        return [i for i, (kx, ky, kz, _axis) in enumerate(cls._KSET)
                if cls._rotated_shape(shape, kx, ky, kz) == shape]

    # ------------------------------------------------------------------
    # deepinv Transform interface
    # ------------------------------------------------------------------

    def get_params(self, x: torch.Tensor) -> dict:
        """Sample a random rotation index (same k for the whole batch)."""
        idx = self._valid_indices[int(torch.randint(len(self._valid_indices), (1,)).item())]
        return {"k_idx": idx}

    def transform(self, x: torch.Tensor, k_idx: int = 0, **kwargs) -> torch.Tensor:
        """Apply the selected 90°-multiple rotation to x.

        :param torch.Tensor x: shape (B, C, D, H, W).
        :param int k_idx: index into ``_KSET``.
        :return: Rotated tensor of same shape.
        """
        kx, ky, kz, axis = self._KSET[k_idx]

        # rot90 on 5-D tensor (B,C,D,H,W): 2=D, 3=H, 4=W
        # icecream on 3-D (D,H,W): kx→(1,2)=(H,W), ky→(0,2)=(D,W), kz→(0,1)=(D,H)
        out = torch.rot90(x,   k=kx, dims=(3, 4))  # (H, W)
        out = torch.rot90(out, k=ky, dims=(2, 4))  # (D, W)
        out = torch.rot90(out, k=kz, dims=(2, 3))  # (D, H)

        if axis != -1:
            out = torch.flip(out, [axis + 2])  # +2 to skip B, C dims

        return out
