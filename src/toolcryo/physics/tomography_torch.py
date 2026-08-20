"""TomographyEMTorch — pure-PyTorch single-axis-tilt parallel-beam projector.

Drop-in for the astra-based :class:`toolcryo.physics.TomographyEM` that runs on
*any* device (CUDA, ROCm/AMD, MPS, CPU), where astra runs only on NVIDIA. Built
from ``torch`` ops alone — ``grid_sample`` plus an FFT ramp filter. Matches astra
to correlation ~0.999998 on the forward projection and ~1.0 on FBP; the "Sampling
scheme" section below is why. 

Backend-portable because :class:`toolcryo.physics.TomographyEM` wraps
``deepinv.physics.TomographyWithAstra``, and astra-toolbox ships CUDA kernels
that AMD/ROCm GPUs cannot run at all. Verified by
``tests/test_tomography_torch.py`` (correctness + live astra parity) and measured
by ``scripts/bench_tomography.py`` (correctness + speed side by side).

Why this is only a 2D problem
-----------------------------
astra's ``parallel3d`` rotates the volume about its z-axis, which in deepinv's
``(n_slices, n_rows, n_cols)`` layout is ``n_slices`` — for this dataset the
**tilt axis Y**, i.e. dim ``-3``, exactly what ``load_mrc_volume(order="astra")``
produces. For a parallel beam no ray crosses between slices: detector row ``v``
sees only volume slice ``v``. The 3D operator is therefore exactly ``n_slices``
independent 2D parallel-beam problems on the ``(n_rows=Z, n_cols=X)`` plane, and
one 2D rotate-and-sum batched over slices is the whole forward operator.

The slices are carried in ``grid_sample``'s **channel** dimension, so a single
``(1, ·, N, 2)`` sampling grid serves every slice without being materialised
per-slice.

Sampling scheme — Joseph's method, matching astra
-------------------------------------------------
astra's ``parallel3d`` GPU kernels do **not** march along the ray. They step
along whichever axis the ray is most aligned with, taking unit steps in that
axis, interpolating in the transverse one, and rescaling by the ray length per
step. This was measured: for a constant volume astra returns
*exactly* ``Z/cos θ`` (6.9282 at θ=30, Z=6), which is ``Z`` samples times
``1/cos θ`` — whereas marching along the ray with unit spacing gives 6.938.
Reproducing that choice took the forward-projection agreement from 17% relative
L2 to ~1%: on a random (full-bandwidth) volume, two quadratures of the same
integral disagree at exactly the level their sample positions differ.

So, with ``a = u - N/2 + 0.5`` the detector coordinate and the volume centred:

* **z-dominant** (``|cos θ| >= |sin θ|``), stepping over rows ``j = 0..Z-1``::

      ζ     = j + 0.5 - Z/2
      x_pix = a/cos θ - ζ·tan θ + X/2      z_pix = j + 0.5
      p_θ(u) = (1/|cos θ|) · Σ_j f(z_pix, x_pix)

* **x-dominant** (``|sin θ| > |cos θ|``), stepping over columns ``i = 0..X-1``::

      ξ     = i + 0.5 - X/2
      z_pix = a/sin θ - ξ·cot θ + Z/2      x_pix = i + 0.5
      p_θ(u) = (1/|sin θ|) · Σ_i f(z_pix, x_pix)

Both parameterise the same family of lines — ``a/cos θ`` is where the ray for
detector bin ``a`` crosses the mid-plane, while ``a·cos θ`` is its closest
approach to the origin — so this is a change of quadrature, not of geometry.

Two more conventions were measured:

* **Scale.** ``object_cell_volume == 1.0`` and ``detector_cell_v_length == 1.0``
  for this geometry, so there is no hidden normalisation and the FBP weighting
  collapses to ``π / (2·n_angles)``.
* **Rotation sign.** A delta at ``(z=1, x=2)`` in an 8×6×10 volume lands at
  detector bin ``3.085`` at θ=30 under astra; the map above predicts ``1.585``
  for ``+θ`` and ``3.085`` for ``-θ``. astra's rotation is therefore ``-θ`` in
  this parameterisation — hence the negation in :meth:`_thetas`.

At θ=0 the z-dominant branch samples exact pixel centres with unit weight, so
the projection is then *exactly* a sum along Z — the sharpest available check on
the origin/half-pixel convention (``test_zero_angle_equals_axis_sum``).
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
import deepinv as dinv
from deepinv.physics.functional.radon import RampFilter


def _grid_sample_vjp(
    cotangent: torch.Tensor, template: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """Transpose of ``grid_sample`` applied to ``cotangent`` (a scatter-add).

    This is the mathematical adjoint of the sampling step, which torch exposes
    only as an autograd backward. ``aten.grid_sampler_2d_backward`` gives it
    directly and is ~3x faster than round-tripping through ``autograd.grad``,
    which has to run a *forward* ``grid_sample`` first purely to build a graph.
    The two agree to fp32 round-off (~2e-6). The ATen op is not part of the
    documented public API, so a version that lacks or renames it falls back to
    the autograd route rather than breaking.

    :param torch.Tensor cotangent: gradient w.r.t. the sampled output.
    :param torch.Tensor template: zeros with the *input's* shape/dtype/device.
    :param torch.Tensor grid: the sampling grid used in the forward direction.
    """
    try:
        grad, _ = torch.ops.aten.grid_sampler_2d_backward(
            cotangent, template, grid, 0, 0, False, [True, False],
        )
        return grad
    except (AttributeError, RuntimeError):  # pragma: no cover - version fallback
        leaf = template.detach().requires_grad_(True)
        with torch.enable_grad():
            samp = F.grid_sample(leaf, grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=False)
        (grad,) = torch.autograd.grad(samp, leaf, grad_outputs=cotangent)
        return grad


def _work_dtype(t: torch.Tensor) -> torch.dtype:
    """Compute dtype: float64 is honoured (gradcheck), everything else is fp32.

    Line integrals in fp16 lose far too much precision, so an autocast region
    must not drag the projector down with it. The three autograd wrappers above
    also carry ``custom_fwd(cast_inputs=torch.float32)``, which makes the fp32
    guarantee structural on CUDA — ``grid_sample`` is autocast-fallthrough and
    returns whatever dtype it is handed, so without that this function is the
    sole guard. It stays as the guard for CPU autocast and for the paths that
    do not enter through the wrappers.
    """
    return torch.float64 if t.dtype == torch.float64 else torch.float32


class _Project(torch.autograd.Function):
    """``A`` as an autograd op whose backward is the exact adjoint.

    Mirrors how deepinv wraps astra (``functional.astra.AutogradTransform``):
    the chunked projection runs under ``no_grad`` and the backward simply calls
    the backprojection, so no intermediate sampling tensor is retained. That
    matters for the unrolled preset, which calls ``A``/``A_adjoint`` once per PGD
    iteration — a retained graph would multiply memory by ``n_iter``.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, x: torch.Tensor, op: "TomographyEMTorch") -> torch.Tensor:
        ctx.op = op
        return op._project(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return ctx.op._backproject_dispatch(grad_out.contiguous()), None


class _Backproject(torch.autograd.Function):
    """``A_adjoint`` as an autograd op; its backward is the projection."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, y: torch.Tensor, op: "TomographyEMTorch") -> torch.Tensor:
        ctx.op = op
        return op._backproject_dispatch(y)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return ctx.op._project(grad_out.contiguous()), None


class _BackprojectPixelDrivenAstraGrad(torch.autograd.Function):
    """Voxel-driven back-projection whose backward is the *forward* projection.

    Used by ``fbp`` in ``adjoint_mode="fast"`` only, to reproduce astra's
    behaviour exactly. deepinv defines the backward of its astra ``A_adjoint``
    wrapper to be the forward projection ``A``, so astra's FBP gradient is
    ``filter . w . A``. That is *not* the transpose of astra's own FBP, because
    ``A`` (Joseph, ray-length weighted) is not the transpose of the voxel-driven
    back-projector (a plain gather) — the same mismatch as
    :meth:`TomographyEMTorch._backproject`, one level up the chain. Measured, it
    is a ~1.15x scale difference plus a shape difference (corr 0.985); the
    ray-length weight ``1/|cos t|`` that ``A`` carries and the gather's transpose
    does not is the origin, though the net factor is not simply its mean.

    In ``"exact"`` mode ``fbp`` skips this wrapper and lets autograd transpose
    the back-projector it actually used, which is mathematically right.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, y: torch.Tensor, op: "TomographyEMTorch") -> torch.Tensor:
        ctx.op = op
        return op._backproject_pixel_driven(y)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return ctx.op._project(grad_out.contiguous()), None


class TomographyEMTorch(dinv.physics.LinearPhysics):
    """Pure-torch drop-in for :class:`toolcryo.physics.TomographyEM`.

    The constructor signature matches ``TomographyEM`` exactly, so the two are
    substitutable at the call site, plus three chunking knobs that only affect
    memory/speed, never the result.

    :param tuple[int,int,int] volume_shape: ``(n_slices, n_rows, n_cols)``, i.e.
        astra order ``(Y, Z, X)`` for this dataset.
    :param angles_deg: 1D array/tensor of tilt angles in degrees.
    :param tuple[int,int] | None detector_shape: ``(V, N)``. Defaults to
        ``(volume_shape[0], volume_shape[2])``. ``V`` must equal ``n_slices`` —
        the separability assumption above.
    :param float pixel_spacing: isotropic voxel size of the object grid.
    :param float detector_spacing: isotropic detector pixel size. Must equal
        ``pixel_spacing`` (parallel beam with a 1:1 slice/row correspondence).
    :param float angle_sign: multiplier applied to ``angles_deg``, the
        rotation-direction calibration knob. Same meaning as in ``TomographyEM``.
    :param bool normalize: rescale ``A``/``A_adjoint`` to unit operator norm,
        exactly as ``TomographyWithAstra`` does (power iteration on a seeded
        random volume).
    :param device: any torch device — unlike ``TomographyEM``, CPU and ROCm work.
    :param str adjoint_mode: which back-projector ``A_adjoint`` uses, and hence
        which gradient training sees (deepinv makes ``A``'s backward = ``A_adjoint``).

        - ``"exact"`` (default): the true transpose of ``A``, so the PGD gradient
          ``A^T(Ax - y)`` is a genuine gradient. ~24x astra at 512^3 (a
          transposed gather is a scatter-add that contends on atomics).
        - ``"fast"``: reproduces astra's own back-projector bit-for-bit (corr
          0.999998), ~10x quicker — but it is *not* a transpose, so its gradient
          is astra's approximation rather than the true one. Pick it to stay
          numerically comparable with an astra run.

        The mode governs *every* backward pass consistently. The ``fbp``
        **output** is unaffected — it is the textbook reconstruction in both
        modes and matches astra either way; only the gradient through it moves.
    :param int | None angle_chunk: angles processed per ``grid_sample`` call.
    :param int | None slice_chunk: slices (channels) per call.
    :param int | None ray_chunk: samples along the ray per call.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        angles_deg,
        detector_shape: tuple[int, int] | None = None,
        pixel_spacing: float = 1.0,
        detector_spacing: float = 1.0,
        angle_sign: float = 1.0,
        normalize: bool = False,
        device: str | torch.device = "cuda",
        adjoint_mode: str = "exact",
        angle_chunk: int | None = None,
        slice_chunk: int | None = None,
        ray_chunk: int | None = None,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.volume_shape = tuple(int(s) for s in volume_shape)
        n_slices, n_rows, n_cols = self.volume_shape

        if detector_shape is None:
            detector_shape = (n_slices, n_cols)
        self.detector_shape = tuple(int(s) for s in detector_shape)

        # --- the separability preconditions (see the module docstring) -------
        if self.detector_shape[0] != n_slices:
            raise NotImplementedError(
                "TomographyEMTorch assumes one detector row per volume slice "
                f"(single-axis parallel beam), got detector rows="
                f"{self.detector_shape[0]} vs n_slices={n_slices}. Use the astra "
                "backend for non-separable geometries."
            )
        if not math.isclose(float(pixel_spacing), float(detector_spacing)):
            raise NotImplementedError(
                "TomographyEMTorch requires detector_spacing == pixel_spacing, got "
                f"{detector_spacing} vs {pixel_spacing}."
            )
        self.pixel_spacing = float(pixel_spacing)
        self.detector_spacing = float(detector_spacing)

        if adjoint_mode not in ("exact", "fast"):
            raise ValueError(
                f"adjoint_mode must be 'exact' or 'fast', got {adjoint_mode!r}."
            )
        self.adjoint_mode = adjoint_mode

        angles = torch.as_tensor(angles_deg, dtype=torch.float32).flatten()
        self.angles_deg = (angles * float(angle_sign)).to(self.device)
        self.n_angles = int(self.angles_deg.numel())
        # Logging only — pre-sign-flip, so it matches the .tlt file. Same
        # private names TomographyEM uses, so trainer.py's one log line reads
        # either backend.
        self._tilt_min = float(angles.min())
        self._tilt_max = float(angles.max())
        self.img_size = self.volume_shape

        # Joseph's method: split the angles by dominant axis. Each group steps
        # over a different number of samples (Z rows vs X columns) and carries
        # its own ray-length weight, so they are swept separately.
        theta = self._thetas()
        cos, sin = theta.cos().abs(), theta.sin().abs()
        z_dominant = cos >= sin
        self._angle_groups = [
            (torch.nonzero(z_dominant).flatten(), "z", n_rows),
            (torch.nonzero(~z_dominant).flatten(), "x", n_cols),
        ]
        # Ray length per unit step along the dominant axis.
        self.ray_weight = torch.where(z_dominant, 1.0 / cos, 1.0 / sin)

        self.angle_chunk, self.slice_chunk, self.ray_chunk = self._auto_chunks(
            angle_chunk, slice_chunk, ray_chunk
        )

        self.filter = RampFilter(dtype=torch.float32).to(self.device)

        # Match TomographyWithAstra bit-for-bit: power iteration on a seeded
        # random volume, ``squared=False`` so operator_norm is ||A||, and A /
        # A_adjoint divide by it while fbp multiplies back by its square.
        self.normalize = False
        if normalize:
            x0 = torch.randn(
                self.volume_shape,
                generator=torch.Generator(self.device).manual_seed(0),
                device=self.device,
            )[None, None]
            self.operator_norm = self.compute_norm(x0, squared=False)
            self.normalize = True

    # ------------------------------------------------------------------
    # chunking
    # ------------------------------------------------------------------

    #: Elements per ``grid_sample`` call the auto-tuner aims for (~256 MB fp32).
    CHUNK_BUDGET_ELEMS = 64_000_000

    def _auto_chunks(self, angle_chunk, slice_chunk, ray_chunk) -> tuple[int, int, int]:
        """Pick chunk sizes that keep one ``grid_sample`` call inside the budget.

        The pre-reduction sample tensor is ``slices x angles x steps x N``, which
        at native resolution (1024 slices, 41 angles, 1024 steps, 1024 columns)
        would be ~9 TB in one go. Chunking is therefore a correctness-preserving
        necessity, not a tuning nicety — ``test_chunking_is_exact`` pins that the
        result does not depend on these numbers.

        Explicitly-passed values are honoured as-is; only ``None`` entries are
        auto-sized, halving the auto ones (slices first, then ray steps) until
        the budget is met.
        """
        n_slices = self.volume_shape[0]
        n_det = self.detector_shape[1]
        max_steps = max(self.volume_shape[1], self.volume_shape[2])

        auto = {"angle": angle_chunk is None, "slice": slice_chunk is None,
                "ray": ray_chunk is None}
        a_c = int(angle_chunk) if angle_chunk else (1 if auto["angle"] else 1)
        s_c = int(slice_chunk) if slice_chunk else n_slices
        r_c = int(ray_chunk) if ray_chunk else max_steps

        # A small problem needs no splitting at all: prefer one big call.
        if auto["angle"] and s_c * self.n_angles * r_c * n_det <= self.CHUNK_BUDGET_ELEMS:
            a_c = self.n_angles

        while s_c * a_c * r_c * n_det > self.CHUNK_BUDGET_ELEMS:
            if auto["angle"] and a_c > 1:
                a_c = max(1, a_c // 2)
            elif auto["slice"] and s_c > 1:
                s_c = max(1, s_c // 2)
            elif auto["ray"] and r_c > 1:
                r_c = max(1, r_c // 2)
            else:
                break       # everything was pinned by the caller — respect it
        return a_c, s_c, r_c

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    def _thetas(self) -> torch.Tensor:
        """Rotation angles in radians, in *this module's* parameterisation.

        Negated relative to the values handed to astra — see the "Rotation sign"
        paragraph of the module docstring, where the convention was measured
        rather than assumed.
        """
        return -torch.deg2rad(self.angles_deg)

    def _grid(
        self, theta: torch.Tensor, steps: torch.Tensor, axis: str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sampling grid for one (angle, step) chunk of a dominant-axis sweep.

        :param torch.Tensor theta: ``(a,)`` rotation angles in radians.
        :param torch.Tensor steps: ``(k,)`` integer indices along the dominant axis.
        :param str axis: ``"z"`` (step over rows) or ``"x"`` (step over columns).
        :param torch.dtype dtype: must match the sampled tensor's dtype.
        :return: ``(1, a*k, N, 2)`` grid in ``grid_sample``'s normalised coords,
            last dim ordered ``(x, z)`` = (width, height).
        """
        _, n_rows, n_cols = self.volume_shape
        n_det = self.detector_shape[1]

        a = (torch.arange(n_det, device=self.device, dtype=dtype)
             - n_det / 2 + 0.5)[None, None, :]                     # (1,1,N)
        theta = theta.to(dtype)
        cos = theta.cos()[:, None, None]                           # (a,1,1)
        sin = theta.sin()[:, None, None]
        step = steps.to(dtype)[None, :, None]                      # (1,k,1)

        if axis == "z":
            zeta = step + 0.5 - n_rows / 2
            x_pix = a / cos - zeta * (sin / cos) + n_cols / 2
            z_pix = (step + 0.5).expand_as(x_pix)
        else:
            xi = step + 0.5 - n_cols / 2
            z_pix = a / sin - xi * (cos / sin) + n_rows / 2
            x_pix = (step + 0.5).expand_as(z_pix)

        # Pixel-centre coordinates: index i has centre i+0.5, so the normalised
        # coordinate for align_corners=False is 2·p/size - 1.
        grid = torch.stack(
            (x_pix * (2.0 / n_cols) - 1.0, z_pix * (2.0 / n_rows) - 1.0), dim=-1
        )                                                          # (a,k,N,2)
        return grid.reshape(1, -1, n_det, 2)

    # ------------------------------------------------------------------
    # the two primitives — chunked, allocation-bounded, autograd-free
    # ------------------------------------------------------------------

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Forward projection, volume ``(B,C,V,Z,X)`` -> sinogram ``(B,C,V,A,N)``."""
        b, c = x.shape[:2]
        n_slices, n_rows, n_cols = self.volume_shape
        n_det = self.detector_shape[1]
        # Slices ride in the channel dimension so one grid serves all of them.
        dt = _work_dtype(x)
        x4 = x.reshape(b * c, n_slices, n_rows, n_cols).to(dt)

        theta = self._thetas()
        out = torch.zeros(
            (b * c, n_slices, self.n_angles, n_det), device=x4.device, dtype=dt,
        )

        with torch.no_grad():
            for idx, axis, n_steps in self._angle_groups:
                for a0 in range(0, idx.numel(), self.angle_chunk):
                    sub = idx[a0:a0 + self.angle_chunk]
                    acc = torch.zeros(
                        (b * c, n_slices, sub.numel(), n_det),
                        device=x4.device, dtype=dt,
                    )
                    for s0 in range(0, n_steps, self.ray_chunk):
                        s1 = min(s0 + self.ray_chunk, n_steps)
                        steps = torch.arange(s0, s1, device=self.device)
                        grid = self._grid(theta[sub], steps, axis, dt)
                        grid = grid.expand(b * c, -1, -1, -1)
                        for v0 in range(0, n_slices, self.slice_chunk):
                            v1 = min(v0 + self.slice_chunk, n_slices)
                            samp = F.grid_sample(
                                x4[:, v0:v1], grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False,
                            )
                            acc[:, v0:v1] += samp.view(
                                b * c, v1 - v0, sub.numel(), s1 - s0, n_det
                            ).sum(dim=3)
                    out[:, :, sub] = acc * self.ray_weight.to(dt)[sub][None, None, :, None]
        out *= self.pixel_spacing
        return out.view(b, c, n_slices, self.n_angles, n_det)

    def _backproject_dispatch(self, y: torch.Tensor) -> torch.Tensor:
        """Back-projection in whichever mode the operator was built with.

        Routed through one place so that ``A``'s autograd backward and
        ``A_adjoint`` can never disagree: in ``"fast"`` mode both use the
        voxel-driven path, which is precisely the (mismatched) pair astra ships,
        and in ``"exact"`` mode both use the true transpose.
        """
        if self.adjoint_mode == "fast":
            out = self._backproject_pixel_driven(y)
            # The voxel-driven path has no ray-length weight and no pixel_spacing
            # factor; astra's A_adjoint does not either, so nothing is applied.
            return out
        return self._backproject(y)

    def _backproject(self, y: torch.Tensor) -> torch.Tensor:
        """Exact transpose of :meth:`_project`, sinogram -> volume.

        The transpose of ``grid_sample`` is its own backward (a scatter-add),
        which torch exposes only through autograd — so this evaluates it on a
        throwaway leaf per chunk and detaches. ``sum`` over the ray axis
        transposes to a broadcast, hence the ``expand`` of the cotangent.

        Unlike astra's pixel-driven backprojection (an *approximate* adjoint),
        this is the exact adjoint, so ``<Ax, y> == <x, A^T y>`` to float
        precision.
        """
        b, c = y.shape[:2]
        n_slices, n_rows, n_cols = self.volume_shape
        n_det = self.detector_shape[1]
        dt = _work_dtype(y)
        y4 = y.reshape(b * c, n_slices, self.n_angles, n_det).to(dt)
        # Fold in the per-angle ray-length weight: it is a diagonal scaling in
        # the forward, hence the same diagonal scaling in the transpose.
        y4 = y4 * self.ray_weight.to(dt)[None, None, :, None]

        theta = self._thetas()
        out = torch.zeros(
            (b * c, n_slices, n_rows, n_cols), device=y4.device, dtype=dt
        )

        # grid_sample's transpose does not depend on the input's values, so one
        # zeros template per chunk *shape* is all the ATen VJP needs.
        templates: dict[int, torch.Tensor] = {}

        def _template(n: int) -> torch.Tensor:
            if n not in templates:
                templates[n] = torch.zeros(
                    (b * c, n, n_rows, n_cols), device=y4.device, dtype=dt
                )
            return templates[n]

        for idx, axis, n_steps in self._angle_groups:
            for a0 in range(0, idx.numel(), self.angle_chunk):
                sub = idx[a0:a0 + self.angle_chunk]
                for s0 in range(0, n_steps, self.ray_chunk):
                    s1 = min(s0 + self.ray_chunk, n_steps)
                    steps = torch.arange(s0, s1, device=self.device)
                    grid = self._grid(theta[sub], steps, axis, dt)
                    grid = grid.expand(b * c, -1, -1, -1)
                    for v0 in range(0, n_slices, self.slice_chunk):
                        v1 = min(v0 + self.slice_chunk, n_slices)
                        cot = (
                            y4[:, v0:v1][:, :, sub, :]
                            .unsqueeze(3)
                            .expand(b * c, v1 - v0, sub.numel(), s1 - s0, n_det)
                            .reshape(b * c, v1 - v0, sub.numel() * (s1 - s0), n_det)
                            .contiguous()
                        )
                        out[:, v0:v1] += _grid_sample_vjp(
                            cot, _template(v1 - v0), grid
                        )
        out *= self.pixel_spacing
        return out.view(b, c, n_slices, n_rows, n_cols)

    # ------------------------------------------------------------------
    # deepinv LinearPhysics interface
    # ------------------------------------------------------------------

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward projection: volume ``(B,C,*volume_shape)`` -> ``(B,C,V,A,N)``."""
        out = _Project.apply(x.contiguous(), self)
        if self.normalize:
            out = out / self.operator_norm
        return out

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Exact adjoint (back-projection): ``(B,C,V,A,N)`` -> volume."""
        out = _Backproject.apply(y.contiguous(), self)
        if self.normalize:
            out = out / self.operator_norm
        return out

    def fbp(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Filtered back-projection, matching ``TomographyEM.fbp``.

        The sinogram is centred before filtering for the reason spelled out in
        ``TomographyEM.fbp``: cryo-ET sinograms carry a large DC pedestal, and
        the ramp filter's zero-padding turns that into a step edge whose
        ringing rivals the signal. The absolute DC level is unrecoverable from a
        limited-angle tilt series anyway.
        """
        return self.fbp_raw(y - y.mean(dim=(-3, -2, -1), keepdim=True))

    def fbp_raw(self, y: torch.Tensor) -> torch.Tensor:
        """FBP without the DC centring — mirrors ``TomographyWithAstra.fbp``."""
        filtered = self.filter(y.to(torch.float32), dim=-1).to(_work_dtype(y))
        # detector_cell_v_length / object_cell_volume == 1 for this geometry
        # (measured in A-0), so the weighting is just the angular quadrature.
        weighted = filtered * (math.pi / (2 * self.n_angles))
        # Output is identical either way; only the *gradient* through fbp
        # differs — see _BackprojectPixelDrivenAstraGrad.
        out = (
            _BackprojectPixelDrivenAstraGrad.apply(weighted, self)
            if self.adjoint_mode == "fast"
            else self._backproject_pixel_driven(weighted)
        )
        if self.normalize:
            # A_adjoint would have divided by the norm once; this path does not
            # call it, so only one factor of the norm is owed (deepinv applies
            # ``/norm`` inside A_adjoint and ``*norm**2`` outside).
            out = out * self.operator_norm
        return out

    def _backproject_pixel_driven(self, y: torch.Tensor) -> torch.Tensor:
        """Voxel-driven back-projection — the textbook FBP backprojector.

        For each voxel, sample the filtered sinogram at the detector position it
        maps to and average over angles, with no ray-length weight. Distinct from
        :meth:`_backproject` (the exact transpose, which FBP does *not* want):
        this matches astra's own FBP (corr 0.9952 vs the IMOD reference, against
        0.9836 for the exact transpose) and is ~10x faster, being one plain
        ``grid_sample`` per angle rather than an autograd backward per chunk.
        """
        b, c = y.shape[:2]
        n_slices, n_rows, n_cols = self.volume_shape
        n_det = self.detector_shape[1]
        dt = _work_dtype(y)
        y4 = y.reshape(b * c, n_slices, self.n_angles, n_det).to(dt)

        theta = self._thetas().to(dt)
        zz, xx = torch.meshgrid(
            torch.arange(n_rows, device=self.device, dtype=dt) + 0.5 - n_rows / 2,
            torch.arange(n_cols, device=self.device, dtype=dt) + 0.5 - n_cols / 2,
            indexing="ij",
        )
        out = torch.zeros(
            (b * c, n_slices, n_rows, n_cols), device=y4.device, dtype=dt
        )

        # Fast path is grad-free; but ``fbp`` must stay differentiable in case a
        # caller puts it inside a graph, so track when the input asks for it.
        track = y.requires_grad or torch.is_grad_enabled() and y.grad_fn is not None
        with contextlib.nullcontext() if track else torch.no_grad():
            for ai in range(self.n_angles):
                # Detector coordinate each voxel projects onto at this angle.
                u = xx * theta[ai].cos() + zz * theta[ai].sin() + n_det / 2
                grid = torch.stack(
                    (u * (2.0 / n_det) - 1.0, torch.zeros_like(u)), dim=-1
                )[None]                                    # (1, Z, X, 2)
                grid = grid.expand(b * c, -1, -1, -1)
                for v0 in range(0, n_slices, self.slice_chunk):
                    v1 = min(v0 + self.slice_chunk, n_slices)
                    # (BC, Vc, 1, N): one detector row per slice, sampled in x.
                    row = y4[:, v0:v1, ai, :].unsqueeze(2)
                    out[:, v0:v1] += F.grid_sample(
                        row, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False,
                    )
        return out.view(b, c, n_slices, n_rows, n_cols)
