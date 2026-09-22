"""Local curvature-admissibility repair for VascularMD vessel splines.

This module is intended as a lightweight post-processing bolt-on for VascularMD.
It operates on already fitted, non-bifurcating vessel splines and enforces the
local circular-tube admissibility condition

    r(t) * kappa(t) <= 1 - epsilon

while minimizing a local deviation functional J from the VascularMD reference
spline. Candidate spline spaces with fewer, unchanged, or additional internal
knots are compared using an AIC-like repair information criterion

    RIC = m_I * log(max(J*, J_epsilon)) + 2 * dim(theta),

where J* is the optimized repair cost, m_I is the number of original VascularMD
observations in the fixed reference repair-support window, and dim(theta) is the
number of free scalar SLSQP variables for that candidate.

Scope
-----
* Repairs ordinary vessel segments only. VascularMD bifurcation-trajectory edges
  are deliberately skipped.
* Addresses local tubular curvature singularity / local self-intersection risk.
* Does not detect or repair non-local vessel-vessel collisions.
* Does not mesh. After repair, use VascularMD's normal cross-section/surface/
  volume meshing functions.

The implementation is derived from the final tested notebook used to develop the
method. The numerical defaults intentionally match that reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import math

import numpy as np
from scipy.interpolate import BSpline as SciPyBSpline
from scipy.optimize import brentq, minimize, minimize_scalar

__all__ = [
    "RepairConfig",
    "ViolationInterval",
    "CandidateResult",
    "RegionRepairReport",
    "SplineRepairResult",
    "EdgeRepairResult",
    "NetworkRepairReport",
    "max_radius_curvature",
    "repair_spline",
    "repair_vascularmd_tree",
    "iter_vascularmd_vessel_edges",
]

__version__ = "0.1.0"


# -----------------------------------------------------------------------------
# Configuration and public result types
# -----------------------------------------------------------------------------


@dataclass
class RepairConfig:
    """Configuration for local VascularMD curvature repair.

    Defaults are the values used in the final development notebook.
    """

    # Scientific model ---------------------------------------------------------
    sigma_radius_over_spatial: float = 0.594
    epsilon: float = 0.05

    # Candidate spline spaces around the current/reference spline.
    max_extra_knots: int = 5
    max_removed_knots: int = 2
    repair_padding_spans: int = 1
    knot_removal_search_padding_spans: int = 1

    # Numerical floor used only to avoid log(0) in the repair information score.
    aic_j_log_floor: float = 1e-30

    # Continuous q(t) = r(t) * kappa(t) peak search ---------------------------
    span_search_samples: int = 21
    span_search_xatol: float = 1e-10

    # Violation detection and constrained optimization -------------------------
    detection_samples: int = 700
    objective_samples: int = 60
    initial_constraint_samples: int = 45
    max_constraint_refinement_rounds: int = 10
    max_new_peak_constraints_per_round: int = 8
    constraint_point_merge_tolerance: float = 1e-8
    feasibility_tolerance: float = 2e-4

    min_radius: float = 1e-5
    min_speed: float = 1e-7
    optimizer_maxiter: int = 140
    optimizer_ftol: float = 1e-9
    max_repair_regions: int = 12

    # Reduced-basis initialization for knot-removal candidates -----------------
    knot_removal_fit_samples: int = 100
    knot_removal_fit_samples_per_control: int = 16

    # Endpoint G1 parameterization ---------------------------------------------
    endpoint_tangent_scale_min: float = 0.10
    endpoint_tangent_scale_max: float = 10.0
    boundary_value_tolerance: float = 1e-7
    tangent_direction_tolerance: float = 1e-7

    @property
    def physical_threshold(self) -> float:
        """Strict optimizer target: 1 - epsilon (0.95 by default)."""
        return 1.0 - self.epsilon

    @property
    def acceptance_threshold(self) -> float:
        """Numerical acceptance threshold used outside the NLP."""
        return self.physical_threshold + self.feasibility_tolerance


@dataclass
class ViolationInterval:
    t_lo: float
    t_hi: float
    t_peak: float
    peak_rk: float


@dataclass
class CandidateResult:
    knot_delta: int
    inserted_knots: List[float]
    removed_knots: List[float]
    spline: Optional[Any]
    feasible: bool
    optimizer_success: bool
    optimizer_message: str
    repair_cost: float
    max_rk: float
    t_max_rk: float
    free_parameter_count: int = 0
    aic_j_score: float = np.inf
    aic_j_points: int = 0
    control_count: int = 0
    active_controls: int = 0
    start_tangent_scale_active: bool = False
    end_tangent_scale_active: bool = False
    refinement_rounds: int = 0
    constraint_points: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "knot_delta": self.knot_delta,
            "inserted_knots": list(self.inserted_knots),
            "removed_knots": list(self.removed_knots),
            "feasible": self.feasible,
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
            "repair_cost_J": self.repair_cost,
            "max_rk": self.max_rk,
            "t_max_rk": self.t_max_rk,
            "control_count": self.control_count,
            "free_parameter_count": self.free_parameter_count,
            "aic_j_score": self.aic_j_score,
            "aic_j_points": self.aic_j_points,
            "active_controls": self.active_controls,
            "start_tangent_scale_active": self.start_tangent_scale_active,
            "end_tangent_scale_active": self.end_tangent_scale_active,
            "refinement_rounds": self.refinement_rounds,
            "constraint_points": self.constraint_points,
        }


@dataclass
class RegionRepairReport:
    before_peak_rk: float
    interval: Tuple[float, float]
    chosen_knot_delta: Optional[int]
    chosen_control_count: Optional[int]
    chosen_aic_j_score: float
    chosen_repair_cost: float
    after_local_max_rk: float
    status: str
    candidates: List[CandidateResult]

    def summary(self, include_candidates: bool = False) -> Dict[str, Any]:
        out = {
            "before_peak_rk": self.before_peak_rk,
            "interval": self.interval,
            "chosen_knot_delta": self.chosen_knot_delta,
            "chosen_control_count": self.chosen_control_count,
            "chosen_aic_j_score": self.chosen_aic_j_score,
            "chosen_repair_cost_J": self.chosen_repair_cost,
            "after_local_max_rk": self.after_local_max_rk,
            "status": self.status,
        }
        if include_candidates:
            out["candidates"] = [c.summary() for c in self.candidates]
        return out


@dataclass
class SplineRepairResult:
    spline: Any
    before_max_rk: float
    after_max_rk: float
    repair_attempted: bool
    repaired: bool
    success: bool
    net_knot_delta: int
    regions: List[RegionRepairReport]

    @property
    def changed(self) -> bool:
        return self.repaired

    def summary(self, include_regions: bool = True, include_candidates: bool = False) -> Dict[str, Any]:
        out = {
            "before_max_rk": self.before_max_rk,
            "after_max_rk": self.after_max_rk,
            "repair_attempted": self.repair_attempted,
            "repaired": self.repaired,
            "success": self.success,
            "net_knot_delta": self.net_knot_delta,
        }
        if include_regions:
            out["regions"] = [r.summary(include_candidates) for r in self.regions]
        return out


@dataclass
class EdgeRepairResult:
    edge: Tuple[Any, Any]
    status: str
    before_max_rk: float = np.nan
    after_max_rk: float = np.nan
    repair: Optional[SplineRepairResult] = None
    error: Optional[str] = None

    @property
    def changed(self) -> bool:
        return bool(self.repair is not None and self.repair.repaired)

    def summary(self, include_regions: bool = False, include_candidates: bool = False) -> Dict[str, Any]:
        out = {
            "edge": self.edge,
            "status": self.status,
            "before_max_rk": self.before_max_rk,
            "after_max_rk": self.after_max_rk,
            "changed": self.changed,
            "error": self.error,
        }
        if self.repair is not None:
            out.update({
                "net_knot_delta": self.repair.net_knot_delta,
                "repair_success": self.repair.success,
            })
            if include_regions:
                out["regions"] = [
                    r.summary(include_candidates) for r in self.repair.regions
                ]
        return out


@dataclass
class NetworkRepairReport:
    """Return value from :func:`repair_vascularmd_tree`.

    ``tree`` is the repaired tree. It is the input object when ``inplace=True``
    and a deep copy when ``inplace=False``.
    """

    tree: Any = field(repr=False)
    edge_results: List[EdgeRepairResult] = field(default_factory=list)
    inplace: bool = False
    skipped_bifurcation_edges: int = 0
    skipped_nonradius_edges: int = 0
    mesh_cache_invalidated: bool = False

    @property
    def n_examined(self) -> int:
        return len(self.edge_results)

    @property
    def n_violating(self) -> int:
        return sum(r.before_max_rk > 0 and r.repair is not None for r in self.edge_results)

    @property
    def n_repaired(self) -> int:
        return sum(r.changed for r in self.edge_results)

    @property
    def n_failed(self) -> int:
        return sum(r.status in {"repair_failed", "error"} for r in self.edge_results)

    def summary(self, include_regions: bool = False, include_candidates: bool = False) -> Dict[str, Any]:
        return {
            "inplace": self.inplace,
            "n_examined": self.n_examined,
            "n_violating": self.n_violating,
            "n_repaired": self.n_repaired,
            "n_failed": self.n_failed,
            "skipped_bifurcation_edges": self.skipped_bifurcation_edges,
            "skipped_nonradius_edges": self.skipped_nonradius_edges,
            "mesh_cache_invalidated": self.mesh_cache_invalidated,
            "edges": [
                r.summary(include_regions, include_candidates)
                for r in self.edge_results
            ],
        }


# -----------------------------------------------------------------------------
# VascularMD interoperability
# -----------------------------------------------------------------------------


def _get_vmd_spline_class():
    """Import VascularMD's Spline class from either common repository layout."""
    try:
        from Spline import Spline  # type: ignore
        return Spline
    except ImportError as first_error:
        try:
            from vascularmd.Spline import Spline  # type: ignore
            return Spline
        except ImportError:
            raise ImportError(
                "VascularMD could not be imported. Ensure its source directory is "
                "on PYTHONPATH so that `from Spline import Spline` works, or install "
                "a package exposing `vascularmd.Spline`."
            ) from first_error


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    # np.trapezoid is the current spelling; np.trapz keeps compatibility with
    # older NumPy versions.
    fn = getattr(np, "trapezoid", np.trapz)
    return float(fn(y, x))


def spline_state(spline: Any) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (knot vector, control points, degree) from a VascularMD spline."""
    curve = spline.get_spl()
    degree = int(getattr(curve, "degree", curve.order - 1))
    U = np.asarray(spline.get_knot(), dtype=float)
    P = np.asarray(spline.get_control_points(), dtype=float)
    if P.ndim != 2 or P.shape[1] < 4:
        raise ValueError(
            "Curvature repair requires a VascularMD vessel spline with 4-D "
            "control points (x, y, z, radius)."
        )
    return U, P, degree


def scipy_bspline_from_vmd(spline: Any) -> SciPyBSpline:
    U, P, degree = spline_state(spline)
    return SciPyBSpline(U, P, degree, axis=0, extrapolate=False)


def vmd_spline_from_scipy(bs: SciPyBSpline) -> Any:
    Spline = _get_vmd_spline_class()
    return Spline(
        np.asarray(bs.c, dtype=float),
        np.asarray(bs.t, dtype=float).tolist(),
        int(bs.k) + 1,
    )


def clone_geometry(spline: Any) -> Any:
    U, P, degree = spline_state(spline)
    cloned = _get_vmd_spline_class()(P.copy(), U.tolist(), degree + 1)
    if hasattr(spline, "_curvature_repair_reference_t"):
        cloned._curvature_repair_reference_t = np.asarray(  # type: ignore[attr-defined]
            spline._curvature_repair_reference_t, dtype=float  # type: ignore[attr-defined]
        ).copy()
    return cloned


def _chord_length_parameters_xyz(data: np.ndarray) -> np.ndarray:
    xyz = np.asarray(data, dtype=float)[:, :3]
    if len(xyz) == 1:
        return np.array([0.0])
    ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(ds)]
    if s[-1] <= 1e-14:
        return np.linspace(0.0, 1.0, len(xyz))
    return s / s[-1]


def infer_reference_parameters(spline: Any) -> np.ndarray:
    """Recover original VascularMD observation parameters for RIC point counting.

    Normal VascularMD fitted vessel splines retain their spatial Model object and
    therefore its parameter vector. A chord-length fallback is used if data are
    available but the Model object is not. Repaired splines created by this module
    also retain the original parameter vector as lightweight provenance.
    """
    if hasattr(spline, "_curvature_repair_reference_t"):
        t = np.asarray(spline._curvature_repair_reference_t, dtype=float)  # type: ignore[attr-defined]
        if t.size:
            return t.copy()

    if hasattr(spline, "get_model"):
        models = spline.get_model()
        if models is not None and len(models) > 0 and models[0] is not None:
            t = np.asarray(models[0].get_t(), dtype=float)
            if t.size:
                return t.copy()

    if hasattr(spline, "get_data"):
        data = spline.get_data()
        if data is not None:
            data = np.asarray(data, dtype=float)
            if data.ndim == 2 and len(data) > 0:
                return _chord_length_parameters_xyz(data)

    raise ValueError(
        "Cannot recover the original VascularMD observation parameters for this "
        "spline. Pass `reference_t=` explicitly to repair_spline()."
    )


# -----------------------------------------------------------------------------
# Differential geometry and continuous q(t)=r*kappa search
# -----------------------------------------------------------------------------


def rk_product_bspline(bs: SciPyBSpline, t: Sequence[float] | np.ndarray) -> np.ndarray:
    t = np.atleast_1d(np.asarray(t, dtype=float))
    val = np.asarray(bs(t), dtype=float)
    d1 = np.asarray(bs.derivative(1)(t), dtype=float)[:, :3]
    d2 = np.asarray(bs.derivative(2)(t), dtype=float)[:, :3]
    speed = np.linalg.norm(d1, axis=1)
    curvature = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(speed**3, 1e-15)
    return val[:, 3] * curvature


def nonzero_knot_spans(bs: SciPyBSpline, window: Tuple[float, float] = (0.0, 1.0)) -> List[Tuple[float, float]]:
    a, b = map(float, window)
    unique = np.unique(np.asarray(bs.t, dtype=float))
    spans: List[Tuple[float, float]] = []
    for left, right in zip(unique[:-1], unique[1:]):
        lo = max(float(left), a)
        hi = min(float(right), b)
        if hi - lo > 1e-14:
            spans.append((lo, hi))
    return spans


def refined_rk_local_maxima_bspline(
    bs: SciPyBSpline,
    window: Tuple[float, float] = (0.0, 1.0),
    samples_per_span: int = 21,
    xatol: float = 1e-10,
) -> List[Tuple[float, float]]:
    """Find candidate local maxima of r*kappa span-by-span and refine them."""
    samples_per_span = max(int(samples_per_span), 5)
    candidates: List[Tuple[float, float]] = []

    def q_scalar(x: float) -> float:
        return float(rk_product_bspline(bs, [x])[0])

    for lo, hi in nonzero_knot_spans(bs, window):
        x = np.linspace(lo, hi, samples_per_span)
        q = rk_product_bspline(bs, x)
        candidates.append((float(x[0]), float(q[0])))
        candidates.append((float(x[-1]), float(q[-1])))

        for j in range(1, len(x) - 1):
            if q[j] >= q[j - 1] and q[j] >= q[j + 1]:
                left, right = float(x[j - 1]), float(x[j + 1])
                if right - left <= 1e-14:
                    continue
                res = minimize_scalar(
                    lambda z: -q_scalar(float(z)),
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": xatol},
                )
                t_peak = float(res.x)
                candidates.append((t_peak, q_scalar(t_peak)))

    candidates.sort(key=lambda z: z[0])
    merged: List[List[float]] = []
    merge_tol = max(10.0 * xatol, 1e-9)
    for t_peak, q_peak in candidates:
        if not merged or abs(t_peak - merged[-1][0]) > merge_tol:
            merged.append([t_peak, q_peak])
        elif q_peak > merged[-1][1]:
            merged[-1] = [t_peak, q_peak]

    return [(float(t), float(q)) for t, q in merged]


def refined_max_rk_bspline(
    bs: SciPyBSpline,
    cfg: RepairConfig,
    window: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float]:
    maxima = refined_rk_local_maxima_bspline(
        bs,
        window=window,
        samples_per_span=cfg.span_search_samples,
        xatol=cfg.span_search_xatol,
    )
    if not maxima:
        t = float(window[0])
        return float(rk_product_bspline(bs, [t])[0]), t
    t_peak, q_peak = max(maxima, key=lambda z: z[1])
    return float(q_peak), float(t_peak)


def max_radius_curvature(
    spline: Any,
    config: Optional[RepairConfig] = None,
    window: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float]:
    """Return continuous-search estimate ``(max(r*kappa), t_at_max)``."""
    cfg = config or RepairConfig()
    return refined_max_rk_bspline(scipy_bspline_from_vmd(spline), cfg, window)


def detect_violation_intervals(spline: Any, cfg: RepairConfig) -> List[ViolationInterval]:
    """Locate connected regions above the numerical acceptance threshold."""
    bs = scipy_bspline_from_vmd(spline)
    threshold = cfg.acceptance_threshold
    t = np.linspace(0.0, 1.0, cfg.detection_samples)
    f = rk_product_bspline(bs, t) - threshold
    mask = f > 0.0

    def root_fun(x: float) -> float:
        return float(rk_product_bspline(bs, [x])[0] - threshold)

    intervals: List[ViolationInterval] = []
    if np.any(mask):
        ids = np.where(mask)[0]
        groups = np.split(ids, np.where(np.diff(ids) > 1)[0] + 1)

        for g in groups:
            i0, i1 = int(g[0]), int(g[-1])
            t_lo, t_hi = float(t[i0]), float(t[i1])

            if i0 > 0:
                try:
                    t_lo = float(brentq(root_fun, t[i0 - 1], t[i0]))
                except ValueError:
                    pass
            if i1 < len(t) - 1:
                try:
                    t_hi = float(brentq(root_fun, t[i1], t[i1 + 1]))
                except ValueError:
                    pass

            local_maxima = refined_rk_local_maxima_bspline(
                bs,
                window=(t_lo, t_hi),
                samples_per_span=cfg.span_search_samples,
                xatol=cfg.span_search_xatol,
            )
            if local_maxima:
                t_peak, peak = max(local_maxima, key=lambda z: z[1])
            else:
                local_t = t[i0 : i1 + 1]
                local_q = rk_product_bspline(bs, local_t)
                j = int(np.argmax(local_q))
                t_peak, peak = float(local_t[j]), float(local_q[j])

            intervals.append(ViolationInterval(t_lo, t_hi, float(t_peak), float(peak)))

    # Narrow-peak fallback in case the coarse connected-component grid misses it.
    qmax, tmax = refined_max_rk_bspline(bs, cfg)
    if qmax > threshold and not intervals:
        knots = np.unique(np.asarray(bs.t, dtype=float))
        j = max(0, min(len(knots) - 2, np.searchsorted(knots, tmax, side="right") - 1))
        intervals.append(
            ViolationInterval(float(knots[j]), float(knots[j + 1]), tmax, qmax)
        )

    return intervals


# -----------------------------------------------------------------------------
# Candidate spline spaces
# -----------------------------------------------------------------------------


def expand_to_knot_spans(
    spline: Any,
    t_lo: float,
    t_hi: float,
    pad_spans: int,
) -> Tuple[float, float]:
    knots = np.unique(np.asarray(spline.get_knot(), dtype=float))
    left = max(0, np.searchsorted(knots, t_lo, side="right") - 1 - pad_spans)
    right = min(len(knots) - 1, np.searchsorted(knots, t_hi, side="left") + pad_spans)
    if right <= left:
        right = min(len(knots) - 1, left + 1)
    return float(knots[left]), float(knots[right])


def choose_extra_knots(spline: Any, interval: ViolationInterval, m: int) -> List[float]:
    if m == 0:
        return []
    a, b = expand_to_knot_spans(spline, interval.t_lo, interval.t_hi, 0)
    if b - a < 1e-10:
        knots = np.unique(np.asarray(spline.get_knot(), dtype=float))
        j = max(0, min(len(knots) - 2, np.searchsorted(knots, interval.t_peak, side="right") - 1))
        a, b = float(knots[j]), float(knots[j + 1])

    proposed = np.linspace(a, b, m + 2)[1:-1]
    existing = list(np.asarray(spline.get_knot(), dtype=float))
    chosen: List[float] = []
    for x in proposed:
        x = float(x)
        if min(abs(x - z) for z in existing + chosen) < 1e-9:
            boundaries = sorted([a, b] + [z for z in existing + chosen if a < z < b])
            gaps = [
                (boundaries[i + 1] - boundaries[i], 0.5 * (boundaries[i + 1] + boundaries[i]))
                for i in range(len(boundaries) - 1)
            ]
            x = max(gaps, key=lambda z: z[0])[1]
        chosen.append(float(x))
    return sorted(chosen)


def insert_simple_knots_exact(spline: Any, knot_locations: Sequence[float]) -> Any:
    """Exact B-spline knot insertion; initial geometry is unchanged."""
    bs = scipy_bspline_from_vmd(spline)
    existing = list(np.asarray(bs.t, dtype=float))
    for u in sorted(map(float, knot_locations)):
        if not (0.0 < u < 1.0):
            raise ValueError(f"Internal knot must lie in (0, 1), got {u}.")
        if np.min(np.abs(np.asarray(existing) - u)) < 1e-9:
            raise ValueError(f"Knot {u} already exists; repeated insertion is disabled.")
        bs = bs.insert_knot(u, m=1)
        existing.append(u)
    return vmd_spline_from_scipy(bs)


def choose_knots_to_remove(
    spline: Any,
    interval: ViolationInterval,
    m: int,
    cfg: RepairConfig,
) -> List[float]:
    """Choose nearby simple internal knots; repeated/end knots are never removed."""
    if m == 0:
        return []
    U = np.asarray(spline.get_knot(), dtype=float)
    vals, counts = np.unique(U, return_counts=True)
    simple_internal = [float(u) for u, c in zip(vals, counts) if 0.0 < u < 1.0 and c == 1]
    if not simple_internal:
        return []

    a, b = expand_to_knot_spans(
        spline,
        interval.t_lo,
        interval.t_hi,
        cfg.knot_removal_search_padding_spans,
    )
    local = [u for u in simple_internal if a - 1e-12 <= u <= b + 1e-12]
    local.sort(key=lambda u: (abs(u - interval.t_peak), u))
    return local[:m]


def _endpoint_handle_lengths_for_basis(
    reference_bs: SciPyBSpline,
    U: np.ndarray,
    degree: int,
    n_ctrl: int,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    d0 = np.asarray(reference_bs.derivative(1)(0.0), dtype=float)[:3]
    d1 = np.asarray(reference_bs.derivative(1)(1.0), dtype=float)[:3]
    n0, n1 = np.linalg.norm(d0), np.linalg.norm(d1)
    if n0 <= 1e-14 or n1 <= 1e-14:
        raise ValueError("Reference endpoint derivative is too small for G1 parameterization.")

    start_denom = float(U[degree + 1] - U[1])
    end_denom = float(U[-2] - U[n_ctrl - 1])
    if start_denom <= 0.0 or end_denom <= 0.0:
        raise ValueError("Degenerate clamped knot vector at endpoint.")

    start_factor = degree / start_denom
    end_factor = degree / end_denom
    return n0 / start_factor, n1 / end_factor, d0 / n0, d1 / n1


def remove_simple_knots_approx(
    reference: Any,
    knot_locations: Sequence[float],
    cfg: RepairConfig,
) -> Any:
    """Create a reduced-basis approximation after deleting selected simple knots."""
    ref_bs = scipy_bspline_from_vmd(reference)
    U = list(np.asarray(ref_bs.t, dtype=float))
    degree = int(ref_bs.k)

    for u in map(float, knot_locations):
        idx = [i for i, z in enumerate(U) if abs(z - u) < 1e-9]
        if len(idx) != 1:
            raise ValueError(f"Knot {u} is not a removable simple knot.")
        U.pop(idx[0])

    U_arr = np.asarray(U, dtype=float)
    n_ctrl = len(U_arr) - degree - 1
    if n_ctrl < degree + 1:
        raise ValueError("Knot removal would leave too few control points for the spline degree.")

    n_fit = max(
        cfg.knot_removal_fit_samples,
        cfg.knot_removal_fit_samples_per_control * n_ctrl,
    )
    t_fit = np.unique(
        np.r_[np.linspace(0.0, 1.0, n_fit), np.unique(U_arr), np.unique(ref_bs.t)]
    )
    A = SciPyBSpline.design_matrix(t_fit, U_arr, degree, extrapolate=False).toarray()
    Y = np.asarray(ref_bs(t_fit), dtype=float)
    Q, *_ = np.linalg.lstsq(A, Y, rcond=None)

    Q[0, :] = np.asarray(ref_bs(0.0), dtype=float)
    Q[-1, :] = np.asarray(ref_bs(1.0), dtype=float)

    alpha0, beta0, T0, T1 = _endpoint_handle_lengths_for_basis(
        ref_bs, U_arr, degree, n_ctrl
    )
    Q[1, :3] = Q[0, :3] + alpha0 * T0
    Q[-2, :3] = Q[-1, :3] - beta0 * T1

    return vmd_spline_from_scipy(
        SciPyBSpline(U_arr, Q, degree, axis=0, extrapolate=False)
    )


def build_candidate_basis(
    reference: Any,
    interval: ViolationInterval,
    knot_delta: int,
    cfg: RepairConfig,
) -> Tuple[Any, List[float], List[float]]:
    if knot_delta > 0:
        inserted = choose_extra_knots(reference, interval, knot_delta)
        return insert_simple_knots_exact(reference, inserted), inserted, []
    if knot_delta < 0:
        requested = -knot_delta
        removed = choose_knots_to_remove(reference, interval, requested, cfg)
        if len(removed) != requested:
            raise ValueError(
                f"Requested removal of {requested} knot(s), but only {len(removed)} "
                "local simple knot(s) are available."
            )
        return remove_simple_knots_approx(reference, removed, cfg), [], removed
    return clone_geometry(reference), [], []


def active_control_indices(spline: Any, core_window: Tuple[float, float]) -> List[int]:
    """Controls whose basis support overlaps the local repair window, excluding endpoints."""
    U, P, degree = spline_state(spline)
    lo, hi = core_window
    active: List[int] = []
    for i in range(len(P)):
        support_lo = U[i]
        support_hi = U[i + degree + 1]
        if support_hi >= lo - 1e-12 and support_lo <= hi + 1e-12:
            active.append(i)
    return [i for i in active if i not in {0, len(P) - 1}]


def affected_window(spline: Any, active: Sequence[int]) -> Tuple[float, float]:
    U, _, degree = spline_state(spline)
    return float(min(U[i] for i in active)), float(max(U[i + degree + 1] for i in active))


def repair_objective_window(
    reference: Any,
    interval: ViolationInterval,
    cfg: RepairConfig,
) -> Tuple[float, float]:
    """Fixed reference support used to compare J across different knot candidates."""
    core = expand_to_knot_spans(
        reference, interval.t_lo, interval.t_hi, cfg.repair_padding_spans
    )
    ref_active = active_control_indices(reference, core)
    if not ref_active:
        return core
    return affected_window(reference, ref_active)


# -----------------------------------------------------------------------------
# Constrained minimum-J repair in one candidate spline space
# -----------------------------------------------------------------------------


def boundary_invariants(reference: Any, candidate: Any, cfg: RepairConfig) -> bool:
    """Check fixed endpoint values and preserved positive G1 tangent directions."""
    for t in (0.0, 1.0):
        p0 = np.asarray(reference.point(t, radius=True), dtype=float)
        p1 = np.asarray(candidate.point(t, radius=True), dtype=float)
        if np.linalg.norm(p0 - p1) > cfg.boundary_value_tolerance:
            return False

    b0 = scipy_bspline_from_vmd(reference)
    b1 = scipy_bspline_from_vmd(candidate)
    for t in (0.0, 1.0):
        d0 = np.asarray(b0.derivative(1)(t), dtype=float)[:3]
        d1 = np.asarray(b1.derivative(1)(t), dtype=float)[:3]
        n0, n1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if n0 <= cfg.min_speed or n1 <= cfg.min_speed:
            return False
        cosine = float(np.dot(d0 / n0, d1 / n1))
        if cosine <= 0.0:
            return False
        if 1.0 - cosine > cfg.tangent_direction_tolerance:
            return False
    return True


def optimize_local_candidate(
    reference: Any,
    interval: ViolationInterval,
    knot_delta: int,
    cfg: RepairConfig,
) -> CandidateResult:
    try:
        candidate0, inserted_knots, removed_knots = build_candidate_basis(
            reference, interval, knot_delta, cfg
        )
    except Exception as exc:
        return CandidateResult(
            knot_delta=knot_delta,
            inserted_knots=[],
            removed_knots=[],
            spline=None,
            feasible=False,
            optimizer_success=False,
            optimizer_message=f"Candidate basis unavailable: {type(exc).__name__}: {exc}",
            repair_cost=np.inf,
            max_rk=np.inf,
            t_max_rk=np.nan,
        )

    core = expand_to_knot_spans(
        reference, interval.t_lo, interval.t_hi, cfg.repair_padding_spans
    )
    active = active_control_indices(candidate0, core)
    if not active:
        return CandidateResult(
            knot_delta=knot_delta,
            inserted_knots=inserted_knots,
            removed_knots=removed_knots,
            spline=None,
            feasible=False,
            optimizer_success=False,
            optimizer_message="No movable local control points in the repair support.",
            repair_cost=np.inf,
            max_rk=np.inf,
            t_max_rk=np.nan,
            control_count=len(candidate0.get_control_points()),
            active_controls=0,
        )

    U, P0, degree = spline_state(candidate0)
    n_ctrl = len(P0)
    affected = affected_window(candidate0, active)
    objective_window = repair_objective_window(reference, interval, cfg)
    ref_bs = scipy_bspline_from_vmd(reference)

    alpha_base, beta_base, T0_hat, T1_hat = _endpoint_handle_lengths_for_basis(
        ref_bs, U, degree, n_ctrl
    )

    start_handle_active = 1 in active
    end_handle_active = (n_ctrl - 2) in active

    # Each tuple is (kind, control index, coordinate index). Only these entries
    # appear in x0, so len(x0) is exactly dim(theta_k); fixed endpoint values are
    # not counted.
    specs: List[Tuple[str, Optional[int], Optional[int]]] = []
    x0: List[float] = []
    bounds: List[Tuple[Optional[float], Optional[float]]] = []

    if start_handle_active:
        specs.append(("start_scale", None, None))
        x0.append(1.0)
        bounds.append((cfg.endpoint_tangent_scale_min, cfg.endpoint_tangent_scale_max))

    if end_handle_active:
        specs.append(("end_scale", None, None))
        x0.append(1.0)
        bounds.append((cfg.endpoint_tangent_scale_min, cfg.endpoint_tangent_scale_max))

    for i in active:
        if i in (1, n_ctrl - 2):
            # Spatial coordinates are represented by the one-dimensional tangent
            # scale above. Radius remains independently movable.
            specs.append(("radius", i, 3))
            x0.append(float(P0[i, 3]))
            bounds.append((cfg.min_radius, None))
        else:
            for d in range(3):
                specs.append(("coord", i, d))
                x0.append(float(P0[i, d]))
                bounds.append((None, None))
            specs.append(("radius", i, 3))
            x0.append(float(P0[i, 3]))
            bounds.append((cfg.min_radius, None))

    x = np.asarray(x0, dtype=float)

    def unpack(xvec: np.ndarray) -> np.ndarray:
        P = P0.copy()
        P[0, :] = np.asarray(ref_bs(0.0), dtype=float)
        P[-1, :] = np.asarray(ref_bs(1.0), dtype=float)

        start_scale = 1.0
        end_scale = 1.0
        for value, spec in zip(xvec, specs):
            kind, i, d = spec
            if kind == "start_scale":
                start_scale = float(value)
            elif kind == "end_scale":
                end_scale = float(value)
            elif kind in {"coord", "radius"}:
                assert i is not None and d is not None
                P[i, d] = float(value)

        # G1: endpoint tangent directions are fixed; only magnitudes can change.
        if start_handle_active:
            P[1, :3] = P[0, :3] + alpha_base * start_scale * T0_hat
        else:
            P[1, :3] = P[0, :3] + alpha_base * T0_hat

        if end_handle_active:
            P[-2, :3] = P[-1, :3] - beta_base * end_scale * T1_hat
        else:
            P[-2, :3] = P[-1, :3] - beta_base * T1_hat
        return P

    def make_bs(xvec: np.ndarray) -> SciPyBSpline:
        return SciPyBSpline(U, unpack(xvec), degree, axis=0, extrapolate=False)

    # J is measured on one fixed support derived from the reference spline, not
    # a candidate-dependent support. That makes J comparable across knot counts.
    t_obj = np.linspace(objective_window[0], objective_window[1], cfg.objective_samples)
    ref_val = np.asarray(ref_bs(t_obj), dtype=float)
    ref_speed = np.linalg.norm(np.asarray(ref_bs.derivative(1)(t_obj))[:, :3], axis=1)
    physical_length = max(_trapz(ref_speed, t_obj), 1e-12)

    def objective(xvec: np.ndarray) -> float:
        val = np.asarray(make_bs(xvec)(t_obj), dtype=float)
        spatial = np.sum((val[:, :3] - ref_val[:, :3]) ** 2, axis=1)
        radial = ((val[:, 3] - ref_val[:, 3]) / cfg.sigma_radius_over_spatial) ** 2
        return _trapz((spatial + radial) * ref_speed, t_obj) / physical_length

    t_constraint = np.unique(
        np.r_[
            np.linspace(affected[0], affected[1], cfg.initial_constraint_samples),
            [u for u in np.unique(U) if affected[0] <= u <= affected[1]],
            interval.t_peak,
        ]
    )

    def physical_constraint(xvec: np.ndarray) -> np.ndarray:
        # Deliberately strict 0.95-style target. Numerical tolerance is used only
        # by the outer continuous validation.
        return cfg.physical_threshold - rk_product_bspline(make_bs(xvec), t_constraint)

    def speed_constraint(xvec: np.ndarray) -> np.ndarray:
        d1 = np.asarray(make_bs(xvec).derivative(1)(t_constraint), dtype=float)[:, :3]
        return np.linalg.norm(d1, axis=1) - cfg.min_speed

    def radius_constraint(xvec: np.ndarray) -> np.ndarray:
        return np.asarray(make_bs(xvec)(t_constraint), dtype=float)[:, 3] - cfg.min_radius

    result = None
    rounds_used = 0

    for round_idx in range(cfg.max_constraint_refinement_rounds):
        rounds_used = round_idx + 1
        result = minimize(
            objective,
            x,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": physical_constraint},
                {"type": "ineq", "fun": speed_constraint},
                {"type": "ineq", "fun": radius_constraint},
            ],
            options={
                "maxiter": cfg.optimizer_maxiter,
                "ftol": cfg.optimizer_ftol,
                "disp": False,
            },
        )
        x = np.asarray(result.x, dtype=float)
        bs = make_bs(x)

        maxima = refined_rk_local_maxima_bspline(
            bs,
            window=affected,
            samples_per_span=cfg.span_search_samples,
            xatol=cfg.span_search_xatol,
        )
        if maxima:
            t_max, max_rk = max(maxima, key=lambda z: z[1])
        else:
            max_rk, t_max = refined_max_rk_bspline(bs, cfg, affected)

        if max_rk <= cfg.acceptance_threshold:
            break

        violating = [
            (t_peak, q_peak)
            for t_peak, q_peak in maxima
            if q_peak > cfg.acceptance_threshold
        ]
        violating.sort(key=lambda z: z[1], reverse=True)
        violating = violating[: cfg.max_new_peak_constraints_per_round]

        new_t: List[float] = []
        for t_peak, _ in violating:
            if np.min(np.abs(t_constraint - t_peak)) > cfg.constraint_point_merge_tolerance:
                new_t.append(float(t_peak))

        if not new_t:
            break
        t_constraint = np.unique(np.r_[t_constraint, new_t])

    final_bs = make_bs(x)
    final_spline = vmd_spline_from_scipy(final_bs)
    max_rk, t_max = refined_max_rk_bspline(final_bs, cfg, affected)

    feasible = (
        result is not None
        and result.success
        and max_rk <= cfg.acceptance_threshold
        and boundary_invariants(reference, final_spline, cfg)
    )

    return CandidateResult(
        knot_delta=knot_delta,
        inserted_knots=inserted_knots,
        removed_knots=removed_knots,
        spline=final_spline,
        feasible=bool(feasible),
        optimizer_success=bool(result.success if result is not None else False),
        optimizer_message=str(result.message if result is not None else "optimizer not run"),
        repair_cost=float(objective(x)),
        max_rk=float(max_rk),
        t_max_rk=float(t_max),
        control_count=n_ctrl,
        free_parameter_count=len(x0),
        active_controls=len(active),
        start_tangent_scale_active=start_handle_active,
        end_tangent_scale_active=end_handle_active,
        refinement_rounds=rounds_used,
        constraint_points=len(t_constraint),
    )


# -----------------------------------------------------------------------------
# Repair-information score and multi-region repair
# -----------------------------------------------------------------------------


def repair_information_point_count(
    original_t: np.ndarray,
    objective_window: Tuple[float, float],
) -> int:
    """Count original VascularMD observations in the fixed repair-support window."""
    original_t = np.asarray(original_t, dtype=float)
    lo, hi = map(float, objective_window)
    count = int(
        np.count_nonzero((original_t >= lo - 1e-12) & (original_t <= hi + 1e-12))
    )
    return max(count, 1)


def score_candidate_aic_j(
    candidate: CandidateResult,
    m_I: int,
    cfg: RepairConfig,
) -> CandidateResult:
    """AIC-like score balancing optimized repair loss against free parameters."""
    if not candidate.feasible or candidate.spline is None:
        return candidate
    candidate.aic_j_points = int(m_I)
    J_star = max(float(candidate.repair_cost), float(cfg.aic_j_log_floor))
    p_k = int(candidate.free_parameter_count)
    candidate.aic_j_score = float(m_I * math.log(J_star) + 2.0 * p_k)
    return candidate


def select_best_feasible_candidate(
    candidates: Sequence[CandidateResult],
) -> Optional[CandidateResult]:
    feasible = [c for c in candidates if c.feasible and np.isfinite(c.aic_j_score)]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda c: (c.aic_j_score, c.repair_cost, abs(c.knot_delta), c.control_count),
    )


def _repair_spline_local_curvature(
    original: Any,
    original_t: np.ndarray,
    cfg: RepairConfig,
) -> SplineRepairResult:
    current = clone_geometry(original)
    before_max, _ = max_radius_curvature(current, cfg)
    reports: List[RegionRepairReport] = []
    net_knot_delta = 0

    for _ in range(cfg.max_repair_regions):
        violations = detect_violation_intervals(current, cfg)
        if not violations:
            break

        interval = max(violations, key=lambda v: v.peak_rk)
        region_reference = clone_geometry(current)
        objective_window = repair_objective_window(region_reference, interval, cfg)
        m_I = repair_information_point_count(original_t, objective_window)
        candidates: List[CandidateResult] = []

        for knot_delta in range(-cfg.max_removed_knots, cfg.max_extra_knots + 1):
            candidate = optimize_local_candidate(region_reference, interval, knot_delta, cfg)
            score_candidate_aic_j(candidate, m_I, cfg)
            candidates.append(candidate)

        chosen = select_best_feasible_candidate(candidates)
        if chosen is None:
            reports.append(
                RegionRepairReport(
                    before_peak_rk=interval.peak_rk,
                    interval=(interval.t_lo, interval.t_hi),
                    chosen_knot_delta=None,
                    chosen_control_count=None,
                    chosen_aic_j_score=np.inf,
                    chosen_repair_cost=np.inf,
                    after_local_max_rk=interval.peak_rk,
                    status="FAILED: no physically feasible candidate",
                    candidates=candidates,
                )
            )
            break

        current = chosen.spline
        net_knot_delta += chosen.knot_delta
        reports.append(
            RegionRepairReport(
                before_peak_rk=interval.peak_rk,
                interval=(interval.t_lo, interval.t_hi),
                chosen_knot_delta=chosen.knot_delta,
                chosen_control_count=chosen.control_count,
                chosen_aic_j_score=chosen.aic_j_score,
                chosen_repair_cost=chosen.repair_cost,
                after_local_max_rk=chosen.max_rk,
                status="repaired",
                candidates=candidates,
            )
        )

    after_max, _ = max_radius_curvature(current, cfg)
    success = after_max <= cfg.acceptance_threshold
    attempted = len(reports) > 0
    repaired = bool(success and attempted)

    # Lightweight provenance for a future repeat call; this does not pretend that
    # the repaired spline is itself a new VascularMD statistical Model fit.
    current._curvature_repair_reference_t = np.asarray(original_t, dtype=float).copy()  # type: ignore[attr-defined]

    return SplineRepairResult(
        spline=current,
        before_max_rk=float(before_max),
        after_max_rk=float(after_max),
        repair_attempted=attempted,
        repaired=repaired,
        success=bool(success),
        net_knot_delta=net_knot_delta,
        regions=reports,
    )


def repair_spline(
    spline: Any,
    config: Optional[RepairConfig] = None,
    reference_t: Optional[Sequence[float]] = None,
) -> SplineRepairResult:
    """Repair one already-fitted VascularMD vessel spline.

    Parameters
    ----------
    spline:
        A 4-D VascularMD ``Spline`` containing x, y, z, radius control points.
    config:
        Optional :class:`RepairConfig`.
    reference_t:
        Parameter values of the original observations. Normally inferred from the
        VascularMD spline's spatial ``Model``. Supply explicitly only for splines
        that no longer retain VascularMD model metadata.

    Returns
    -------
    SplineRepairResult
        The input spline is never mutated by this function.
    """
    cfg = config or RepairConfig()
    before_max, _ = max_radius_curvature(spline, cfg)

    if before_max <= cfg.acceptance_threshold:
        out = clone_geometry(spline)
        if reference_t is not None:
            out._curvature_repair_reference_t = np.asarray(reference_t, dtype=float).copy()  # type: ignore[attr-defined]
        return SplineRepairResult(
            spline=out,
            before_max_rk=float(before_max),
            after_max_rk=float(before_max),
            repair_attempted=False,
            repaired=False,
            success=True,
            net_knot_delta=0,
            regions=[],
        )

    if reference_t is None:
        original_t = infer_reference_parameters(spline)
    else:
        original_t = np.asarray(reference_t, dtype=float)
        if original_t.ndim != 1 or original_t.size == 0:
            raise ValueError("reference_t must be a non-empty one-dimensional sequence.")

    return _repair_spline_local_curvature(spline, original_t, cfg)


# -----------------------------------------------------------------------------
# VascularMD arterial-tree adapter
# -----------------------------------------------------------------------------


def iter_vascularmd_vessel_edges(tree: Any) -> Iterable[Tuple[Tuple[Any, Any], Any]]:
    """Yield ordinary VascularMD vessel edges, excluding bifurcation trajectories.

    VascularMD's own cross-section code treats an edge as an ordinary vessel edge
    when neither endpoint node has type ``"bif"``. The same rule is used here.
    """
    if not hasattr(tree, "get_model_graph"):
        raise TypeError("tree must expose VascularMD's get_model_graph() method.")
    G = tree.get_model_graph()
    if G is None:
        raise ValueError("The VascularMD tree has no model graph. Run model_network() first.")

    for edge in G.edges():
        if G.nodes[edge[0]].get("type") == "bif" or G.nodes[edge[1]].get("type") == "bif":
            continue
        spline = G.edges[edge].get("spline")
        if spline is not None:
            yield edge, spline


def _invalidate_vascularmd_mesh_cache(tree: Any, G: Any) -> None:
    """Keep the repaired model graph but force downstream VascularMD meshing to rebuild."""
    if hasattr(tree, "set_model_graph"):
        # Do not propagate the repaired model back into the original full/topological
        # centerline graphs. The package is a model-level post-processor.
        tree.set_model_graph(G, replace=True, down_replace=False)
    else:
        tree._model_graph = G
        tree._crsec_graph = None

    # VascularMD's set_model_graph clears _crsec_graph but does not necessarily
    # clear already-created surface/volume meshes in every version.
    if hasattr(tree, "_crsec_graph"):
        tree._crsec_graph = None
    if hasattr(tree, "_surface_mesh"):
        tree._surface_mesh = None
    if hasattr(tree, "_volume_mesh"):
        tree._volume_mesh = None


def repair_vascularmd_tree(
    tree: Any,
    config: Optional[RepairConfig] = None,
    *,
    inplace: bool = False,
    raise_on_failure: bool = False,
    verbose: bool = False,
) -> NetworkRepairReport:
    """Repair all offending non-bifurcating vessel splines in a VascularMD tree.

    The function assumes ``tree.model_network(...)`` has already been run.
    Bifurcation-trajectory edges (edges incident on a ``type == 'bif'`` node) are
    left untouched. Ordinary vessel splines are continuously screened using
    ``max(r*kappa)``; only those above ``config.acceptance_threshold`` enter the
    constrained repair.

    Successful repaired splines replace the corresponding ``model_graph`` edge
    splines. Failed repairs leave the original spline in place. If at least one
    edge changes, VascularMD's downstream cross-section/surface/volume caches are
    invalidated so the user can call ``compute_cross_sections`` / ``mesh_volume``
    normally afterwards.

    Notes
    -----
    This function deliberately does not remesh and does not propagate repaired
    geometry back into the original SWC/full graph. It is a post-processing step
    on the fitted VascularMD model graph.
    """
    cfg = config or RepairConfig()

    if inplace:
        target = tree
    else:
        try:
            target = copy.deepcopy(tree)
        except Exception as exc:
            raise RuntimeError(
                "Could not deep-copy the VascularMD tree. Retry with inplace=True."
            ) from exc

    if not hasattr(target, "get_model_graph"):
        raise TypeError("tree must expose VascularMD's get_model_graph() method.")
    G = target.get_model_graph()
    if G is None:
        raise ValueError("The VascularMD tree has no model graph. Run model_network() first.")

    edge_results: List[EdgeRepairResult] = []
    skipped_bif = 0
    skipped_nonradius = 0
    changed_any = False

    for edge in list(G.edges()):
        node0_type = G.nodes[edge[0]].get("type")
        node1_type = G.nodes[edge[1]].get("type")
        if node0_type == "bif" or node1_type == "bif":
            skipped_bif += 1
            continue

        spline = G.edges[edge].get("spline")
        if spline is None:
            continue

        try:
            _, P, _ = spline_state(spline)
            if P.shape[1] < 4:
                skipped_nonradius += 1
                continue

            before_max, _ = max_radius_curvature(spline, cfg)
            if verbose:
                print(f"edge {edge}: max(r*kappa)={before_max:.6f}")

            if before_max <= cfg.acceptance_threshold:
                edge_results.append(
                    EdgeRepairResult(
                        edge=edge,
                        status="already_admissible",
                        before_max_rk=float(before_max),
                        after_max_rk=float(before_max),
                    )
                )
                continue

            reference_t = infer_reference_parameters(spline)
            repair = _repair_spline_local_curvature(spline, reference_t, cfg)

            if repair.success and repair.repaired:
                G.edges[edge]["spline"] = repair.spline
                changed_any = True
                status = "repaired"
            else:
                status = "repair_failed"

            edge_results.append(
                EdgeRepairResult(
                    edge=edge,
                    status=status,
                    before_max_rk=float(before_max),
                    after_max_rk=float(repair.after_max_rk),
                    repair=repair,
                )
            )

            if verbose:
                print(
                    f"  -> {status}: {before_max:.6f} -> "
                    f"{repair.after_max_rk:.6f}, net knot delta={repair.net_knot_delta:+d}"
                )

            if status == "repair_failed" and raise_on_failure:
                raise RuntimeError(
                    f"No accepted repair found for VascularMD model edge {edge}."
                )

        except Exception as exc:
            if raise_on_failure:
                raise
            edge_results.append(
                EdgeRepairResult(
                    edge=edge,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if verbose:
                print(f"edge {edge}: ERROR {type(exc).__name__}: {exc}")

    if changed_any:
        _invalidate_vascularmd_mesh_cache(target, G)

    return NetworkRepairReport(
        tree=target,
        edge_results=edge_results,
        inplace=inplace,
        skipped_bifurcation_edges=skipped_bif,
        skipped_nonradius_edges=skipped_nonradius,
        mesh_cache_invalidated=changed_any,
    )
