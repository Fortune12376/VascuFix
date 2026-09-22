# VascuFix

### Geometry-Preserving Curvature Repair for Cerebral Arterial Models

VascuFix is a lightweight post-processing extension for [VascularMD](https://github.com/megdec/vascularmd). It detects non-bifurcating vessel segments whose circular tubular reconstruction violates the local curvature admissibility condition and locally repairs those splines before meshing.

The intended workflow is:

```text
VascularMD centerline model
        ↓
      VascuFix
        ↓
repaired VascularMD model
        ↓
VascularMD surface / volume meshing
```

VascuFix does **not** replace VascularMD and does not implement a separate mesher. It modifies only the fitted vessel splines that require repair and then returns control to the standard VascularMD workflow.

---

## Motivation

For a centerline $c(t)$ with circular vessel radius $r(t)$, a local tubular representation becomes singular when the vessel radius reaches the local radius of curvature. In terms of curvature $\kappa(t)$,

$$
r(t)\kappa(t) < 1.
$$

VascuFix enforces the configurable safety margin

$$
r(t)\kappa(t) \le 1-\varepsilon,
$$

with the default

$$
\varepsilon = 0.05,
\qquad
r(t)\kappa(t)\le 0.95.
$$

The objective is not to globally smooth or straighten the vessel. Instead, VascuFix searches for the smallest local modification of the VascularMD reconstruction that satisfies the curvature constraint while preserving the original vessel geometry and radius profile as closely as possible.

---

## Features

- Continuous screening of fitted VascularMD vessel splines using $r\kappa$.
- Repairs only violating **non-bifurcating vessel segments**.
- Leaves already admissible vessels unchanged.
- Local constrained optimization rather than global smoothing.
- Simultaneous centerline and radius repair.
- Preserves vessel endpoint positions and endpoint radii.
- Preserves endpoint tangent directions through a $G^1$ parameterization.
- Tests reduced, original, and enriched local spline spaces.
- Uses a repair-information criterion to balance geometric preservation against additional model complexity.
- Performs continuous post-optimization peak searches so feasibility is not based only on discrete optimizer sample points.
- Writes successful repairs back into the VascularMD `model_graph`.
- Invalidates old VascularMD cross-section / mesh caches so the repaired network can be meshed normally.
- Leaves VascularMD bifurcation geometry untouched.

---

## Installation

VascuFix requires an existing working installation of **VascularMD**.

Clone this repository and install its Python dependencies:

```bash
git clone https://github.com/<YOUR_USERNAME>/VascuFix.git
cd VascuFix
pip install -r requirements.txt
```

The package can also be installed directly from the repository root with:

```bash
pip install .
```

VascularMD itself must already be importable in the Python environment.

---

## Quick start

First build the arterial model normally with VascularMD:

```python
from ArterialTree import ArterialTree

tree = ArterialTree(...)

tree.model_network(...)
```

Then run VascuFix:

```python
from vascufix import (
    RepairConfig,
    repair_vascularmd_tree,
)

config = RepairConfig()

report = repair_vascularmd_tree(
    tree,
    config=config,
    inplace=True,
)
```

The repaired splines are now stored directly in the VascularMD model graph.

Continue with the normal VascularMD meshing workflow:

```python
tree.compute_cross_sections(
    N=32,
    d=0.35,
    parallel=False,
)

tree.mesh_volume()

mesh = tree.get_volume_mesh()
```

The resulting network contains the original VascularMD bifurcations and unchanged vessel segments together with the locally repaired vessel splines.

---

## Non-destructive use

By default, VascuFix does not modify the supplied tree:

```python
report = repair_vascularmd_tree(tree)

repaired_tree = report.tree
```

`repaired_tree` is a deep copy of the original VascularMD tree.

To modify the existing VascularMD model directly:

```python
report = repair_vascularmd_tree(
    tree,
    inplace=True,
)
```

---

## Inspecting the repair

The network repair call returns a report:

```python
print(report.n_examined)
print(report.n_violating)
print(report.n_repaired)
print(report.n_failed)
```

Detailed per-edge information is available through:

```python
for edge_result in report.edge_results:
    print(edge_result.summary())
```

The report records whether each vessel changed, its maximum $r\kappa$ before and after repair, the selected change in spline complexity, the repair objective, and the repair-information score.

---

## Repairing a single VascularMD spline

The lower-level API can be used independently of an `ArterialTree`:

```python
from vascufix import repair_spline

result = repair_spline(spline)

repaired_spline = result.spline

print(result.before_max_rk)
print(result.after_max_rk)
print(result.net_knot_delta)
```

The input spline is not modified by `repair_spline`.

To inspect a spline without repairing it:

```python
from vascufix import max_radius_curvature

q_max, t_max = max_radius_curvature(spline)

print(q_max, t_max)
```

---

## Method

### 1. Curvature admissibility

For the fitted vessel centerline $c(t)$,

```math
\kappa(t)
=
\frac{\|c'(t)\times c''(t)\|}
     {\|c'(t)\|^3}.
```

VascuFix defines

$$
q(t)=r(t)\kappa(t)
$$

and uses the default physical target

$$
q(t)\le0.95.
$$

The optimizer is constrained to the strict value `0.95`. A small numerical tolerance is used only when deciding whether the final continuously checked spline is acceptable:

$$
q_{\max}\le0.9502.
$$

### 2. Local repair objective

Let $c_0(t)$ and $r_0(t)$ denote the original VascularMD reconstruction. For a repair interval $I$, VascuFix minimizes

```math
J
=
\frac{1}{L_I}
\int_I
\left[
\|c(t)-c_0(t)\|^2
+
\frac{(r(t)-r_0(t))^2}
     {\rho^2}
\right]
\,ds_0,
```

where the default relative radius/spatial uncertainty is

$$
\rho = 0.594.
$$

Thus the repair is explicitly driven toward the original VascularMD vessel rather than toward a globally smoothed trajectory.

### 3. Endpoint continuity

Endpoint positions and endpoint radii are preserved.

For endpoint tangent directions, the adjacent control points are parameterized as

$$
P_1=P_0+\alpha_0 s_0\hat T_0,
$$

$$
P_{n-2}=P_{n-1}-\beta_0 s_1\hat T_1,
$$

with positive scalar variables $s_0,s_1$.

This preserves the original endpoint tangent directions while allowing their magnitudes to adapt during local repair.

### 4. Candidate spline spaces

For a violating region, VascuFix tests several nearby spline complexities around the original VascularMD spline. By default this permits up to two knot removals and five additional knots.

For every candidate spline space, VascuFix solves the constrained minimum-$J$ problem.

Candidates that fail the continuous physical admissibility test are discarded.

### 5. Repair-information criterion

Among physically admissible candidates, VascuFix balances preservation of the original vessel against the number of free repair variables using

```math
\mathrm{RIC}_k
=
m_I
\log\!\left(
\max(J_k^*,J_\varepsilon)
\right)
+
2p_k,
```

where

- $J_k^*$ is the optimized repair objective for candidate $k$,
- $m_I$ is the number of original VascularMD observation locations associated with the fixed repair-support interval,
- $p_k=\dim(\theta_k)$ is the number of free scalar optimization variables,
- $J_\varepsilon$ is only a numerical floor preventing $\log(0)$.

The feasible candidate with the smallest RIC is selected.

### 6. Continuous feasibility checking

The optimizer initially enforces the curvature constraint at a finite set of locations.

After each solve, VascuFix searches every non-zero knot span for continuous local maxima of $r\kappa$. If a missed violating peak is found, that location is added to the constraint set and the optimization is repeated.

Therefore optimizer success alone is not treated as proof of physical admissibility.

---

## Configuration

The default configuration reproduces the settings used during development:

```python
from vascufix import RepairConfig

config = RepairConfig(
    epsilon=0.05,
    sigma_radius_over_spatial=0.594,
    max_removed_knots=2,
    max_extra_knots=5,
    feasibility_tolerance=2e-4,
)
```

Important parameters include:

| Parameter | Default | Meaning |
|---|---:|---|
| `epsilon` | `0.05` | Safety margin in $r\kappa \le 1-\varepsilon$ |
| `sigma_radius_over_spatial` | `0.594` | Relative weighting of radius and centerline changes in $J$ |
| `max_removed_knots` | `2` | Maximum local reduction in spline complexity |
| `max_extra_knots` | `5` | Maximum local increase in spline complexity |
| `repair_padding_spans` | `1` | Knot-span padding around the violating region |
| `feasibility_tolerance` | `2e-4` | Numerical acceptance tolerance after continuous checking |
| `min_radius` | `1e-5` | Positive-radius safeguard |
| `min_speed` | `1e-7` | Non-degenerate centerline-speed safeguard |

Most users should not need to modify the lower-level numerical optimization and sampling parameters.

---

## Integration with VascularMD

VascuFix operates on the fitted VascularMD `model_graph`.

Ordinary vessel edges store their fitted spline as:

```python
tree.get_model_graph().edges[edge]["spline"]
```

When a repair succeeds, VascuFix replaces that spline with the repaired version.

Edges associated directly with VascularMD bifurcation geometry are skipped.

After any successful repair, downstream cross-section and mesh data are invalidated. The user can then call the normal VascularMD functions:

```python
tree.compute_cross_sections(...)
tree.mesh_surface()
tree.mesh_volume()
```

VascuFix does not propagate repaired geometry back into the original SWC/full graph.

---

## Scope and limitations

VascuFix currently addresses **local curvature-induced tubular inadmissibility** of non-bifurcating vessels.

It does not currently:

- modify VascularMD bifurcation models,
- resolve collisions between geometrically distant portions of the same vessel,
- resolve collisions between different vessels,
- guarantee that every possible cause of a poor or inverted mesh element is removed,
- replace VascularMD's surface or hexahedral volume meshing methods.

The enforced condition $r\kappa<1$ addresses the local singularity of a circular normal-tube construction. Other geometric and meshing failure modes may require separate treatment.

The current implementation assumes circular vessel cross-sections because the VascularMD/SWC representation provides a scalar radius rather than cross-sectional eccentricity or orientation.

---

## Relationship to VascularMD

VascuFix is intended as a post-processing extension to VascularMD rather than a fork or replacement.

VascularMD provides the centerline-based vascular reconstruction, bifurcation model, and structured surface/volume meshing framework. VascuFix operates between the modeling and meshing stages to repair local curvature violations in ordinary vessel segments.

Users should cite VascularMD when using its reconstruction or meshing framework.

Decroocq, M., Frindel, C., Rougé, P., Ohta, M., & Lavoué, G. (2023). Modeling and hexahedral meshing of cerebral arterial networks from centerlines. Medical Image Analysis, 89, 102912.
---

## Citation

If you use VascuFix in academic work, please cite the repository.

VascuFix relies on the VascularMD framework. Please also cite:

> Decroocq, M., Frindel, C., Rougé, P., Ohta, M., & Lavoué, G. (2023).
> *Modeling and hexahedral meshing of cerebral arterial networks from centerlines.*
> Medical Image Analysis, 89, 102912.

---

## License

VascuFix is distributed under the license included in the repository's `LICENSE` file.

VascularMD is a separate project with its own license and citation requirements. Users should review and comply with the applicable VascularMD terms when using the two projects together.

---

## Status

VascuFix is currently an early research release focused on cerebral arterial networks reconstructed with VascularMD.

The current implementation targets non-bifurcating vessel segments and is intended for reproducible research and further validation.
