# lau-spectral-zeta

> Spectral zeta function of the agent — heat trace, functional equation, regularized dimension, determinant, and Riemann hypothesis analogue for agent stability.

## What This Does

Given the Laplacian Δ of an agent (from its state-transition topology), this crate computes the **spectral zeta function** ζ_Δ(s) = Σ λ_n^{-s} and everything that flows from it:

- **Heat trace** Θ(t) = tr(e^{-tΔ}) — the trace of the heat kernel
- **Resolvent trace** tr((Δ−λ)^{-1}) — meromorphic continuation to zeta
- **Functional equation** ζ_Δ(s) ↔ ζ_Δ(d−s) — symmetry of the completed zeta
- **Regularized dimension** ζ_Δ(0) — the *correct* tr(id), accounting for conformal anomaly
- **Spectral determinant** det Δ = e^{−ζ'_Δ(0)} — the agent's partition function
- **Spectral zeros** — where ζ_Δ(s) = 0, and a **Riemann hypothesis analogue**
- **Agent stability** — zeros in the critical strip signal metastable states

This is spectral geometry applied to agent state spaces. If all non-trivial zeros of ζ_Δ lie on the critical line Re(s) = d/2, the agent is maximally stable. Off-line zeros mean metastability.

## Key Idea

On a Riemannian manifold, the spectrum of the Laplacian encodes the geometry. The spectral zeta function packages this into a single meromorphic function whose special values carry deep information:

- ζ_Δ(0) = regularized dimension (not just counting states — accounting for regularization)
- ζ'_Δ(0) = log(det Δ) = the free energy of the system
- Zeros of ζ_Δ(s) = resonance spectrum of the agent

The Riemann hypothesis becomes an engineering question: **are the agent's spectral resonances well-organized?** If yes, the agent is thermodynamically stable. If not, there are metastable states that could cause unexpected transitions.

## Install

```toml
[dependencies]
lau-spectral-zeta = "0.1.0"
```

Or:

```sh
cargo add lau-spectral-zeta
```

Dependencies: `nalgebra` 0.33, `serde` + `serde_json`, `num-complex` 0.4, `num-traits` 0.2.

## Quick Start

```rust
use lau_spectral_zeta::*;

// Build a Laplacian for a graph topology
let lap = Laplacian::build(LaplacianKind::Cycle(6));
let spectrum = Spectrum::from_matrix(&lap.to_matrix());

// Spectral zeta function
let zeta = SpectralZeta::new(spectrum.clone(), ZetaConfig::default());
println!("ζ(1) = {:.4}", zeta.evaluate(1.0).value);
println!("ζ(2) = {:.4}", zeta.evaluate(2.0).value);
println!("ζ'(0) = {:.4}", zeta.derivative(0.0));

// Heat trace
let heat = HeatTrace::new(spectrum.clone(), ZetaConfig::default());
let ht = heat.evaluate(1.0);
println!("Θ(1.0) = {:.4} ({} terms, error < {:.6})", ht.theta, ht.terms_used, ht.truncation_error);

// Resolvent trace
let resolvent = ResolventTrace::new(spectrum.clone(), ZetaConfig::default());
use num_complex::Complex64;
let r = resolvent.evaluate(Complex64::new(-1.0, 0.0));
println!("tr(R(-1)) = {:.4}", r.trace);

// Spectral determinant = partition function
let det = SpectralDeterminant::new(zeta.clone());
let d = det.compute();
println!("det Δ = {:.4} ({} zero modes)", d.determinant, d.zero_modes);
println!("Free energy = {:.4}", det.free_energy());

// Regularized dimension
let reg = RegularizedTrace::new(zeta.clone());
let dim = reg.regularized_dimension();
println!("ζ(0) = {} (raw: {})", dim.zeta_zero, dim.raw_dimension);

// Stability analysis (Riemann hypothesis for agents)
let stability = AgentStability::new(zeta.clone(), 2.0);
let report = stability.analyze();
println!("Stability: {:.2} ({:?})", report.stability_score, report.classification);
println!("Riemann hypothesis holds: {}", report.riemann_hypothesis_holds);
println!("Metastable states: {}", report.metastable_count);

// Functional equation
let fe = FunctionalEquation::new(zeta, 2.0);
let xi = fe.completed_zeta(1.0);
println!("Ξ(1) = {:.4}, Ξ(d-1) = {:.4}", xi.xi, xi.xi_dual);
```

## API Reference

### Laplacian Construction (`laplacian`)

| `LaplacianKind` | Graph | Description |
|----------------|-------|-------------|
| `Path(n)` | P_n | 1D path graph with Neumann-like boundaries |
| `Cycle(n)` | C_n | Cycle graph (ring) |
| `Complete(n)` | K_n | Complete graph |
| `Star(n)` | S_n | Star graph (one hub) |
| `Grid { rows, cols }` | Grid | 2D lattice with 4-connectivity |
| `Discrete1D(n)` | — | 1D second-difference operator |

Also supports building from adjacency matrices: `Laplacian::from_adjacency()` and `Laplacian::from_adjacency_normalized()`.

### `SpectralZeta` — Core Zeta Function

- `evaluate(s: f64)` → ζ_Δ(s) for real s
- `evaluate_complex(s: Complex64)` → ζ_Δ(s) for complex s
- `evaluate_range(&[s])` → batch evaluation
- `evaluate_negative(s)` → ζ_Δ(s) for s < 0 (finite spectrum → just sum powers)
- `derivative(s)` → ζ'_Δ(s) via central differences
- `second_derivative(s)` → ζ''_Δ(s)
- `partial_sums(s, n)` → convergence analysis
- `is_zero(s, tol)` → zero test

### `HeatTrace` — Heat Kernel Trace

- `evaluate(t)` → Θ(t) = Σ e^{−tλ_n}
- `evaluate_range(&[t])` → batch
- `short_time_expansion(order)` → Weyl asymptotics coefficients a_k
- `zeta_from_heat_kernel(s, t_max, n_quad)` → ζ via Mellin transform quadrature
- `mellin_transform(s: Complex64)` → direct complex zeta

### `ResolventTrace` — Resolvent Trace

- `evaluate(z: Complex64)` → tr((Δ−z)^{-1})
- `zeta_from_resolvent(s)` → ζ via resolvent summation
- `zeta_complex(s)` → complex zeta from resolvent
- `zeta_derivative(s)` → numerical ζ'

### `FunctionalEquation` — Symmetry Analysis

- `verify(s, tol)` → check ζ(s) ↔ ζ(d−s) relation
- `verify_range(&[s])` → batch verification
- `completed_zeta(s)` → Ξ(s) = π^{−s/2} Γ(s/2) ζ(s)
- `check_symmetry(s)` → test Ξ(s) = Ξ(d−s)

### `RegularizedTrace` — ζ-Regularized Quantities

- `regularized_dimension()` → ζ(0), anomaly, ζ'(0)
- `tr_id()` → ζ(0) = regularized tr(identity)
- `tr_laplacian()` → ζ(−1) = Σ λ_n
- `tr_laplacian_squared()` → ζ(−2) = Σ λ_n²
- `tr_power(k)` → ζ(−k) = Σ λ_n^k
- `conformal_anomaly()` → Seeley-DeWitt a_0 − ζ(0)

### `SpectralDeterminant` — Partition Function

- `compute()` → det Δ = e^{−ζ'(0)}, log det, zero modes
- `product_determinant()` → Π λ_n (direct product)
- `free_energy()` → F = ζ'(0)
- `partition_function()` → Z = det Δ
- `ratio(&other)` → det Δ₁ / det Δ₂

### `SpectralZeros` — Zero Finding

- `find_zeros(s_range, imag_range, resolution, tol)` → grid search + Newton refinement
- `classify_zeros(zeros, dim)` → critical strip / critical line / exterior
- `count_trivial_zeros()` → zeros from Γ factors

### `AgentStability` — Stability Analysis

- `analyze()` → full `StabilityReport` with score, classification, Riemann hypothesis check
- `quick_score()` → fast proxy using spectral gap and condition number
- `compare(&other)` → head-to-head stability comparison
- `perturbation_stability(δ)` → stability under eigenvalue perturbation

Stability classes: `Stable` (all zeros on critical line), `Metastable` (some off-line), `Unstable` (many exterior), `Indeterminate` (no zeros found).

## How It Works

### Spectral Zeta Pipeline

1. **Construct Laplacian**: Build a graph Laplacian from the agent's state transition topology.
2. **Eigendecomposition**: Compute eigenvalues λ₁ ≤ λ₂ ≤ … ≤ λ_n using symmetric eigendecomposition.
3. **Zeta evaluation**: Sum ζ_Δ(s) = Σ_{λ>0} λ^{−s} (finite spectrum, always converges for Re(s) > 0).
4. **Analytic continuation**: For s ≤ 0, finite sums mean ζ(−k) = Σ λ_n^k — no divergence to worry about.
5. **Special values**: ζ(0) = count of positive eigenvalues, ζ'(0) = −Σ ln(λ_n), det Δ = Π λ_n.

### Zero Finding

1. **Grid search**: Evaluate |ζ(s)| on a grid in the complex plane.
2. **Seed detection**: Find grid cells where |ζ(s)| < 1.0.
3. **Newton refinement**: s_{n+1} = s_n − ζ(s_n) / ζ'(s_n), iterating to convergence.
4. **Classification**: Separate zeros into critical strip (0 < Re(s) < d), critical line (Re(s) = d/2), and exterior.

### Stability Scoring

- **Full analysis**: Find zeros, classify them. Score = (zeros on critical line) / (total interior zeros). Riemann hypothesis holds if score = 1.0.
- **Quick score**: Proxy using spectral gap (λ_min) and condition number (λ_max/λ_min): `0.5 × gap_score + 0.5 × condition_score`.
- **Perturbation stability**: Whether λ_min − δ > 0 for perturbation strength δ.

## The Math

### Spectral Zeta Function

For a Laplacian Δ with positive eigenvalues {λ_n}:

```
ζ_Δ(s) = Σ_{n=1}^{N} λ_n^{-s}
```

This is the agent-world analogue of the Riemann zeta function ζ(s) = Σ n^{−s}, but with the Laplacian eigenvalues replacing the integers.

### Heat Trace and Mellin Transform

The heat trace Θ(t) = tr(e^{−tΔ}) = Σ e^{−tλ_n} is related to ζ via:

```
ζ_Δ(s) = (1/Γ(s)) ∫₀^∞ t^{s−1} Θ(t) dt
```

This Mellin transform converts the heat kernel (a function of time) into the zeta function (a function of the spectral parameter).

### Resolvent Trace

The resolvent R(z) = (Δ − z)^{-1} has trace:

```
tr(R(z)) = Σ 1/(λ_n − z)
```

This is meromorphic with poles at eigenvalues. The zeta function is obtained by contour integration of the resolvent.

### Functional Equation

For operators on d-dimensional spaces, the completed zeta:

```
Ξ(s) = π^{−s/2} Γ(s/2) ζ_Δ(s)
```

satisfies the functional equation Ξ(s) = Ξ(d − s), generalizing the Riemann ξ-function's symmetry ξ(s) = ξ(1 − s).

### Spectral Determinant

```
det Δ = exp(−ζ'_Δ(0)) = Π_{λ_n > 0} λ_n
```

This is the zeta-regularized determinant, equal to the product of positive eigenvalues. Its logarithm is the free energy of the system.

### Regularized Dimension

```
ζ_Δ(0) = N_pos  (number of positive eigenvalues)
ζ'_Δ(0) = −Σ ln(λ_n)
```

For continuous operators, ζ(0) differs from the naive dimension by the conformal anomaly. For finite matrices, they agree (anomaly = 0).

### Riemann Hypothesis Analogue

The critical strip is 0 < Re(s) < d. The critical line is Re(s) = d/2.

- **All non-trivial zeros on critical line** → agent is maximally stable
- **Zeros off critical line but in strip** → metastable states exist
- **Zeros outside strip** → instability

This mirrors the classical Riemann hypothesis: "all non-trivial zeros of ζ(s) have Re(s) = 1/2."

## Tests

96 unit tests. Run with:

```sh
cargo test
```

Test categories:
- Laplacian construction (path, cycle, complete, star, grid, adjacency, normalized)
- Heat trace (evaluation, monotonicity, short-time expansion, Mellin transform, gamma function)
- Resolvent trace (pole detection, zeta from resolvent, derivative)
- Spectral zeta (evaluation, monotonicity, complex, derivatives, known spectra)
- Functional equation (verification, completed zeta, symmetry)
- Regularized traces (dimension, tr(id), tr(Δ), tr(Δ²), conformal anomaly)
- Determinant (compute, product, ratio, free energy, zero modes)
- Spectral zeros (finding, Newton refinement, classification)
- Stability (analysis, scoring, comparison, perturbation)

## License

MIT
