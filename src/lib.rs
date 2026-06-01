//! # lau-spectral-zeta
//!
//! Spectral zeta function of the agent — heat trace, functional equation,
//! regularized dimension, determinant, and Riemann hypothesis analogue
//! for agent stability.
//!
//! Given the Laplacian Δ of an agent (from `lau-landauer-meter` or
//! `lau-conservation-spectral`), this crate computes:
//!
//! - **ζ_Δ(s) = Σ λ_n^{-s}** — the spectral zeta function
//! - **Θ(t) = tr(e^{-tΔ})** — the heat trace, Mellin-transformed to ζ
//! - **tr((Δ-λ)^{-1})** — the resolvent trace, analytically continued to ζ
//! - **Functional equation** — ζ_Δ(s) ↔ ζ_Δ(1-s) symmetry
//! - **ζ_Δ(0)** — regularized dimension (conformal anomaly, the *correct* tr(id))
//! - **det Δ = e^{-ζ'_Δ(0)}** — the agent's partition function
//! - **Spectral zeros** — where ζ_Δ(s) = 0, the Riemann hypothesis analogue
//! - **Agent stability** — zeros in the critical strip signal metastable states

pub mod heat_trace;
pub mod resolvent;
pub mod spectral_zeta;
pub mod functional_equation;
pub mod regularized;
pub mod determinant;
pub mod spectral_zeros;
pub mod stability;
pub mod laplacian;

pub use heat_trace::HeatTrace;
pub use resolvent::ResolventTrace;
pub use spectral_zeta::SpectralZeta;
pub use functional_equation::FunctionalEquation;
pub use regularized::RegularizedTrace;
pub use determinant::SpectralDeterminant;
pub use spectral_zeros::SpectralZeros;
pub use stability::AgentStability;
pub use laplacian::Laplacian;

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Spectral data extracted from a Laplacian matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum {
    /// Real eigenvalues sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Dimension of the Laplacian (number of states).
    pub dimension: usize,
}

impl Spectrum {
    /// Extract spectrum from a Laplacian matrix by symmetric eigendecomposition.
    pub fn from_matrix(laplacian: &DMatrix<f64>) -> Self {
        let sym = laplacian.clone().into_owned();
        let eig = sym.symmetric_eigen();
        let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().map(|&x| x).collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Spectrum {
            dimension: eigenvalues.len(),
            eigenvalues,
        }
    }

    /// Create a spectrum from raw eigenvalues.
    pub fn from_eigenvalues(eigenvalues: Vec<f64>) -> Self {
        let dim = eigenvalues.len();
        let mut eigenvalues = eigenvalues;
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Spectrum {
            dimension: dim,
            eigenvalues,
        }
    }

    /// Positive eigenvalues only (excludes zero mode).
    pub fn positive_eigenvalues(&self) -> Vec<f64> {
        self.eigenvalues.iter().copied().filter(|&λ| λ > 1e-15).collect()
    }

    /// Number of zero eigenvalues (nullity).
    pub fn nullity(&self) -> usize {
        self.eigenvalues.iter().filter(|&&λ| λ.abs() < 1e-12).count()
    }
}

/// Configuration for spectral zeta computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZetaConfig {
    /// Number of terms in heat trace expansion.
    pub heat_trace_terms: usize,
    /// Cutoff for eigenvalue truncation in direct summation.
    pub eigenvalue_cutoff: usize,
    /// Convergence tolerance for iterative methods.
    pub tolerance: f64,
    /// Maximum iterations for Newton's method in zero finding.
    pub max_iterations: usize,
    /// Regularization parameter (small ε to avoid divergences).
    pub epsilon: f64,
}

impl Default for ZetaConfig {
    fn default() -> Self {
        ZetaConfig {
            heat_trace_terms: 50,
            eigenvalue_cutoff: 1000,
            tolerance: 1e-10,
            max_iterations: 100,
            epsilon: 1e-14,
        }
    }
}
