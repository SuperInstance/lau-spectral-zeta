//! Regularized traces and dimension.
//!
//! ζ_Δ(0) = regularized dimension (conformal anomaly, the *correct* tr(id)).
//! ζ_Δ(-1) = Σ λ_n (regularized, divergent series → finite).
//! ζ_Δ(-k) = Σ λ_n^k via analytic continuation (Ramanujan summation).

use crate::{SpectralZeta, Spectrum, ZetaConfig};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Regularized trace computation engine.
#[derive(Debug, Clone)]
pub struct RegularizedTrace {
    zeta: SpectralZeta,
}

/// Result of regularized trace computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizedTraceResult {
    /// The power k (where we compute ζ(-k)).
    pub k: f64,
    /// ζ_Δ(-k) = Σ λ_n^k (regularized).
    pub value: f64,
    /// Naive (divergent) sum if applicable.
    pub naive_sum: f64,
    /// Whether regularization was needed.
    pub regularized: bool,
}

/// Regularized dimension result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizedDimensionResult {
    /// ζ_Δ(0): the regularized dimension.
    pub zeta_zero: f64,
    /// The raw dimension (number of positive eigenvalues).
    pub raw_dimension: usize,
    /// The anomaly: difference between raw and regularized dimension.
    pub anomaly: f64,
    /// ζ'_Δ(0) (used in determinant computation).
    pub zeta_prime_zero: f64,
}

impl RegularizedTrace {
    /// Create from spectral zeta.
    pub fn new(zeta: SpectralZeta) -> Self {
        RegularizedTrace { zeta }
    }

    /// Create from Laplacian matrix.
    pub fn from_matrix(laplacian: &DMatrix<f64>, config: ZetaConfig) -> Self {
        let spectrum = Spectrum::from_matrix(laplacian);
        let zeta = SpectralZeta::new(spectrum, config);
        RegularizedTrace { zeta }
    }

    /// Compute the regularized dimension: ζ_Δ(0).
    ///
    /// This is the *correct* completion of tr(id), accounting for
    /// the conformal anomaly. For an N-dimensional system:
    /// ζ(0) = N - (regularization correction).
    ///
    /// Computed via analytic continuation from the heat kernel.
    pub fn regularized_dimension(&self) -> RegularizedDimensionResult {
        let pos_eigs = self.zeta.spectrum().positive_eigenvalues();
        let raw_dim = pos_eigs.len();

        // ζ(s→0) via analytic continuation:
        // ζ(s) = Σ λ_n^{-s} = Σ exp(-s ln λ_n)
        //       = N - s Σ ln(λ_n) + O(s²)
        // So ζ(0) = N (number of positive eigenvalues)
        // And ζ'(0) = -Σ ln(λ_n)
        let zeta_zero = raw_dim as f64;

        let zeta_prime_zero: f64 = pos_eigs.iter().map(|&λ| -λ.ln()).sum();

        let anomaly = raw_dim as f64 - zeta_zero;

        RegularizedDimensionResult {
            zeta_zero,
            raw_dimension: raw_dim,
            anomaly,
            zeta_prime_zero,
        }
    }

    /// Compute regularized trace: ζ_Δ(-k) = Σ λ_n^k (regularized).
    ///
    /// For finite-dimensional systems this is just a finite sum.
    /// For the spectral-theoretic interpretation, this is the
    /// zeta-regularized version of the divergent series Σ λ_n^k.
    pub fn regularized_sum(&self, k: f64) -> RegularizedTraceResult {
        let pos_eigs = self.zeta.spectrum().positive_eigenvalues();

        let value: f64 = pos_eigs.iter().map(|&λ| λ.powf(k)).sum();
        let naive_sum = value; // For finite spectra, they agree

        RegularizedTraceResult {
            k,
            value,
            naive_sum,
            regularized: false, // Finite spectrum = no regularization needed
        }
    }

    /// Regularized trace of identity: ζ_Δ(0) = the correct tr(id).
    pub fn tr_id(&self) -> f64 {
        self.regularized_dimension().zeta_zero
    }

    /// Regularized trace of Laplacian: ζ_Δ(-1) = Σ λ_n.
    pub fn tr_laplacian(&self) -> f64 {
        self.regularized_sum(1.0).value
    }

    /// Regularized trace of Laplacian²: ζ_Δ(-2) = Σ λ_n².
    pub fn tr_laplacian_squared(&self) -> f64 {
        self.regularized_sum(2.0).value
    }

    /// Regularized trace of arbitrary power: ζ_Δ(-k) = Σ λ_n^k.
    pub fn tr_power(&self, k: f64) -> f64 {
        self.regularized_sum(k).value
    }

    /// Compute the conformal anomaly coefficient.
    /// This is a_0 - ζ(0) in the heat kernel expansion, where a_0 is
    /// the leading Seeley-DeWitt coefficient.
    pub fn conformal_anomaly(&self) -> f64 {
        self.regularized_dimension().anomaly
    }

    /// Access the spectral zeta.
    pub fn zeta(&self) -> &SpectralZeta {
        &self.zeta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laplacian::Laplacian;
    use crate::laplacian::LaplacianKind;

    fn test_reg() -> RegularizedTrace {
        let lap = Laplacian::build(LaplacianKind::Path(5));
        let spec = Spectrum::from_matrix(&lap.to_matrix());
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        RegularizedTrace::new(zeta)
    }

    #[test]
    fn test_regularized_dimension() {
        let reg = test_reg();
        let result = reg.regularized_dimension();
        assert!(result.zeta_zero >= 0.0);
        assert_eq!(result.raw_dimension, result.zeta_zero as usize);
    }

    #[test]
    fn test_regularized_dimension_complete_graph() {
        // K_4: eigenvalues 0, 4, 4, 4 → positive: 3
        let spec = Spectrum::from_eigenvalues(vec![0.0, 4.0, 4.0, 4.0]);
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        let reg = RegularizedTrace::new(zeta);
        let result = reg.regularized_dimension();
        assert!((result.zeta_zero - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_zeta_prime_zero() {
        let reg = test_reg();
        let result = reg.regularized_dimension();
        // ζ'(0) = -Σ ln(λ_n) should be negative (since λ_n > 1 for most)
        // Could be positive if eigenvalues are between 0 and 1
        assert!(result.zeta_prime_zero.is_finite());
    }

    #[test]
    fn test_tr_id() {
        let reg = test_reg();
        let tr_id = reg.tr_id();
        assert!(tr_id > 0.0);
    }

    #[test]
    fn test_tr_laplacian() {
        let reg = test_reg();
        let tr = reg.tr_laplacian();
        assert!(tr > 0.0);
    }

    #[test]
    fn test_tr_laplacian_squared() {
        let reg = test_reg();
        let tr = reg.tr_laplacian_squared();
        assert!(tr > 0.0);
    }

    #[test]
    fn test_tr_laplacian_ge_laplacian() {
        // Since λ² ≥ λ for λ ≥ 1, tr(L²) ≥ tr(L) when all λ ≥ 1
        let reg = test_reg();
        let tr_l = reg.tr_laplacian();
        let tr_l2 = reg.tr_laplacian_squared();
        assert!(tr_l2 >= tr_l - 1e-10);
    }

    #[test]
    fn test_regularized_sum_k0() {
        let reg = test_reg();
        let result = reg.regularized_sum(0.0);
        // ζ(0) should equal number of positive eigenvalues
        assert!((result.value - reg.tr_id()).abs() < 1e-10);
    }

    #[test]
    fn test_conformal_anomaly() {
        let reg = test_reg();
        let anomaly = reg.conformal_anomaly();
        // For finite spectrum, anomaly should be 0
        assert!(anomaly.abs() < 1e-10);
    }

    #[test]
    fn test_from_matrix() {
        let lap = Laplacian::build(LaplacianKind::Complete(4));
        let reg = RegularizedTrace::from_matrix(&lap.to_matrix(), ZetaConfig::default());
        let result = reg.regularized_dimension();
        assert!(result.zeta_zero > 0.0);
    }

    #[test]
    fn test_regularized_dimension_serializable() {
        let reg = test_reg();
        let result = reg.regularized_dimension();
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("zeta_zero"));
    }

    #[test]
    fn test_tr_power_arbitrary() {
        let reg = test_reg();
        let tr = reg.tr_power(0.5);
        assert!(tr > 0.0);
    }
}
