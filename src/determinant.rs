//! Spectral determinant: det Δ = e^{-ζ'_Δ(0)}
//!
//! The spectral determinant is the agent's partition function.
//! It encodes the full spectral information into a single number.

use crate::{RegularizedTrace, SpectralZeta, Spectrum, ZetaConfig};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Spectral determinant computation engine.
#[derive(Debug, Clone)]
pub struct SpectralDeterminant {
    zeta: SpectralZeta,
}

/// Result of spectral determinant computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterminantResult {
    /// ζ'_Δ(0) value.
    pub zeta_prime_zero: f64,
    /// det Δ = e^{-ζ'_Δ(0)}.
    pub determinant: f64,
    /// log(det Δ) = -ζ'_Δ(0).
    pub log_determinant: f64,
    /// Number of zero modes (these are excluded).
    pub zero_modes: usize,
}

/// Comparison of determinant with another operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterminantRatio {
    /// log(det Δ₁ / det Δ₂) = -ζ'_₁(0) + ζ'_₂(0).
    pub log_ratio: f64,
    /// det Δ₁ / det Δ₂.
    pub ratio: f64,
}

impl SpectralDeterminant {
    /// Create from spectral zeta.
    pub fn new(zeta: SpectralZeta) -> Self {
        SpectralDeterminant { zeta }
    }

    /// Create from Laplacian matrix.
    pub fn from_matrix(laplacian: &DMatrix<f64>, config: ZetaConfig) -> Self {
        let spectrum = Spectrum::from_matrix(laplacian);
        let zeta = SpectralZeta::new(spectrum, config);
        SpectralDeterminant { zeta }
    }

    /// Compute the spectral determinant: det Δ = e^{-ζ'_Δ(0)}.
    ///
    /// The derivative ζ'(0) = -Σ_{λ_n > 0} ln(λ_n).
    /// So det Δ = Π_{λ_n > 0} λ_n (product of positive eigenvalues).
    pub fn compute(&self) -> DeterminantResult {
        let reg = RegularizedTrace::new(self.zeta.clone());
        let dim_result = reg.regularized_dimension();
        let zero_modes = self.zeta.spectrum().nullity();

        DeterminantResult {
            zeta_prime_zero: dim_result.zeta_prime_zero,
            determinant: (-dim_result.zeta_prime_zero).exp(),
            log_determinant: -dim_result.zeta_prime_zero,
            zero_modes,
        }
    }

    /// Compute det Δ as the product of positive eigenvalues.
    pub fn product_determinant(&self) -> f64 {
        let pos_eigs = self.zeta.spectrum().positive_eigenvalues();
        pos_eigs.iter().product()
    }

    /// Compute the ratio of two determinants.
    pub fn ratio(&self, other: &SpectralDeterminant) -> DeterminantRatio {
        let det1 = self.compute();
        let det2 = other.compute();
        let log_ratio = det1.log_determinant - det2.log_determinant;
        DeterminantRatio {
            log_ratio,
            ratio: log_ratio.exp(),
        }
    }

    /// Compute the effective free energy: F = -log(det Δ) = ζ'_Δ(0).
    pub fn free_energy(&self) -> f64 {
        self.compute().zeta_prime_zero
    }

    /// Compute the partition function Z = det Δ = e^{-ζ'_Δ(0)}.
    pub fn partition_function(&self) -> f64 {
        self.compute().determinant
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

    fn test_det() -> SpectralDeterminant {
        let lap = Laplacian::build(LaplacianKind::Path(5));
        let spec = Spectrum::from_matrix(&lap.to_matrix());
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        SpectralDeterminant::new(zeta)
    }

    #[test]
    fn test_compute_determinant() {
        let det = test_det();
        let result = det.compute();
        assert!(result.determinant > 0.0);
        assert!(result.log_determinant.is_finite());
    }

    #[test]
    fn test_determinant_equals_product() {
        let det = test_det();
        let det_result = det.compute().determinant;
        let prod = det.product_determinant();
        assert!((det_result - prod).abs() / prod.abs() < 1e-8);
    }

    #[test]
    fn test_log_det_equals_negative_zeta_prime() {
        let det = test_det();
        let result = det.compute();
        assert!((result.log_determinant + result.zeta_prime_zero).abs() < 1e-10);
    }

    #[test]
    fn test_free_energy() {
        let det = test_det();
        let fe = det.free_energy();
        assert!(fe.is_finite());
    }

    #[test]
    fn test_partition_function() {
        let det = test_det();
        let z = det.partition_function();
        assert!(z > 0.0);
        assert!((z - det.compute().determinant).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_ratio() {
        let lap1 = Laplacian::build(LaplacianKind::Path(4));
        let lap2 = Laplacian::build(LaplacianKind::Cycle(4));
        let det1 = SpectralDeterminant::from_matrix(&lap1.to_matrix(), ZetaConfig::default());
        let det2 = SpectralDeterminant::from_matrix(&lap2.to_matrix(), ZetaConfig::default());
        let ratio = det1.ratio(&det2);
        assert!(ratio.ratio.is_finite());
        assert!(ratio.ratio > 0.0);
    }

    #[test]
    fn test_complete_graph_determinant() {
        // K_4: eigenvalues 0, 4, 4, 4 → det = 4*4*4 = 64
        let spec = Spectrum::from_eigenvalues(vec![0.0, 4.0, 4.0, 4.0]);
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        let det = SpectralDeterminant::new(zeta);
        assert!((det.product_determinant() - 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_modes_counted() {
        let spec = Spectrum::from_eigenvalues(vec![0.0, 1.0, 2.0]);
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        let det = SpectralDeterminant::new(zeta);
        let result = det.compute();
        assert_eq!(result.zero_modes, 1);
    }

    #[test]
    fn test_from_matrix() {
        let lap = Laplacian::build(LaplacianKind::Cycle(6));
        let det = SpectralDeterminant::from_matrix(&lap.to_matrix(), ZetaConfig::default());
        let result = det.compute();
        assert!(result.determinant > 0.0);
    }

    #[test]
    fn test_determinant_serializable() {
        let det = test_det();
        let result = det.compute();
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("determinant"));
    }

    #[test]
    fn test_larger_graph_determinant() {
        let lap = Laplacian::build(LaplacianKind::Complete(10));
        let det = SpectralDeterminant::from_matrix(&lap.to_matrix(), ZetaConfig::default());
        let result = det.compute();
        assert!(result.determinant > 0.0);
        assert!(result.log_determinant.is_finite());
    }
}
