//! Spectral zeta function: ζ_Δ(s) = Σ_n λ_n^{-s}
//!
//! The central object. From it we derive regularized dimension,
//! determinant, functional equation, and spectral zeros.

use crate::{Spectrum, ZetaConfig};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Spectral zeta function engine.
#[derive(Debug, Clone)]
pub struct SpectralZeta {
    spectrum: Spectrum,
    config: ZetaConfig,
}

/// Evaluation result for the spectral zeta function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZetaResult {
    /// The point s where ζ was evaluated.
    pub s: f64,
    /// ζ_Δ(s) value.
    pub value: f64,
    /// Number of eigenvalue terms used.
    pub terms_used: usize,
}

/// Evaluation result for complex s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZetaComplexResult {
    /// The complex point s.
    pub s: Complex64,
    /// ζ_Δ(s) value.
    pub value: Complex64,
}

impl SpectralZeta {
    /// Create from spectrum.
    pub fn new(spectrum: Spectrum, config: ZetaConfig) -> Self {
        SpectralZeta { spectrum, config }
    }

    /// Create from Laplacian matrix.
    pub fn from_matrix(laplacian: &DMatrix<f64>, config: ZetaConfig) -> Self {
        let spectrum = Spectrum::from_matrix(laplacian);
        SpectralZeta { spectrum, config }
    }

    /// Evaluate ζ_Δ(s) = Σ_n λ_n^{-s} for real s.
    pub fn evaluate(&self, s: f64) -> ZetaResult {
        let pos_eigs = self.spectrum.positive_eigenvalues();
        let n_terms = pos_eigs.len().min(self.config.eigenvalue_cutoff);

        let mut value = 0.0;
        for &lambda in pos_eigs.iter().take(n_terms) {
            value += lambda.powf(-s);
        }

        ZetaResult {
            s,
            value,
            terms_used: n_terms,
        }
    }

    /// Evaluate ζ_Δ(s) for complex s.
    pub fn evaluate_complex(&self, s: Complex64) -> ZetaComplexResult {
        let pos_eigs = self.spectrum.positive_eigenvalues();
        let n_terms = pos_eigs.len().min(self.config.eigenvalue_cutoff);

        let mut value = Complex64::new(0.0, 0.0);
        for &lambda in pos_eigs.iter().take(n_terms) {
            let log_l = lambda.ln();
            let exponent = Complex64::new(-s.re * log_l, -s.im * log_l);
            value += exponent.exp();
        }

        ZetaComplexResult { s, value }
    }

    /// Evaluate ζ_Δ(s) over a range of real s values.
    pub fn evaluate_range(&self, s_values: &[f64]) -> Vec<ZetaResult> {
        s_values.iter().map(|&s| self.evaluate(s)).collect()
    }

    /// Numerical derivative ζ'_Δ(s) via central differences.
    pub fn derivative(&self, s: f64) -> f64 {
        let eps = 1e-7;
        let plus = self.evaluate(s + eps).value;
        let minus = self.evaluate(s - eps).value;
        (plus - minus) / (2.0 * eps)
    }

    /// Second derivative ζ''_Δ(s).
    pub fn second_derivative(&self, s: f64) -> f64 {
        let eps = 1e-5;
        let mid = self.evaluate(s).value;
        let plus = self.evaluate(s + eps).value;
        let minus = self.evaluate(s - eps).value;
        (plus - 2.0 * mid + minus) / (eps * eps)
    }

    /// Evaluate using zeta regularization for s < 0 (divergent series → finite).
    /// Uses Ramanujan summation: Σ λ_n^k = ζ_Δ(-k) analytically continued.
    pub fn evaluate_negative(&self, s: f64) -> f64 {
        // For negative s, direct summation diverges.
        // Use analytic continuation via the heat kernel:
        // ζ(s) = (1/Γ(s)) ∫₀^∞ t^{s-1} Θ(t) dt
        // For s < 0, we can compute from the short-time expansion of Θ(t).

        let pos_eigs = self.spectrum.positive_eigenvalues();
        let _n = pos_eigs.len();

        // For finite matrices, we can just sum directly (it's finite!)
        // ζ_Δ(-k) = Σ λ_n^k for finite spectrum
        let k = -s;
        let mut value = 0.0;
        for &lambda in &pos_eigs {
            value += lambda.powf(k);
        }
        value
    }

    /// Compute the partial sums for convergence analysis.
    pub fn partial_sums(&self, s: f64, n_terms: usize) -> Vec<f64> {
        let pos_eigs = self.spectrum.positive_eigenvalues();
        let mut sums = Vec::new();
        let mut cumulative = 0.0;
        for (_i, &lambda) in pos_eigs.iter().take(n_terms).enumerate() {
            cumulative += lambda.powf(-s);
            sums.push(cumulative);
        }
        sums
    }

    /// Check if ζ_Δ(s) = 0 at the given point (within tolerance).
    pub fn is_zero(&self, s: f64, tolerance: Option<f64>) -> bool {
        let tol = tolerance.unwrap_or(self.config.tolerance);
        self.evaluate(s).value.abs() < tol
    }

    /// Access the spectrum.
    pub fn spectrum(&self) -> &Spectrum {
        &self.spectrum
    }

    /// Access the config.
    pub fn config(&self) -> &ZetaConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laplacian::Laplacian;
    use crate::laplacian::LaplacianKind;

    fn test_zeta() -> SpectralZeta {
        let lap = Laplacian::build(LaplacianKind::Path(5));
        let spec = Spectrum::from_matrix(&lap.to_matrix());
        SpectralZeta::new(spec, ZetaConfig::default())
    }

    #[test]
    fn test_evaluate_positive_s() {
        let zeta = test_zeta();
        let result = zeta.evaluate(1.0);
        assert!(result.value > 0.0);
        assert!(result.terms_used > 0);
    }

    #[test]
    fn test_evaluate_s2() {
        let zeta = test_zeta();
        let result = zeta.evaluate(2.0);
        assert!(result.value > 0.0);
    }

    #[test]
    fn test_zeta_decreases_with_s() {
        // Use eigenvalues all > 1 to guarantee monotonicity
        let spec = Spectrum::from_eigenvalues(vec![0.0, 2.0, 3.0, 5.0]);
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        let z1 = zeta.evaluate(1.0).value;
        let z2 = zeta.evaluate(2.0).value;
        let z3 = zeta.evaluate(3.0).value;
        assert!(z1 > z2);
        assert!(z2 > z3);
    }

    #[test]
    fn test_evaluate_complex() {
        let zeta = test_zeta();
        let result = zeta.evaluate_complex(Complex64::new(2.0, 1.0));
        assert!(result.value.norm() > 0.0);
    }

    #[test]
    fn test_complex_real_part_matches() {
        let zeta = test_zeta();
        let real_val = zeta.evaluate(2.0).value;
        let complex_val = zeta.evaluate_complex(Complex64::new(2.0, 0.0)).value;
        assert!((real_val - complex_val.re).abs() < 1e-10);
    }

    #[test]
    fn test_derivative_negative() {
        // ζ'(s) < 0 for s > 0 when all eigenvalues > 1
        let spec = Spectrum::from_eigenvalues(vec![0.0, 2.0, 3.0, 5.0]);
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        let deriv = zeta.derivative(1.0);
        assert!(deriv < 0.0);
    }

    #[test]
    fn test_second_derivative() {
        let zeta = test_zeta();
        let d2 = zeta.second_derivative(2.0);
        // ζ''(s) > 0 for s > 0 (zeta is convex)
        assert!(d2 > 0.0);
    }

    #[test]
    fn test_evaluate_negative_s() {
        let zeta = test_zeta();
        let val = zeta.evaluate_negative(-1.0);
        // ζ(-1) = Σ λ_n, which should be positive
        assert!(val > 0.0);
    }

    #[test]
    fn test_partial_sums() {
        let zeta = test_zeta();
        let pos_count = zeta.spectrum().positive_eigenvalues().len();
        let sums = zeta.partial_sums(1.0, pos_count);
        assert_eq!(sums.len(), pos_count);
        // Should be monotonically increasing
        for i in 1..sums.len() {
            assert!(sums[i] > sums[i - 1]);
        }
    }

    #[test]
    fn test_evaluate_range() {
        let zeta = test_zeta();
        let results = zeta.evaluate_range(&[1.0, 2.0, 3.0]);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_is_zero() {
        let zeta = test_zeta();
        // Very unlikely to be zero at s=1
        assert!(!zeta.is_zero(1.0, None));
    }

    #[test]
    fn test_from_matrix() {
        let lap = Laplacian::build(LaplacianKind::Complete(4));
        let zeta = SpectralZeta::from_matrix(&lap.to_matrix(), ZetaConfig::default());
        let result = zeta.evaluate(1.0);
        assert!(result.value > 0.0);
    }

    #[test]
    fn test_known_spectrum() {
        // Complete graph K_4: eigenvalues are 4, 4, 4, 0
        // Positive eigenvalues: [4, 4, 4]
        // ζ(1) = 3 * 4^{-1} = 0.75
        // ζ(2) = 3 * 4^{-2} = 0.1875
        let spec = Spectrum::from_eigenvalues(vec![0.0, 4.0, 4.0, 4.0]);
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        assert!((zeta.evaluate(1.0).value - 0.75).abs() < 1e-10);
        assert!((zeta.evaluate(2.0).value - 0.1875).abs() < 1e-10);
    }

    #[test]
    fn test_path_graph_known() {
        // Path P_3: eigenvalues of 1,-1,2;0,-1,1 Laplacian
        // Actually: P_3 eigenvalues are 2 - 2cos(πk/3), k=0,1,2
        // = 0, 1, 3
        // ζ(1) = 1^{-1} + 3^{-1} = 1.333...
        let spec = Spectrum::from_eigenvalues(vec![0.0, 1.0, 3.0]);
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        let expected = 1.0 / 1.0 + 1.0 / 3.0;
        assert!((zeta.evaluate(1.0).value - expected).abs() < 1e-10);
    }
}
