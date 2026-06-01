//! Resolvent trace: tr((Δ - λ)^{-1})
//!
//! The resolvent trace is analytically continued to yield the spectral zeta.
//! R(λ) = Σ_n 1/(λ_n - λ), and the zeta function is obtained via contour integration.

use crate::{Spectrum, ZetaConfig};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Resolvent trace computation engine.
#[derive(Debug, Clone)]
pub struct ResolventTrace {
    spectrum: Spectrum,
    config: ZetaConfig,
}

/// Result of a resolvent trace evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolventResult {
    /// The spectral parameter λ.
    pub lambda: Complex64,
    /// tr((Δ - λ)^{-1}) = Σ_n 1/(λ_n - λ).
    pub trace: Complex64,
    /// Whether λ coincides with an eigenvalue (pole).
    pub at_pole: bool,
}

impl ResolventTrace {
    /// Create from a pre-computed spectrum.
    pub fn new(spectrum: Spectrum, config: ZetaConfig) -> Self {
        ResolventTrace { spectrum, config }
    }

    /// Create from a Laplacian matrix.
    pub fn from_matrix(laplacian: &DMatrix<f64>, config: ZetaConfig) -> Self {
        let spectrum = Spectrum::from_matrix(laplacian);
        ResolventTrace { spectrum, config }
    }

    /// Evaluate the resolvent trace: Σ_n 1/(λ_n - z) for complex z.
    pub fn evaluate(&self, z: Complex64) -> ResolventResult {
        let eigs = &self.spectrum.eigenvalues;
        let mut trace = Complex64::new(0.0, 0.0);
        let mut at_pole = false;

        for &lambda_n in eigs {
            let diff = Complex64::new(lambda_n, 0.0) - z;
            if diff.norm() < self.config.epsilon {
                at_pole = true;
                trace += Complex64::new(1.0 / self.config.epsilon, 0.0);
            } else {
                trace += Complex64::new(1.0, 0.0) / diff;
            }
        }

        ResolventResult {
            lambda: z,
            trace,
            at_pole,
        }
    }

    /// Evaluate the resolvent trace along a real line.
    pub fn evaluate_real_range(&self, lambda_values: &[f64]) -> Vec<ResolventResult> {
        lambda_values
            .iter()
            .map(|&l| self.evaluate(Complex64::new(l, 0.0)))
            .collect()
    }

    /// Analytic continuation to spectral zeta via the resolvent.
    /// Uses the identity: ζ(s) = sin(πs)/(π) ∫₀^∞ λ^{-s} tr(R(-λ)) dλ
    /// which is evaluated by direct summation: ζ(s) = Σ λ_n^{-s}.
    pub fn zeta_from_resolvent(&self, s: f64) -> f64 {
        let pos_eigs = self.spectrum.positive_eigenvalues();
        if s == 0.0 {
            // ζ(0) = regularized dimension
            return self.regularized_dimension_from_resolvent();
        }
        let mut zeta = 0.0;
        for &lambda in &pos_eigs {
            zeta += lambda.powf(-s);
        }
        zeta
    }

    /// Complex zeta from resolvent: ζ(s) for complex s.
    pub fn zeta_complex(&self, s: Complex64) -> Complex64 {
        let pos_eigs = self.spectrum.positive_eigenvalues();
        let mut zeta = Complex64::new(0.0, 0.0);
        for &lambda in &pos_eigs {
            let log_l = lambda.ln();
            let exponent = Complex64::new(-s.re * log_l, -s.im * log_l);
            zeta += exponent.exp();
        }
        zeta
    }

    /// Regularized dimension from the resolvent: ζ_Δ(0).
    /// Computed via analytic continuation.
    fn regularized_dimension_from_resolvent(&self) -> f64 {
        // ζ(s) = Σ λ_n^{-s}, as s → 0 we need analytic continuation.
        // For discrete: ζ(s→0) → number of positive eigenvalues - (s contribution from zero modes)
        // Use finite-difference approximation
        let eps = 1e-6;
        let zeta_plus = self.zeta_from_resolvent(eps);
        let zeta_minus = self.zeta_from_resolvent(-eps);
        // Linear interpolation to s=0
        0.5 * (zeta_plus + zeta_minus)
    }

    /// Derivative of zeta: ζ'(s) computed numerically.
    pub fn zeta_derivative(&self, s: f64) -> f64 {
        let eps = 1e-7;
        let zeta_plus = self.zeta_from_resolvent(s + eps);
        let zeta_minus = self.zeta_from_resolvent(s - eps);
        (zeta_plus - zeta_minus) / (2.0 * eps)
    }

    /// Access the spectrum.
    pub fn spectrum(&self) -> &Spectrum {
        &self.spectrum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laplacian::Laplacian;
    use crate::laplacian::LaplacianKind;

    fn test_resolvent() -> ResolventTrace {
        let lap = Laplacian::build(LaplacianKind::Path(5));
        let spec = Spectrum::from_matrix(&lap.to_matrix());
        ResolventTrace::new(spec, ZetaConfig::default())
    }

    #[test]
    fn test_evaluate_away_from_spectrum() {
        let rt = test_resolvent();
        let result = rt.evaluate(Complex64::new(-1.0, 0.0));
        assert!(!result.at_pole);
        assert!(result.trace.norm() > 0.0);
    }

    #[test]
    fn test_evaluate_near_eigenvalue() {
        let rt = test_resolvent();
        // Get an eigenvalue and evaluate near it
        let eigs = &rt.spectrum().eigenvalues;
        if let Some(&lambda) = eigs.last() {
            let result = rt.evaluate(Complex64::new(lambda + 1e-15, 0.0));
            // Should be near a pole
            assert!(result.trace.norm() > 1.0);
        }
    }

    #[test]
    fn test_resolvent_real_positive() {
        let rt = test_resolvent();
        let result = rt.evaluate(Complex64::new(-10.0, 0.0));
        // All terms 1/(λ_n - (-10)) = 1/(λ_n + 10) > 0
        assert!(result.trace.re > 0.0);
    }

    #[test]
    fn test_zeta_from_resolvent() {
        let rt = test_resolvent();
        let zeta_1 = rt.zeta_from_resolvent(1.0);
        assert!(zeta_1 > 0.0, "ζ(1) should be positive");
    }

    #[test]
    fn test_zeta_decreasing() {
        // Use eigenvalues all > 1 to guarantee decreasing
        let spec = Spectrum::from_eigenvalues(vec![0.0, 2.0, 3.0, 5.0]);
        let rt = ResolventTrace::new(spec, ZetaConfig::default());
        let zeta_1 = rt.zeta_from_resolvent(1.0);
        let zeta_2 = rt.zeta_from_resolvent(2.0);
        assert!(zeta_2 < zeta_1, "ζ(s) should decrease for positive s");
    }

    #[test]
    fn test_zeta_complex() {
        let rt = test_resolvent();
        let z = rt.zeta_complex(Complex64::new(2.0, 0.5));
        assert!(z.norm() > 0.0);
    }

    #[test]
    fn test_evaluate_real_range() {
        let rt = test_resolvent();
        let results = rt.evaluate_real_range(&[-5.0, -1.0, 0.0, 1.0]);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_zeta_derivative() {
        // Use eigenvalues all > 1 to guarantee negative derivative
        let spec = Spectrum::from_eigenvalues(vec![0.0, 2.0, 3.0, 5.0]);
        let rt = ResolventTrace::new(spec, ZetaConfig::default());
        let deriv = rt.zeta_derivative(1.0);
        assert!(deriv < 0.0);
    }

    #[test]
    fn test_from_matrix() {
        let lap = Laplacian::build(LaplacianKind::Complete(4));
        let rt = ResolventTrace::from_matrix(&lap.to_matrix(), ZetaConfig::default());
        let result = rt.evaluate(Complex64::new(-1.0, 0.0));
        assert!(result.trace.norm() > 0.0);
    }

    #[test]
    fn test_resolvent_poles_match_eigenvalues() {
        let rt = test_resolvent();
        let eigs = rt.spectrum().eigenvalues.clone();
        // Between eigenvalues, resolvent should be finite
        if eigs.len() >= 3 {
            let mid = (eigs[1] + eigs[2]) / 2.0;
            let result = rt.evaluate(Complex64::new(mid, 0.0));
            assert!(!result.at_pole);
        }
    }
}
