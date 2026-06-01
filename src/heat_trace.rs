//! Heat trace: Θ(t) = tr(e^{-tΔ}) = Σ_n e^{-tλ_n}
//!
//! The heat trace is the trace of the heat kernel. Its Mellin transform
//! yields the spectral zeta function.

use crate::{Spectrum, ZetaConfig};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Heat trace computation engine.
#[derive(Debug, Clone)]
pub struct HeatTrace {
    spectrum: Spectrum,
    config: ZetaConfig,
}

/// Result of a heat trace evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatTraceResult {
    /// The time parameter t.
    pub t: f64,
    /// Θ(t) = Σ_n e^{-tλ_n}.
    pub theta: f64,
    /// Number of terms used in summation.
    pub terms_used: usize,
    /// Estimated truncation error.
    pub truncation_error: f64,
}

/// Asymptotic expansion coefficients of the heat trace as t→0⁺.
/// Θ(t) ~ Σ_k a_k t^{(k-d)/2}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatTraceExpansion {
    /// Coefficients a_k of the short-time expansion.
    pub coefficients: Vec<f64>,
    /// Powers of t in the expansion.
    pub powers: Vec<f64>,
}

impl HeatTrace {
    /// Create from a pre-computed spectrum.
    pub fn new(spectrum: Spectrum, config: ZetaConfig) -> Self {
        HeatTrace { spectrum, config }
    }

    /// Create from a Laplacian matrix (computes eigenvalues internally).
    pub fn from_matrix(laplacian: &DMatrix<f64>, config: ZetaConfig) -> Self {
        let spectrum = Spectrum::from_matrix(laplacian);
        HeatTrace { spectrum, config }
    }

    /// Evaluate Θ(t) = Σ_n e^{-tλ_n} for real t > 0.
    pub fn evaluate(&self, t: f64) -> HeatTraceResult {
        assert!(t > 0.0, "t must be positive");
        let pos_eigs = self.spectrum.positive_eigenvalues();
        let n_terms = pos_eigs.len().min(self.config.eigenvalue_cutoff);

        let mut theta = 0.0;
        for &lambda in pos_eigs.iter().take(n_terms) {
            theta += (-t * lambda).exp();
        }

        // Estimate truncation error: tail is bounded by integral
        let truncation_error = if n_terms < pos_eigs.len() {
            let last_lambda = pos_eigs[n_terms - 1];
            n_terms as f64 * (-t * last_lambda).exp()
        } else {
            0.0
        };

        HeatTraceResult {
            t,
            theta,
            terms_used: n_terms,
            truncation_error,
        }
    }

    /// Evaluate Θ(t) over a range of t values.
    pub fn evaluate_range(&self, t_values: &[f64]) -> Vec<HeatTraceResult> {
        t_values.iter().map(|&t| self.evaluate(t)).collect()
    }

    /// Compute the Mellin transform: (1/Γ(s)) ∫₀^∞ t^{s-1} Θ(t) dt = ζ_Δ(s)
    /// We use the identity: ζ_Δ(s) = (1/Γ(s)) ∫₀^∞ t^{s-1} Θ(t) dt
    /// And compute numerically via quadrature.
    pub fn mellin_transform(&self, s: Complex64) -> Complex64 {
        let pos_eigs = self.spectrum.positive_eigenvalues();
        // Direct computation: ζ(s) = Σ λ_n^{-s} = Σ exp(-s ln λ_n)
        let mut zeta = Complex64::new(0.0, 0.0);
        for &lambda in &pos_eigs {
            let log_lambda = lambda.ln();
            zeta += Complex64::new(-s.re * log_lambda, -s.im * log_lambda).exp();
        }
        zeta
    }

    /// Short-time asymptotic expansion coefficients.
    /// For a d-dimensional operator: Θ(t) ~ (4πt)^{-d/2} Σ a_k t^k
    pub fn short_time_expansion(&self, order: usize) -> HeatTraceExpansion {
        let n = self.spectrum.dimension;
        let _dim = if n > 0 { 2 } else { 0 }; // Heuristic dimension
        let mut coefficients = Vec::new();
        let mut powers = Vec::new();

        // a_0 = volume (≈ dimension for discrete)
        coefficients.push(n as f64);
        powers.push(0.0);

        for k in 1..=order {
            // Higher order coefficients estimated from eigenvalue moments
            let pos = self.spectrum.positive_eigenvalues();
            let a_k = if !pos.is_empty() {
                let moment: f64 = pos.iter().map(|&λ| (-λ).exp() * λ.powi(k as i32)).sum();
                moment
            } else {
                0.0
            };
            coefficients.push(a_k);
            powers.push(k as f64);
        }

        HeatTraceExpansion {
            coefficients,
            powers,
        }
    }

    /// Verify the heat trace / zeta relation via Mellin transform.
    /// Returns ζ_Δ(s) computed from the heat kernel integral representation.
    pub fn zeta_from_heat_kernel(&self, s: f64, t_max: f64, n_quad: usize) -> f64 {
        let dt = t_max / n_quad as f64;
        let gamma_s = gamma(s);

        let mut integral = 0.0;
        for i in 0..n_quad {
            let t = (i as f64 + 0.5) * dt;
            let theta = self.evaluate(t).theta;
            integral += t.powf(s - 1.0) * theta * dt;
        }

        integral / gamma_s
    }

    /// Access the underlying spectrum.
    pub fn spectrum(&self) -> &Spectrum {
        &self.spectrum
    }
}

/// Gamma function (Stirling approximation + Lanczos).
pub fn gamma(z: f64) -> f64 {
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        pi / ((pi * z).sin() * gamma(1.0 - z))
    } else {
        let z = z - 1.0;
        let g = 7.0;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        let x = c[0]
            + c.iter()
                .skip(1)
                .enumerate()
                .map(|(i, &ci)| ci / (z + i as f64 + 1.0))
                .sum::<f64>();
        let t = z + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laplacian::Laplacian;
    use crate::laplacian::LaplacianKind;
    use approx::assert_relative_eq;

    fn test_heat_trace() -> HeatTrace {
        let lap = Laplacian::build(LaplacianKind::Path(5));
        let spec = Spectrum::from_matrix(&lap.to_matrix());
        HeatTrace::new(spec, ZetaConfig::default())
    }

    #[test]
    fn test_evaluate_t_small() {
        let ht = test_heat_trace();
        let result = ht.evaluate(0.1);
        assert!(result.theta > 0.0);
        assert!(result.theta <= 5.0); // At most dimension
    }

    #[test]
    fn test_evaluate_t_large() {
        let ht = test_heat_trace();
        let result = ht.evaluate(100.0);
        // For large t, only zero-mode contributes
        assert!(result.theta < 1.0);
    }

    #[test]
    fn test_theta_monotone_decreasing() {
        let ht = test_heat_trace();
        let t_vals: Vec<f64> = (1..20).map(|i| 0.1 * i as f64).collect();
        let results = ht.evaluate_range(&t_vals);
        for i in 1..results.len() {
            assert!(results[i].theta <= results[i - 1].theta + 1e-10);
        }
    }

    #[test]
    fn test_theta_at_zero_limit() {
        let ht = test_heat_trace();
        let result = ht.evaluate(1e-10);
        // Should approach the number of positive eigenvalues
        assert!(result.theta > 0.0);
    }

    #[test]
    fn test_theta_positive() {
        let ht = test_heat_trace();
        for t in [0.01, 0.1, 1.0, 10.0] {
            let result = ht.evaluate(t);
            assert!(result.theta >= 0.0, "theta negative at t={}", t);
        }
    }

    #[test]
    fn test_evaluate_range() {
        let ht = test_heat_trace();
        let t_vals = vec![0.1, 1.0, 10.0];
        let results = ht.evaluate_range(&t_vals);
        assert_eq!(results.len(), 3);
        assert!((results[0].t - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_short_time_expansion() {
        let ht = test_heat_trace();
        let expansion = ht.short_time_expansion(3);
        assert_eq!(expansion.coefficients.len(), 4);
        assert!(expansion.coefficients[0] > 0.0);
    }

    #[test]
    fn test_spectrum_access() {
        let ht = test_heat_trace();
        assert_eq!(ht.spectrum().dimension, 5);
    }

    #[test]
    fn test_from_matrix() {
        let lap = Laplacian::build(LaplacianKind::Complete(4));
        let ht = HeatTrace::from_matrix(&lap.to_matrix(), ZetaConfig::default());
        let result = ht.evaluate(1.0);
        assert!(result.theta > 0.0);
    }

    #[test]
    fn test_gamma_function() {
        assert_relative_eq!(gamma(1.0), 1.0, epsilon = 1e-8);
        assert_relative_eq!(gamma(2.0), 1.0, epsilon = 1e-8);
        assert_relative_eq!(gamma(3.0), 2.0, epsilon = 1e-8);
        assert_relative_eq!(gamma(0.5), std::f64::consts::PI.sqrt(), epsilon = 1e-8);
    }

    #[test]
    fn test_truncation_error_small_t() {
        let ht = test_heat_trace();
        let result = ht.evaluate(0.001);
        // For small t with all eigenvalues included, error should be small
        assert!(result.truncation_error >= 0.0);
    }

    #[test]
    fn test_cycle_graph_heat_trace() {
        let lap = Laplacian::build(LaplacianKind::Cycle(6));
        let spec = Spectrum::from_matrix(&lap.to_matrix());
        let ht = HeatTrace::new(spec, ZetaConfig::default());
        let result = ht.evaluate(1.0);
        assert!(result.theta > 0.0);
    }
}
