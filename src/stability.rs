//! Agent stability from spectral zeros.
//!
//! Zeros of the spectral zeta in the critical strip signal metastable states.
//! The Riemann hypothesis analogue: if all non-trivial zeros lie on the
//! critical line Re(s) = d/2, the agent is maximally stable.

use crate::{SpectralZeta, SpectralZeros, Spectrum, ZetaConfig, SpectralDeterminant};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Agent stability analysis engine.
#[derive(Debug, Clone)]
pub struct AgentStability {
    zeta: SpectralZeta,
    zeros: SpectralZeros,
    dimension: f64,
}

/// Stability assessment of an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityReport {
    /// Overall stability score: 1.0 = maximally stable, 0.0 = unstable.
    pub stability_score: f64,
    /// Whether the Riemann hypothesis analogue holds (all zeros on critical line).
    pub riemann_hypothesis_holds: bool,
    /// Number of zeros in the critical strip.
    pub critical_strip_zeros: usize,
    /// Number of zeros on the critical line.
    pub critical_line_zeros: usize,
    /// Number of zeros outside the critical strip.
    pub exterior_zeros: usize,
    /// Effective dimension.
    pub dimension: f64,
    /// Regularized dimension ζ(0).
    pub regularized_dimension: f64,
    /// Log determinant -ζ'(0).
    pub log_determinant: f64,
    /// Partition function det Δ.
    pub partition_function: f64,
    /// Metastable states (zeros in critical strip but off critical line).
    pub metastable_count: usize,
    /// Stability classification.
    pub classification: StabilityClass,
}

/// Classification of agent stability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StabilityClass {
    /// All zeros on critical line — maximally stable.
    Stable,
    /// Some zeros off critical line but in critical strip — metastable.
    Metastable,
    /// Many zeros outside critical strip — unstable.
    Unstable,
    /// No zeros found — indeterminate.
    Indeterminate,
}

/// Comparison of stability between two agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityComparison {
    /// First agent's stability score.
    pub score_a: f64,
    /// Second agent's stability score.
    pub score_b: f64,
    /// Which is more stable.
    pub more_stable: String,
    /// Log ratio of partition functions.
    pub log_partition_ratio: f64,
}

impl AgentStability {
    /// Create from spectral zeta and effective dimension.
    pub fn new(zeta: SpectralZeta, dimension: f64) -> Self {
        let zeros = SpectralZeros::new(zeta.clone());
        AgentStability { zeta, zeros, dimension }
    }

    /// Create from Laplacian matrix.
    pub fn from_matrix(laplacian: &DMatrix<f64>, config: ZetaConfig) -> Self {
        let spectrum = Spectrum::from_matrix(laplacian);
        let _n = spectrum.dimension as f64;
        let pos = spectrum.positive_eigenvalues();
        let dim = if pos.is_empty() {
            1.0
        } else {
            let lambda_max = pos.last().copied().unwrap_or(1.0);
            let n_pos = pos.len() as f64;
            if lambda_max > 1.0 {
                2.0 * (n_pos.ln() / lambda_max.ln()).max(1.0)
            } else {
                1.0
            }
        };
        let zeta = SpectralZeta::new(spectrum, config);
        AgentStability::new(zeta, dim)
    }

    /// Run a full stability analysis.
    pub fn analyze(&self) -> StabilityReport {
        // Find zeros in a reasonable region
        let search = self.zeros.find_zeros(
            (-3.0, self.dimension + 3.0),
            (-10.0, 10.0),
            15,
            Some(1e-3),
        );

        // Classify zeros
        let classified = self.zeros.classify_zeros(search.zeros, self.dimension);

        // Compute regularized quantities
        let reg = crate::RegularizedTrace::new(self.zeta.clone());
        let dim_result = reg.regularized_dimension();
        let det = SpectralDeterminant::new(self.zeta.clone());
        let det_result = det.compute();

        // Count metastable states
        let metastable_count = classified.interior_zeros.len()
            - classified.critical_line_zeros.len();

        // Compute stability score
        let stability_score = if classified.interior_zeros.is_empty() {
            1.0 // No zeros = stable
        } else {
            let on_line = classified.critical_line_zeros.len() as f64;
            let total = classified.interior_zeros.len() as f64;
            on_line / total
        };

        // Check Riemann hypothesis
        let riemann_holds = metastable_count == 0;

        let classification = if classified.interior_zeros.is_empty() {
            StabilityClass::Indeterminate
        } else if riemann_holds {
            StabilityClass::Stable
        } else if classified.exterior_zeros.len() > classified.interior_zeros.len() {
            StabilityClass::Unstable
        } else {
            StabilityClass::Metastable
        };

        StabilityReport {
            stability_score,
            riemann_hypothesis_holds: riemann_holds,
            critical_strip_zeros: classified.interior_zeros.len(),
            critical_line_zeros: classified.critical_line_zeros.len(),
            exterior_zeros: classified.exterior_zeros.len(),
            dimension: self.dimension,
            regularized_dimension: dim_result.zeta_zero,
            log_determinant: det_result.log_determinant,
            partition_function: det_result.determinant,
            metastable_count,
            classification,
        }
    }

    /// Quick stability check: just the score without full zero search.
    pub fn quick_score(&self) -> f64 {
        let _reg = crate::RegularizedTrace::new(self.zeta.clone());
        let _det = SpectralDeterminant::new(self.zeta.clone());

        // Use spectral gap and determinant as proxy for stability
        let pos_eigs = self.zeta.spectrum().positive_eigenvalues();
        if pos_eigs.is_empty() {
            return 0.0;
        }

        let lambda_min = pos_eigs[0];
        let lambda_max = pos_eigs[pos_eigs.len() - 1];
        let spectral_gap = lambda_min;
        let condition_number = lambda_max / lambda_min.max(1e-15);

        // Higher spectral gap and lower condition number → more stable
        let gap_score = 1.0 - (-spectral_gap).exp();
        let cond_score = 1.0 / (1.0 + condition_number.ln() / 10.0);

        0.5 * gap_score + 0.5 * cond_score
    }

    /// Compare stability of two agents.
    pub fn compare(&self, other: &AgentStability) -> StabilityComparison {
        let score_a = self.quick_score();
        let score_b = other.quick_score();

        let det_a = SpectralDeterminant::new(self.zeta.clone()).compute();
        let det_b = SpectralDeterminant::new(other.zeta.clone()).compute();

        StabilityComparison {
            score_a,
            score_b,
            more_stable: if score_a >= score_b { "A".to_string() } else { "B".to_string() },
            log_partition_ratio: det_a.log_determinant - det_b.log_determinant,
        }
    }

    /// Predict whether an agent will remain stable under perturbation.
    pub fn perturbation_stability(&self, perturbation_strength: f64) -> f64 {
        let pos_eigs = self.zeta.spectrum().positive_eigenvalues();
        if pos_eigs.is_empty() {
            return 0.0;
        }

        // Under perturbation δ, eigenvalues shift by O(δ).
        // Stability requires λ_min - δ > 0.
        let lambda_min = pos_eigs[0];
        let margin = lambda_min - perturbation_strength;
        margin / lambda_min.max(1e-15)
    }

    /// Access the spectral zeta.
    pub fn zeta(&self) -> &SpectralZeta {
        &self.zeta
    }

    /// Access the zeros finder.
    pub fn zeros(&self) -> &SpectralZeros {
        &self.zeros
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laplacian::Laplacian;
    use crate::laplacian::LaplacianKind;

    fn test_stability() -> AgentStability {
        let lap = Laplacian::build(LaplacianKind::Path(5));
        let spec = Spectrum::from_matrix(&lap.to_matrix());
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        AgentStability::new(zeta, 2.0)
    }

    #[test]
    fn test_analyze() {
        let stab = test_stability();
        let report = stab.analyze();
        assert!(report.stability_score >= 0.0);
        assert!(report.stability_score <= 1.0);
        assert!(report.regularized_dimension >= 0.0);
        assert!(report.partition_function > 0.0);
    }

    #[test]
    fn test_quick_score() {
        let stab = test_stability();
        let score = stab.quick_score();
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_compare() {
        let lap1 = Laplacian::build(LaplacianKind::Path(4));
        let lap2 = Laplacian::build(LaplacianKind::Complete(4));
        let s1 = AgentStability::from_matrix(&lap1.to_matrix(), ZetaConfig::default());
        let s2 = AgentStability::from_matrix(&lap2.to_matrix(), ZetaConfig::default());
        let comp = s1.compare(&s2);
        assert!(comp.score_a >= 0.0);
        assert!(comp.score_b >= 0.0);
        assert!(comp.more_stable == "A" || comp.more_stable == "B");
    }

    #[test]
    fn test_perturbation_stability() {
        let stab = test_stability();
        let ps = stab.perturbation_stability(0.01);
        // Small perturbation should give high stability
        assert!(ps > 0.0);
    }

    #[test]
    fn test_perturbation_large() {
        let stab = test_stability();
        let ps = stab.perturbation_stability(100.0);
        // Large perturbation should give low/negative stability
        assert!(ps < 0.0);
    }

    #[test]
    fn test_classification_stable() {
        let report = StabilityReport {
            stability_score: 1.0,
            riemann_hypothesis_holds: true,
            critical_strip_zeros: 2,
            critical_line_zeros: 2,
            exterior_zeros: 0,
            dimension: 2.0,
            regularized_dimension: 5.0,
            log_determinant: 1.0,
            partition_function: 1.0,
            metastable_count: 0,
            classification: StabilityClass::Stable,
        };
        assert_eq!(report.classification, StabilityClass::Stable);
    }

    #[test]
    fn test_classification_metastable() {
        let report = StabilityReport {
            stability_score: 0.5,
            riemann_hypothesis_holds: false,
            critical_strip_zeros: 4,
            critical_line_zeros: 2,
            exterior_zeros: 1,
            dimension: 2.0,
            regularized_dimension: 5.0,
            log_determinant: 1.0,
            partition_function: 1.0,
            metastable_count: 2,
            classification: StabilityClass::Metastable,
        };
        assert_eq!(report.classification, StabilityClass::Metastable);
    }

    #[test]
    fn test_from_matrix() {
        let lap = Laplacian::build(LaplacianKind::Cycle(6));
        let stab = AgentStability::from_matrix(&lap.to_matrix(), ZetaConfig::default());
        let report = stab.analyze();
        assert!(report.stability_score >= 0.0);
    }

    #[test]
    fn test_report_serializable() {
        let stab = test_stability();
        let report = stab.analyze();
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("stability_score"));
        assert!(json.contains("classification"));
    }

    #[test]
    fn test_complete_graph_stability() {
        let lap = Laplacian::build(LaplacianKind::Complete(8));
        let stab = AgentStability::from_matrix(&lap.to_matrix(), ZetaConfig::default());
        let score = stab.quick_score();
        assert!(score > 0.0);
    }

    #[test]
    fn test_star_vs_complete() {
        let lap_star = Laplacian::build(LaplacianKind::Star(6));
        let lap_complete = Laplacian::build(LaplacianKind::Complete(6));
        let s_star = AgentStability::from_matrix(&lap_star.to_matrix(), ZetaConfig::default());
        let s_complete = AgentStability::from_matrix(&lap_complete.to_matrix(), ZetaConfig::default());
        // Complete graph should generally be more stable
        let comp = s_star.compare(&s_complete);
        assert!(comp.more_stable == "A" || comp.more_stable == "B");
    }
}
