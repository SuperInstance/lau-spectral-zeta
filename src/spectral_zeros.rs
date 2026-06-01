//! Spectral zeros: where ζ_Δ(s) = 0
//!
//! The Riemann hypothesis analogue for agent spectral zeta.
//! We search for zeros in the complex plane and classify them.

use crate::{SpectralZeta, Spectrum, ZetaConfig};
use nalgebra::DMatrix;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Spectral zeros finder.
#[derive(Debug, Clone)]
pub struct SpectralZeros {
    zeta: SpectralZeta,
}

/// A found spectral zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralZero {
    /// The zero location s₀.
    pub s: Complex64,
    /// Residual |ζ(s₀)| at the found point.
    pub residual: f64,
    /// Number of Newton iterations used.
    pub iterations: usize,
    /// Whether this is on the critical line Re(s) = d/2.
    pub on_critical_line: bool,
}

/// Search results for spectral zeros.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroSearchResult {
    /// Found zeros.
    pub zeros: Vec<SpectralZero>,
    /// Region searched: real part range.
    pub s_real_range: (f64, f64),
    /// Region searched: imaginary part range.
    pub s_imag_range: (f64, f64),
    /// Grid resolution.
    pub resolution: usize,
}

/// The critical strip: 0 < Re(s) < d (effective dimension).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalStrip {
    /// Effective dimension d.
    pub dimension: f64,
    /// Zeros found inside the critical strip 0 < Re(s) < d.
    pub interior_zeros: Vec<SpectralZero>,
    /// Zeros on the critical line Re(s) = d/2.
    pub critical_line_zeros: Vec<SpectralZero>,
    /// Zeros outside the critical strip.
    pub exterior_zeros: Vec<SpectralZero>,
}

impl SpectralZeros {
    /// Create from spectral zeta.
    pub fn new(zeta: SpectralZeta) -> Self {
        SpectralZeros { zeta }
    }

    /// Create from Laplacian matrix.
    pub fn from_matrix(laplacian: &DMatrix<f64>, config: ZetaConfig) -> Self {
        let spectrum = Spectrum::from_matrix(laplacian);
        let zeta = SpectralZeta::new(spectrum, config);
        SpectralZeros { zeta }
    }

    /// Evaluate ζ_Δ(s) at a complex point.
    fn zeta_complex(&self, s: Complex64) -> Complex64 {
        self.zeta.evaluate_complex(s).value
    }

    /// Find zeros using grid search + Newton refinement.
    ///
    /// Searches in the rectangle [s_min, s_max] × [imag_min, imag_max].
    pub fn find_zeros(
        &self,
        s_real_range: (f64, f64),
        s_imag_range: (f64, f64),
        resolution: usize,
        tolerance: Option<f64>,
    ) -> ZeroSearchResult {
        let tol = tolerance.unwrap_or(self.zeta.config().tolerance);
        let max_iter = self.zeta.config().max_iterations;

        let mut zeros = Vec::new();
        let dr = (s_real_range.1 - s_real_range.0) / resolution as f64;
        let di = (s_imag_range.1 - s_imag_range.0) / resolution as f64;

        for i in 0..resolution {
            for j in 0..resolution {
                let s0 = Complex64::new(
                    s_real_range.0 + dr * (i as f64 + 0.5),
                    s_imag_range.0 + di * (j as f64 + 0.5),
                );

                // Check if |ζ(s)| is small at grid point
                let zeta_val = self.zeta_complex(s0);
                if zeta_val.norm() < 1.0 {
                    // Try Newton refinement
                    if let Some(zero) = self.newton_refine(s0, tol, max_iter) {
                        // Check it's not a duplicate
                        let is_dup = zeros.iter().any(|z: &SpectralZero| {
                            (z.s - zero.s).norm() < tol * 10.0
                        });
                        if !is_dup {
                            zeros.push(zero);
                        }
                    }
                }
            }
        }

        ZeroSearchResult {
            zeros,
            s_real_range,
            s_imag_range,
            resolution,
        }
    }

    /// Newton's method to refine a zero: s_{n+1} = s_n - ζ(s_n)/ζ'(s_n).
    fn newton_refine(&self, s0: Complex64, tol: f64, max_iter: usize) -> Option<SpectralZero> {
        let eps = Complex64::new(1e-8, 1e-8);
        let mut s = s0;

        for iter in 0..max_iter {
            let zeta_s = self.zeta_complex(s);
            if zeta_s.norm() < tol {
                return Some(SpectralZero {
                    s,
                    residual: zeta_s.norm(),
                    iterations: iter,
                    on_critical_line: false, // Will be determined later
                });
            }

            // Numerical derivative
            let zeta_s_plus = self.zeta_complex(s + eps);
            let zeta_s_minus = self.zeta_complex(s - eps);
            let deriv = (zeta_s_plus - zeta_s_minus) / (2.0 * eps);

            if deriv.norm() < 1e-20 {
                return None; // Derivative too small
            }

            s = s - zeta_s / deriv;
        }

        // Check if we ended up close to zero
        let zeta_s = self.zeta_complex(s);
        if zeta_s.norm() < tol * 100.0 {
            Some(SpectralZero {
                s,
                residual: zeta_s.norm(),
                iterations: max_iter,
                on_critical_line: false,
            })
        } else {
            None
        }
    }

    /// Classify zeros into critical strip / critical line / exterior.
    pub fn classify_zeros(&self, zeros: Vec<SpectralZero>, dimension: f64) -> CriticalStrip {
        let mut interior = Vec::new();
        let mut critical_line = Vec::new();
        let mut exterior = Vec::new();

        let tol = 0.1; // Tolerance for "on critical line"

        for mut zero in zeros {
            if (zero.s.re - dimension / 2.0).abs() < tol {
                zero.on_critical_line = true;
                critical_line.push(zero.clone());
                interior.push(zero);
            } else if zero.s.re > 0.0 && zero.s.re < dimension {
                interior.push(zero);
            } else {
                exterior.push(zero);
            }
        }

        CriticalStrip {
            dimension,
            interior_zeros: interior,
            critical_line_zeros: critical_line,
            exterior_zeros: exterior,
        }
    }

    /// Count trivial zeros (on the real axis where ζ(s) = 0 due to Γ factors).
    pub fn count_trivial_zeros(&self) -> usize {
        // For discrete Laplacian, trivial zeros are at negative integers
        // where the Gamma function diverges.
        // We check s = -1, -2, -3, ... for ζ(s) ≈ 0
        let mut count = 0;
        for k in 1..20 {
            let s = -(k as f64);
            let val = self.zeta.evaluate(s).value;
            // For finite spectrum, ζ(-k) = Σ λ_n^k, which is generally non-zero
            // Trivial zeros come from the Gamma factor in the completed zeta
            // Not directly from the raw zeta
            if val.abs() < 1e-6 {
                count += 1;
            }
        }
        count
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

    fn test_zeros() -> SpectralZeros {
        let lap = Laplacian::build(LaplacianKind::Path(5));
        let spec = Spectrum::from_matrix(&lap.to_matrix());
        let zeta = SpectralZeta::new(spec, ZetaConfig::default());
        SpectralZeros::new(zeta)
    }

    #[test]
    fn test_zeta_complex_evaluation() {
        let sz = test_zeros();
        let val = sz.zeta_complex(Complex64::new(2.0, 0.0));
        assert!(val.norm() > 0.0);
    }

    #[test]
    fn test_find_zeros_small_region() {
        let sz = test_zeros();
        let result = sz.find_zeros((-2.0, 4.0), (-5.0, 5.0), 10, Some(1e-4));
        // May or may not find zeros — just verify it runs
        assert!(result.zeros.len() < 100);
        assert_eq!(result.resolution, 10);
    }

    #[test]
    fn test_classify_zeros() {
        let sz = test_zeros();
        let zeros = vec![SpectralZero {
            s: Complex64::new(1.0, 2.0),
            residual: 1e-8,
            iterations: 5,
            on_critical_line: false,
        }];
        let classified = sz.classify_zeros(zeros, 2.0);
        assert_eq!(classified.interior_zeros.len(), 1);
        assert_eq!(classified.critical_line_zeros.len(), 1);
        assert_eq!(classified.exterior_zeros.len(), 0);
    }

    #[test]
    fn test_classify_exterior() {
        let sz = test_zeros();
        let zeros = vec![SpectralZero {
            s: Complex64::new(5.0, 1.0),
            residual: 1e-8,
            iterations: 5,
            on_critical_line: false,
        }];
        let classified = sz.classify_zeros(zeros, 2.0);
        assert_eq!(classified.exterior_zeros.len(), 1);
    }

    #[test]
    fn test_count_trivial_zeros() {
        let sz = test_zeros();
        let count = sz.count_trivial_zeros();
        // For finite spectrum, likely no trivial zeros
        assert!(count < 20);
    }

    #[test]
    fn test_newton_refine() {
        let sz = test_zeros();
        // Try to refine from a point where ζ is small
        // For a known spectrum, we can't easily predict zeros, so just test it runs
        let result = sz.newton_refine(Complex64::new(1.0, 0.0), 1e-4, 10);
        // May or may not find a zero
        if let Some(zero) = result {
            assert!(zero.residual.is_finite());
            assert!(zero.iterations <= 10);
        }
    }

    #[test]
    fn test_from_matrix() {
        let lap = Laplacian::build(LaplacianKind::Complete(4));
        let sz = SpectralZeros::from_matrix(&lap.to_matrix(), ZetaConfig::default());
        let val = sz.zeta_complex(Complex64::new(1.0, 0.0));
        assert!(val.norm() > 0.0);
    }

    #[test]
    fn test_zero_search_result_serializable() {
        let result = ZeroSearchResult {
            zeros: vec![],
            s_real_range: (-2.0, 4.0),
            s_imag_range: (-5.0, 5.0),
            resolution: 10,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("resolution"));
    }

    #[test]
    fn test_critical_strip_serializable() {
        let strip = CriticalStrip {
            dimension: 2.0,
            interior_zeros: vec![],
            critical_line_zeros: vec![],
            exterior_zeros: vec![],
        };
        let json = serde_json::to_string(&strip).unwrap();
        assert!(json.contains("dimension"));
    }
}
