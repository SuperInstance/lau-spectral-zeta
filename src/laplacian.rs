//! Laplacian construction utilities.
//!
//! Build Laplacian matrices for common agent topologies so that
//! spectral zeta computations can be applied directly.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Types of agent Laplacians that can be constructed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LaplacianKind {
    /// Graph Laplacian from an adjacency matrix.
    Graph,
    /// Normalized graph Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}.
    Normalized,
    /// Random walk Laplacian: L_rw = I - D^{-1} A.
    RandomWalk,
    /// 1D discrete Laplacian (second difference operator) of given size.
    Discrete1D(usize),
    /// Cycle graph Laplacian of given size.
    Cycle(usize),
    /// Complete graph Laplacian of given size.
    Complete(usize),
    /// Star graph Laplacian of given size.
    Star(usize),
    /// Path graph Laplacian of given size.
    Path(usize),
    /// Grid (2D lattice) Laplacian with given rows and columns.
    Grid { rows: usize, cols: usize },
}

/// A Laplacian with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Laplacian {
    /// The Laplacian matrix.
    pub matrix: Vec<Vec<f64>>,
    /// What kind of Laplacian this is.
    pub kind: LaplacianKind,
    /// Dimension (number of nodes/states).
    pub dim: usize,
}

impl Laplacian {
    /// Build a Laplacian of the given kind.
    pub fn build(kind: LaplacianKind) -> Self {
        let matrix = match &kind {
            LaplacianKind::Discrete1D(n) => discrete_1d(*n),
            LaplacianKind::Cycle(n) => cycle(*n),
            LaplacianKind::Complete(n) => complete(*n),
            LaplacianKind::Star(n) => star(*n),
            LaplacianKind::Path(n) => discrete_1d(*n),
            LaplacianKind::Grid { rows, cols } => grid(*rows, *cols),
            LaplacianKind::Graph | LaplacianKind::Normalized | LaplacianKind::RandomWalk => {
                // These require external adjacency; return identity as placeholder
                let _n = 1;
                vec![vec![1.0]]
            }
        };
        let dim = matrix.len();
        Laplacian { matrix, kind, dim }
    }

    /// Convert to nalgebra DMatrix.
    pub fn to_matrix(&self) -> DMatrix<f64> {
        let n = self.dim;
        let mut data = Vec::with_capacity(n * n);
        for row in &self.matrix {
            data.extend_from_slice(row);
        }
        DMatrix::from_row_slice(n, n, &data)
    }

    /// Build from an adjacency matrix (graph Laplacian).
    pub fn from_adjacency(adj: &[Vec<f64>]) -> Self {
        let n = adj.len();
        let mut lap = vec![vec![0.0; n]; n];
        for i in 0..n {
            let degree: f64 = adj[i].iter().sum();
            lap[i][i] = degree;
            for j in 0..n {
                if i != j {
                    lap[i][j] = -adj[i][j];
                }
            }
        }
        Laplacian {
            matrix: lap,
            kind: LaplacianKind::Graph,
            dim: n,
        }
    }

    /// Build normalized graph Laplacian from adjacency.
    pub fn from_adjacency_normalized(adj: &[Vec<f64>]) -> Self {
        let n = adj.len();
        let degrees: Vec<f64> = adj.iter().map(|row| row.iter().sum()).collect();
        let mut lap = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j && degrees[i] > 0.0 {
                    lap[i][j] = 1.0;
                } else if i != j && adj[i][j] > 0.0 && degrees[i] > 0.0 && degrees[j] > 0.0 {
                    lap[i][j] = -adj[i][j] / (degrees[i] * degrees[j]).sqrt();
                }
            }
        }
        Laplacian {
            matrix: lap,
            kind: LaplacianKind::Normalized,
            dim: n,
        }
    }

    /// Build from raw matrix.
    pub fn from_raw(matrix: Vec<Vec<f64>>) -> Self {
        let n = matrix.len();
        Laplacian {
            matrix,
            kind: LaplacianKind::Graph,
            dim: n,
        }
    }
}

fn discrete_1d(n: usize) -> Vec<Vec<f64>> {
    let mut lap = vec![vec![0.0; n]; n];
    for i in 0..n {
        lap[i][i] = 2.0;
        if i > 0 {
            lap[i][i - 1] = -1.0;
        }
        if i + 1 < n {
            lap[i][i + 1] = -1.0;
        }
    }
    // Fix boundary: Neumann-like
    if n > 0 {
        lap[0][0] = if n == 1 { 0.0 } else { 1.0 };
        lap[n - 1][n - 1] = if n == 1 { 0.0 } else { 1.0 };
    }
    lap
}

fn cycle(n: usize) -> Vec<Vec<f64>> {
    let mut lap = vec![vec![0.0; n]; n];
    for i in 0..n {
        lap[i][i] = 2.0;
        lap[i][(i + 1) % n] = -1.0;
        lap[i][(i + n - 1) % n] = -1.0;
    }
    lap
}

fn complete(n: usize) -> Vec<Vec<f64>> {
    let mut lap = vec![vec![-1.0; n]; n];
    for i in 0..n {
        lap[i][i] = (n - 1) as f64;
    }
    lap
}

fn star(n: usize) -> Vec<Vec<f64>> {
    let mut lap = vec![vec![0.0; n]; n];
    if n <= 1 {
        return lap;
    }
    // Node 0 is center
    lap[0][0] = (n - 1) as f64;
    for i in 1..n {
        lap[0][i] = -1.0;
        lap[i][0] = -1.0;
        lap[i][i] = 1.0;
    }
    lap
}

fn grid(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let n = rows * cols;
    let mut lap = vec![vec![0.0; n]; n];
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let mut deg = 0.0;
            // 4-connectivity
            if r > 0 {
                lap[idx][(r - 1) * cols + c] = -1.0;
                deg += 1.0;
            }
            if r + 1 < rows {
                lap[idx][(r + 1) * cols + c] = -1.0;
                deg += 1.0;
            }
            if c > 0 {
                lap[idx][r * cols + c - 1] = -1.0;
                deg += 1.0;
            }
            if c + 1 < cols {
                lap[idx][r * cols + c + 1] = -1.0;
                deg += 1.0;
            }
            lap[idx][idx] = deg;
        }
    }
    lap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_1d_dim3() {
        let lap = Laplacian::build(LaplacianKind::Path(3));
        assert_eq!(lap.dim, 3);
        let m = lap.to_matrix();
        assert!((m[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((m[(1, 1)] - 2.0).abs() < 1e-10);
        assert!((m[(0, 1)] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cycle_dim4() {
        let lap = Laplacian::build(LaplacianKind::Cycle(4));
        let m = lap.to_matrix();
        assert!((m[(0, 3)] - (-1.0)).abs() < 1e-10);
        assert!((m[(0, 1)] - (-1.0)).abs() < 1e-10);
        assert!((m[(0, 0)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_complete_dim4() {
        let lap = Laplacian::build(LaplacianKind::Complete(4));
        let m = lap.to_matrix();
        assert!((m[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((m[(0, 1)] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_star_dim5() {
        let lap = Laplacian::build(LaplacianKind::Star(5));
        let m = lap.to_matrix();
        assert!((m[(0, 0)] - 4.0).abs() < 1e-10);
        assert!((m[(1, 0)] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_grid() {
        let lap = Laplacian::build(LaplacianKind::Grid { rows: 2, cols: 3 });
        assert_eq!(lap.dim, 6);
    }

    #[test]
    fn test_from_adjacency() {
        let adj = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let lap = Laplacian::from_adjacency(&adj);
        let m = lap.to_matrix();
        assert!((m[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((m[(0, 1)] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_laplacian() {
        let adj = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];
        let lap = Laplacian::from_adjacency_normalized(&adj);
        let m = lap.to_matrix();
        assert!((m[(1, 1)] - 1.0).abs() < 1e-10);
        assert!((m[(0, 2)] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_is_positive_semidefinite() {
        for kind in [
            LaplacianKind::Path(5),
            LaplacianKind::Cycle(5),
            LaplacianKind::Complete(5),
            LaplacianKind::Star(5),
        ] {
            let lap = Laplacian::build(kind);
            let m = lap.to_matrix();
            let eig = m.symmetric_eigen();
            for &e in eig.eigenvalues.iter() {
                assert!(e >= -1e-10, "Negative eigenvalue {} for {:?}", e, lap.kind);
            }
        }
    }

    #[test]
    fn test_from_raw() {
        let raw = vec![vec![2.0, -1.0], vec![-1.0, 2.0]];
        let lap = Laplacian::from_raw(raw);
        assert_eq!(lap.dim, 2);
    }
}
