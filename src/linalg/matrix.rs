use std::fmt;
use std::ops::{Add, Mul};

pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { rows, cols, data }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![1.0; cols]; rows],
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn transpose(&self) -> Self {
        let mut transposed_data = vec![vec![0.0; self.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed_data[j][i] = self.data[i][j];
            }
        }
        Self::new(transposed_data)
    }

    pub fn add_matrix(&self, other: &Self) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = Self::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result.data[i][j] = result.data[i][j] + self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }
}

impl<'a, 'b> Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn add(self, other: &'b Matrix) -> Self::Output {
        self.add_matrix(other)
    }
}

impl<'a, 'b> Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, other: &'b Matrix) -> Self::Output {
        self.matmul(other)
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("data", &self.data)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn test_new() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let matrix = Matrix::new(data.clone());
        assert_eq!(matrix.data, data);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 2);
    }

    #[test]
    fn test_ones() {
        let ones = Matrix::ones(2, 2);
        assert_eq!(ones.data, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    }

    #[test]
    fn test_zeros() {
        let zeros = Matrix::zeros(2, 2);
        assert_eq!(zeros.data, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    }

    #[test]
    fn test_add() {
        let a = Matrix::ones(2, 2);
        let b = Matrix::ones(2, 2);
        let result = &a + &b;
        assert_eq!(result.data, vec![vec![2.0, 2.0], vec![2.0, 2.0]]);
    }

    #[test]
    fn test_matmul() {
        let a = Matrix::ones(2, 3);
        let b = Matrix::ones(3, 2);
        let result = &a * &b;
        assert_eq!(result.data, vec![vec![3.0, 3.0], vec![3.0, 3.0]]);
    }

    #[test]
    fn test_transpose() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let matrix = Matrix::new(data.clone());
        let transposed_matrix = matrix.transpose();
        assert_eq!(transposed_matrix.data, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
        assert_eq!(transposed_matrix.rows, 2);
        assert_eq!(transposed_matrix.cols, 2);
    }

    #[test]
    #[should_panic]
    fn test_add_incompatible_matrices() {
        let a = Matrix::ones(2, 2);
        let b = Matrix::ones(3, 2);
        let _ = &a + &b;
    }

    #[test]
    #[should_panic]
    fn test_matmul_incompatible_matrices() {
        let a = Matrix::ones(2, 3);
        let b = Matrix::ones(4, 2);
        let _ = &a * &b;
    }
}
