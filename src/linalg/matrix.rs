use std::fmt;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::vector::Vector;

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
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

    pub fn identity(size: usize) -> Self {
        let mut data = vec![vec![0.0; size]; size];
        for i in 0..size {
            data[i][i] = 1.0;
        }
        Self {
            rows: size,
            cols: size,
            data,
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..rows)
            .map(|_| (0..cols).map(|_| rng.gen::<f64>()).collect())
            .collect();
        Self { rows, cols, data }
    }

    pub fn normal(rows: usize, cols: usize, mean: f64, std: f64) -> Self {
        let normal = Normal::new(mean, std).unwrap();
        let mut rng = rand::thread_rng();

        let data = (0..rows)
            .map(|_| (0..cols).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        Self { rows, cols, data }
    }

    pub fn kaiming_normal(rows: usize, cols: usize) -> Self {
        let kappa = f64::sqrt(2.0 / rows as f64);
        let normal = Normal::new(0.0, kappa).unwrap();
        let mut rng = rand::thread_rng();

        let data = (0..rows)
            .map(|_| (0..cols).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        Self { rows, cols, data }
    }

    pub fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let data = self
            .data
            .iter()
            .map(|row| row.iter().map(|&x| f(x)).collect())
            .collect();
        Self::new(data)
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

    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }

    pub fn sum(&self, axis: usize) -> Vector {
        match axis {
            0 => {
                let mut result = vec![0.0; self.cols];
                for i in 0..self.rows {
                    for j in 0..self.cols {
                        result[j] += self.data[i][j];
                    }
                }
                Vector::new(result)
            }
            1 => {
                let mut result = vec![0.0; self.rows];
                for i in 0..self.rows {
                    for j in 0..self.cols {
                        result[i] += self.data[i][j];
                    }
                }
                Vector::new(result)
            }
            _ => panic!("Axis {} is not supported.", axis),
        }
    }
}

impl<'a, 'b> Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn add(self, other: &'b Matrix) -> Self::Output {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }
}

impl<'a, 'b> Add<&'b Vector> for &'a Matrix {
    type Output = Matrix;

    fn add(self, vector: &'b Vector) -> Self::Output {
        assert_eq!(self.cols, vector.size);

        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + vector.data[j];
            }
        }
        result
    }
}

impl Add<f64> for Matrix {
    type Output = Matrix;

    fn add(self, scalar: f64) -> Self::Output {
        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + scalar;
            }
        }
        result
    }
}

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, other: &Matrix) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] += other.data[i][j];
            }
        }
    }
}

impl<'a, 'b> Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, other: &'b Matrix) -> Self::Output {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        result
    }
}

impl Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Self::Output {
        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * scalar;
            }
        }
        result
    }
}

impl<'a, 'b> Sub<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn sub(self, other: &'b Matrix) -> Self::Output {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        result
    }
}

impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, other: &Matrix) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] -= other.data[i][j];
            }
        }
    }
}

impl Neg for Matrix {
    type Output = Matrix;

    fn neg(self) -> Self::Output {
        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = -self.data[i][j];
            }
        }
        result
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                if (self.data[i][j] - other.data[i][j]).abs() > f64::EPSILON {
                    return false;
                }
            }
        }

        true
    }
}

impl Eq for Matrix {}

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
    fn test_subtract() {
        let a = Matrix::ones(2, 2);
        let b = Matrix::ones(2, 2);
        let result = &a - &b;
        assert_eq!(result.data, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    }

    #[test]
    fn test_mul_elementwise() {
        let a = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::new(vec![vec![2.0, 3.0], vec![4.0, 5.0]]);
        let result = &a * &b;
        assert_eq!(result.data, vec![vec![2.0, 6.0], vec![12.0, 20.0]]);
    }

    #[test]
    fn test_matmul() {
        let a = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::new(vec![vec![2.0, 3.0], vec![4.0, 5.0]]);
        let result = a.matmul(&b);
        assert_eq!(result.data, vec![vec![10.0, 13.0], vec![22.0, 29.0]]);
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
    fn test_mul_incompatible_matrices() {
        let a = Matrix::ones(2, 2);
        let b = Matrix::ones(3, 2);
        let _ = &a * &b;
    }
}
