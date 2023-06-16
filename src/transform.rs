use std::f64;

use crate::linalg::matrix::Matrix;

pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn relu_deriv(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub struct Transform {
    weights: Matrix,
}

impl Transform {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Matrix::new(vec![vec![0.1; output_size]; input_size]);
        Self { weights}
    }

    pub fn forward(&self, inp: &Matrix) -> Matrix {
        assert_eq!(inp.cols, self.weights.rows);
        let product = (inp * &self.weights).apply(relu);
        product
    }

    pub fn backward(&self, inp: &Matrix, err: &Matrix) -> Matrix {
        assert_eq!(inp.cols, self.weights.rows);
        assert_eq!(err.rows, inp.rows);
        let product = inp * &self.weights;
        let fn_deriv = product.apply(relu_deriv);
        let out = &(err * &fn_deriv) * &self.weights.transpose();
        out
    }
}
