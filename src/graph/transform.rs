use crate::linalg::{
    math::{Activation, Function},
    matrix::Matrix,
};

use super::variable::Variable;

pub struct Transform {
    params: Matrix,
    function: Function,
}

impl Transform {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let params = Matrix::new(vec![vec![0.1; output_size]; input_size]);
        let function = Function::new(Activation::ReLU);
        Self { function, params }
    }

    pub fn forward(&self, inp: &Variable) -> Matrix {
        assert_eq!(inp.size, self.params.rows);
        let product = inp.data.matmul(&self.params);
        let out = self.function.forward(&product);
        out
    }

    pub fn backward(&self, inp: &Variable, err: &Matrix) -> Matrix {
        assert_eq!(inp.size, self.params.rows);
        let product = &inp.data.matmul(&self.params);
        let fn_deriv = self.function.backward(&product);
        let err_deriv = err * &fn_deriv;
        let out = err_deriv.matmul(&self.params.transpose());
        out
    }
}
