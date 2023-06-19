use crate::linalg::{
    math::{Activation, Function},
    matrix::Matrix,
};

use super::variable::Variable;

pub struct Transform {
    pub params: Matrix,
    pub fixed: bool,
    function: Function,
}

impl Transform {
    pub fn new(params: Matrix, activation: Activation, fixed: bool) -> Self {
        let function = Function::new(activation);
        Self { function, params, fixed }
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

    pub fn backward_params(&self, inp: &Variable, err: &Matrix) -> Matrix {
        assert_eq!(inp.size, self.params.rows);
        let product = &inp.data.matmul(&self.params);
        let fn_deriv = self.function.backward(&product);
        let err_deriv = err * &fn_deriv;
        let out = inp.data.transpose().matmul(&err_deriv);
        out
    }
}
