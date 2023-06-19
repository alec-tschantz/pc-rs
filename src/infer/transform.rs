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
    pub fn new(params: Matrix, activation: Activation) -> Self {
        let function = Function::new(activation);
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
