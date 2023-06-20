use crate::graph::Function;
use crate::linalg::{
    math::{Activation, ActivationFunction},
    matrix::Matrix,
};

use super::variable::GaussianVariable;

#[derive(Clone, Debug)]
pub struct GaussianFunction {
    pub params: Matrix,
    activation: ActivationFunction,
    fixed: bool,
}

impl GaussianFunction {
    pub fn new(params: Matrix, activation: Activation) -> Self {
        let activation = ActivationFunction::new(activation);
        Self {
            params,
            activation,
            fixed: false,
        }
    }
}

impl Function<GaussianVariable> for GaussianFunction {
    fn forward(&self, inp: &GaussianVariable) -> Matrix {
        assert_eq!(inp.size, self.params.rows);
        let product = inp.data.matmul(&self.params);
        self.activation.forward(&product)
    }

    fn backward(&self, inp: &GaussianVariable, target: &GaussianVariable) -> (Matrix, Matrix) {
        assert_eq!(inp.size, self.params.rows);
        let product = &inp.data.matmul(&self.params);
        let pred = self.activation.forward(&product);
        let err = &target.data - &pred;
        let fn_deriv = self.activation.backward(&product);
        let err_deriv = &err * &fn_deriv;
        let err_proj = err_deriv.matmul(&self.params.transpose());
        let source_deriv = err_proj;
        let target_deriv = -err;
        (source_deriv, target_deriv)
    }

    fn backward_params(&self, inp: &GaussianVariable, target: &GaussianVariable) -> Matrix {
        assert_eq!(inp.size, self.params.rows);
        let product = &inp.data.matmul(&self.params);
        let pred = self.activation.forward(&product);
        let err = &target.data - &pred;
        let fn_deriv = self.activation.backward(&product);
        let err_deriv = &err * &fn_deriv;
        inp.data.transpose().matmul(&err_deriv)
    }

    fn update(&mut self, derivative: Matrix) {
        if !self.fixed {
            self.params += &derivative.apply(|v| v * 0.001);
        }
    }
}
