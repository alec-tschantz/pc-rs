use crate::graph::Function;
use crate::linalg::{
    math::{Activation, ActivationFunction},
    matrix::Matrix,
    vector::Vector,
};

use super::variable::GaussianVariable;

pub struct GaussianFunction {
    weights: Matrix,
    bias: Vector,
    activation: ActivationFunction,
    fixed: bool,
}

impl GaussianFunction {
    pub fn new(weights: Matrix, bias: Vector, activation: Activation) -> Self {
        let activation = ActivationFunction::new(activation);
        Self {
            weights,
            bias,
            activation,
            fixed: false,
        }
    }
}

impl Function<GaussianVariable> for GaussianFunction {
    fn forward(&self, inp: &GaussianVariable) -> Matrix {
        assert_eq!(inp.size, self.weights.rows);
        let product = inp.data.matmul(&self.weights);
        &self.activation.forward(&product) + &self.bias
    }

    fn backward(&self, inp: &GaussianVariable, target: &GaussianVariable) -> (Matrix, Matrix) {
        assert_eq!(inp.size, self.weights.rows);
        let product = &inp.data.matmul(&self.weights);
        let pred = &self.activation.forward(&product) + &self.bias;
        let err = &target.data - &pred;
        let fn_deriv = self.activation.backward(&product);
        let err_deriv = &err * &fn_deriv;
        let err_proj = err_deriv.matmul(&self.weights.transpose());
        let source_deriv = err_proj;
        let target_deriv = -err;
        (source_deriv, target_deriv)
    }

    fn backward_params(
        &self,
        inp: &GaussianVariable,
        target: &GaussianVariable,
    ) -> (Matrix, Vector) {
        assert_eq!(inp.size, self.weights.rows);
        let product = &inp.data.matmul(&self.weights);
        let pred = &self.activation.forward(&product) + &self.bias;
        let err = &target.data - &pred;
        let fn_deriv = self.activation.backward(&product);
        let err_deriv = &err * &fn_deriv;
        let weight_deriv = inp.data.transpose().matmul(&err_deriv);
        let bias_deriv = err.sum(0);
        (weight_deriv, bias_deriv)
    }

    fn update(&mut self, derivatives: (Matrix, Vector)) {
        if !self.fixed {
            let (weight_deriv, bias_deriv) = derivatives;
            self.weights += &weight_deriv.apply(|v| v * 0.01);
            self.bias += &bias_deriv.apply(|v| v * 0.01);
        }
    }
}
