use std::fmt;
use std::hash::{Hash, Hasher};

use crate::graph::{Function, Variable};
use crate::linalg::{
    math::{Activation, ActivationFunction},
    matrix::Matrix,
};

pub struct GaussianDerivative {
    pub data: Matrix,
}

impl GaussianDerivative {
    pub fn new(data: Matrix) -> Self {
        Self { data }
    }
}

pub struct GaussianVariable {
    pub name: String,
    pub size: usize,
    pub data: Matrix,
    pub fixed: bool,
}

impl GaussianVariable {
    pub fn new(name: &str, data: Matrix, fixed: bool) -> Self {
        let name = name.to_string();
        let size = data.cols;
        Self {
            name,
            size,
            data,
            fixed,
        }
    }
}

impl Variable<GaussianDerivative> for GaussianVariable {
    fn update(&mut self, derivatives: &Vec<GaussianDerivative>) {
        if !self.fixed {
            for derivative in derivatives {
                self.data += &derivative.data.apply(|v| v * 0.01);
            }
        }
    }
}

impl Hash for GaussianVariable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl PartialEq for GaussianVariable {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for GaussianVariable {}

impl Clone for GaussianVariable {
    fn clone(&self) -> Self {
        let name = self.name.clone();
        let fixed = self.fixed;
        let size = self.size;
        let data = self.data.clone();
        Self {
            name,
            size,
            data,
            fixed,
        }
    }
}

impl fmt::Debug for GaussianVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GaussianVariable")
            .field("name", &self.name)
            .field("size", &self.size)
            .field("data", &self.data)
            .field("fixed", &self.fixed)
            .finish()
    }
}

pub struct GaussianFunction {
    pub params: Matrix,
    pub fixed: bool,
    function: ActivationFunction,
}

impl GaussianFunction {
    pub fn new(params: Matrix, activation: Activation, fixed: bool) -> Self {
        let function = ActivationFunction::new(activation);
        Self {
            function,
            params,
            fixed,
        }
    }
}

impl Function<GaussianVariable, GaussianDerivative> for GaussianFunction {
    fn forward(&self, inp: &GaussianVariable) -> GaussianVariable {
        assert_eq!(inp.size, self.params.rows);
        let product = inp.data.matmul(&self.params);
        let out = self.function.forward(&product);
        GaussianVariable::new(&inp.name, out, false)
    }

    fn backward(
        &self,
        inp: &GaussianVariable,
        target: &GaussianVariable,
    ) -> (GaussianDerivative, GaussianDerivative) {
        assert_eq!(inp.size, self.params.rows);
        let product = &inp.data.matmul(&self.params);
        let pred = self.function.forward(&product);
        let err = &target.data - &pred;
        let fn_deriv = self.function.backward(&product);
        let err_deriv = &err * &fn_deriv;
        let out = err_deriv.matmul(&self.params.transpose());
        let source_deriv = GaussianDerivative::new(out);
        let target_deriv = GaussianDerivative::new(-err);
        (source_deriv, target_deriv)
    }
}
