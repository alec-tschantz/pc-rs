use super::matrix::Matrix;

#[derive(Clone, Debug)]
pub enum Activation {
    Linear,
}

#[derive(Clone, Debug)]
pub struct ActivationFunction {
    activation: Activation,
}

impl ActivationFunction {
    pub fn new(activation: Activation) -> Self {
        Self { activation }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        match self.activation {
            Activation::Linear => input.clone(),
        }
    }

    pub fn backward(&self, input: &Matrix) -> Matrix {
        match self.activation {
            Activation::Linear => Matrix::ones(input.rows, input.cols),
        }
    }
}
