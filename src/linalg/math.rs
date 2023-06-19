use super::matrix::Matrix;

pub enum Activation {
    Linear,
}

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
