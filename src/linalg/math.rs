use super::matrix::Matrix;

pub enum Activation {
    Linear,
    ReLU,
}

pub struct Function {
    activation: Activation,
}

impl Function {
    pub fn new(activation: Activation) -> Self {
        Self { activation }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        match self.activation {
            Activation::Linear => input.clone(),
            Activation::ReLU => input.apply(|v| v.max(0.0)),
            
        }
    }

    pub fn backward(&self, input: &Matrix) -> Matrix {
        match self.activation {
            Activation::Linear => Matrix::ones(input.rows, input.cols),
            Activation::ReLU => input.apply(|v| if v > 0.0 { 1.0 } else { 0.0 }),
        }
    }
}
