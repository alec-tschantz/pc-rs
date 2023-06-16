use super::matrix::Matrix;

pub enum Activation {
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
            Activation::ReLU => input.apply(|v| v.max(0.0)),
        }
    }

    pub fn backward(&self, input: &Matrix) -> Matrix {
        match self.activation {
            Activation::ReLU => input.apply(|v| if v > 0.0 { 1.0 } else { 0.0 }),
        }
    }
}
