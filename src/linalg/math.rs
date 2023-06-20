use super::matrix::Matrix;

#[allow(dead_code)]
pub enum Activation {
    Linear,
    ReLU,
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
            Activation::ReLU => input.apply(|elem| elem.max(0.0)),
        }
    }

    pub fn backward(&self, input: &Matrix) -> Matrix {
        match self.activation {
            Activation::Linear => Matrix::ones(input.rows, input.cols),
            Activation::ReLU => input.apply(|elem| if elem > 0.0 { 1.0 } else { 0.0 }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let activation = ActivationFunction::new(Activation::Linear);
        let input = Matrix::new(vec![vec![-1.0, 2.0], vec![3.0, -4.0]]);
        let output = activation.forward(&input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_linear_backward() {
        let activation = ActivationFunction::new(Activation::Linear);
        let input = Matrix::new(vec![vec![-1.0, 2.0], vec![3.0, -4.0]]);
        let output = activation.backward(&input);
        let expected = Matrix::new(vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_relu_forward() {
        let activation = ActivationFunction::new(Activation::ReLU);
        let input = Matrix::new(vec![vec![-1.0, 2.0], vec![3.0, -4.0]]);
        let output = activation.forward(&input);
        let expected = Matrix::new(vec![vec![0.0, 2.0], vec![3.0, 0.0]]);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_relu_backward() {
        let activation = ActivationFunction::new(Activation::ReLU);
        let input = Matrix::new(vec![vec![-1.0, 2.0], vec![3.0, -4.0]]);
        let output = activation.backward(&input);
        let expected = Matrix::new(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        assert_eq!(output, expected);
    }
}
