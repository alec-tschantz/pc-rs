use crate::graph::Variable;
use crate::linalg::matrix::Matrix;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GaussianVariable {
    pub size: usize,
    pub data: Matrix,
    fixed: bool,
}

impl GaussianVariable {
    pub fn new(data: Matrix, fixed: bool) -> Self {
        let size = data.cols;
        Self { size, data, fixed }
    }

    pub fn set_data(&mut self, data: Matrix) {
        assert_eq!(data.cols, self.size);
        self.data = data;
    }
}

impl Variable for GaussianVariable {
    fn update(&mut self, derivatives: &Vec<Matrix>) {
        if !self.fixed {
            for derivative in derivatives {
                self.data += &derivative.apply(|v| v * 0.01);
            }
        }
    }
}
