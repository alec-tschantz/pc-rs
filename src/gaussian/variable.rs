use crate::graph::Variable;
use crate::linalg::matrix::Matrix;

#[derive(Clone, Debug, PartialEq, Eq)]
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

impl Variable for GaussianVariable {
    fn update(&mut self, derivatives: &Vec<Matrix>) {
        if !self.fixed {
            for derivative in derivatives {
                self.data += &derivative.apply(|v| v * 0.01);
            }
        }
    }
}
