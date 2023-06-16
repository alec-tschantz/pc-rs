use crate::linalg::matrix::Matrix;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Sub;

pub struct Variable {
    pub name: String,
    pub size: usize,
    pub data: Matrix,
}

impl Variable {
    pub fn new(name: &str, size: usize) -> Self {
        let data = Matrix::ones(1, size);
        let name = name.to_string();
        Self { name, size, data }
    }
}

impl Sub<&Matrix> for &Variable {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Self::Output {
        assert_eq!(self.data.rows, other.rows);
        assert_eq!(self.data.cols, other.cols);
        &self.data - other
    }
}

impl Hash for Variable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Variable {}

impl Clone for Variable {
    fn clone(&self) -> Self {
        let name = self.name.clone();
        let size = self.size;
        let data = self.data.clone();
        Self { name, size, data }
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("size", &self.size)
            .field("data", &self.data)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{Matrix, Variable};

    #[test]
    fn test_subtract_matrix_from_variable() {
        let var = Variable::new("test", 2);
        let mat = Matrix::new(vec![vec![0.5, 0.5]]);
        let result = &var - &mat;
        assert_eq!(result.data, vec![vec![0.5, 0.5]]);
    }

    #[test]
    #[should_panic]
    fn test_subtract_incompatible_matrix_from_variable() {
        let var = Variable::new("test", 2);
        let mat = Matrix::new(vec![vec![0.5]]);
        let _ = &var - &mat;
    }
}
