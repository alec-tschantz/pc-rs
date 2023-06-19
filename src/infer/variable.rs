use crate::linalg::matrix::Matrix;
use std::fmt;
use std::hash::{Hash, Hasher};

pub struct Variable {
    pub name: String,
    pub fixed: bool,
    pub size: usize,
    pub data: Matrix,
}

impl Variable {
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

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("name", &self.name)
            .field("size", &self.size)
            .field("data", &self.data)
            .field("fixed", &self.fixed)
            .finish()
    }
}
