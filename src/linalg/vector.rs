use std::{fmt, ops::AddAssign};

use rand::Rng;

pub struct Vector {
    pub size: usize,
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        let size = data.len();
        Self { size, data }
    }

    pub fn ones(size: usize) -> Self {
        Self {
            size,
            data: vec![1.0; size],
        }
    }

    pub fn zeros(size: usize) -> Self {
        Self {
            size,
            data: vec![0.0; size],
        }
    }

    pub fn random(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..size).map(|_| rng.gen::<f64>()).collect();
        Self { size, data }
    }

    pub fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Self::new(data)
    }
}

impl AddAssign<&Vector> for Vector {
    fn add_assign(&mut self, other: &Vector) {
        assert_eq!(self.size, other.size);
        for i in 0..self.size {
            self.data[i] += other.data[i];
        }
    }
}

impl Clone for Vector {
    fn clone(&self) -> Self {
        Self {
            size: self.size,
            data: self.data.clone(),
        }
    }
}

impl fmt::Debug for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vector")
            .field("size", &self.size)
            .field("data", &self.data)
            .finish()
    }
}
