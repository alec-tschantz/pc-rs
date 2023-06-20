pub mod gaussian;
pub mod graph;
pub mod linalg;

pub use crate::gaussian::{function::GaussianFunction, variable::GaussianVariable};
pub use crate::graph::{Function, Graph, Variable};
pub use crate::linalg::{math::Activation, matrix::Matrix};
