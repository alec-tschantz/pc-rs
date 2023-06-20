pub mod linalg;
pub mod graph;
pub mod gaussian;

pub use crate::graph::{Variable, Function, Graph};
pub use crate::gaussian::{function::GaussianFunction, variable::GaussianVariable};
pub use crate::linalg::{math::Activation, matrix::Matrix};