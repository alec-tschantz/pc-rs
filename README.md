# pc-rs

Predictive coding in Rust.

```rust

mod linalg;
mod graph;

use crate::graph::{variable::Variable, transform::Transform, graph::Graph};
use std::collections::HashSet;

fn main() {
    let mu = Variable::new("mu", 64);
    let data = Variable::new("data", 32);
    let transform = Transform::new(mu.size, data.size);

    let mut graph = Graph::new();
    graph.add_variable(mu.clone());
    graph.add_variable(data.clone());
    graph.add_transform(mu.clone(), data.clone(), transform);

    graph.forward();
    
    let mut target_variables = HashSet::new();
    target_variables.insert(data.clone());
    let errors = graph.compute_error(&target_variables);

    println!("Errors: {:?}", errors);
}
```

To run a demo:

```bash
cargo run
```

To run tests:

```bash
cargo test
```

## Examples 
 
 ```rust
pub trait Forward<T, E> {
    fn forward(&self, input: &T, target: &T) -> Result<E>;
}

pub trait Backward<T, E, D> {
    fn backward(&self, input: &T, target: &T, error: &E) -> Result<(D, D)>;
}

pub trait Update<D> {
    fn update(&mut self, derivative: &D);
}

pub trait Function<T, E, D>: Forward<T, E> + Backward<T, E, D> {}

pub struct Edge<F> {
    pub source: usize,
    pub target: usize,
    pub func: F,
}

pub struct Graph<T: Update<D>, F: Function<T, E, D>, E, D> {
    pub nodes: Vec<T>,
    pub edges: Vec<Edge<F>>,
}
```

```rust
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct GaussianVariable {
    pub mean: f64,
    pub stddev: f64,
}

#[derive(Debug)]
pub struct GaussianFunction;

#[derive(Debug, Clone, Copy)]
pub struct GaussianError {
    pub error: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct GaussianDerivative {
    pub d_mean: f64,
    pub d_stddev: f64,
}

impl Forward<GaussianVariable, GaussianError> for GaussianFunction {
    fn forward(&self, input: &GaussianVariable, target: &GaussianVariable) -> Result<GaussianError> {
        Ok(GaussianError {
            error: (input.mean - target.mean).abs(),
        })
    }
}

impl Backward<GaussianVariable, GaussianError, GaussianDerivative> for GaussianFunction {
    fn backward(
        &self,
        input: &GaussianVariable,
        target: &GaussianVariable,
        error: &GaussianError,
    ) -> Result<(GaussianDerivative, GaussianDerivative)> {
        Ok((
            GaussianDerivative {
                d_mean: error.error,
                d_stddev: 0.0,
            },
            GaussianDerivative {
                d_mean: -error.error,
                d_stddev: 0.0,
            },
        ))
    }
}

impl Update<GaussianDerivative> for GaussianVariable {
    fn update(&mut self, derivative: &GaussianDerivative) {
        self.mean += derivative.d_mean;
        self.stddev += derivative.d_stddev;
    }
}


impl Function<GaussianVariable, GaussianError, GaussianDerivative> for GaussianFunction {}
```