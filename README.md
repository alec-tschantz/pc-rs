# pc-rs

Predictive coding in Rust.

```rust
mod gaussian;
mod graph;
mod linalg;

use crate::gaussian::{function::GaussianFunction, variable::GaussianVariable};
use graph::Graph;
use linalg::{math::Activation, matrix::Matrix};

fn main() {
    let prior_values = Matrix::new(vec![vec![10.0; 32]]);
    let mu_values = Matrix::new(vec![vec![1.0; 32]]);
    let data_values = Matrix::new(vec![vec![10.0; 32]]);

    let prior = GaussianVariable::new("prior", prior_values, true);
    let mu = GaussianVariable::new("mu", mu_values, false);
    let data = GaussianVariable::new("data", data_values, true);

    let prior_mu_transform = GaussianFunction::new(Matrix::identity(32), Activation::Linear, false);
    let mu_data_transform = GaussianFunction::new(Matrix::identity(32), Activation::Linear, false);

    let mut graph = Graph::<GaussianVariable, GaussianFunction>::new();
    let prior_index = graph.add_node(prior);
    let mu_index = graph.add_node(mu);
    let data_index = graph.add_node(data);
    graph.add_edges(vec![
        (prior_index, mu_index, prior_mu_transform),
        (mu_index, data_index, mu_data_transform),
    ]);


    for _ in 0..100 {
        graph.infer();
    }
    graph.learn();

    let mu = graph.get_node(mu_index).unwrap();
    println!("mu: {:?}", mu);

    let preds = graph.forward();
    let mu_data_pred = preds.get(&(mu_index, data_index)).unwrap();
    println!("mu_data_pred: {:?}", mu_data_pred);
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
