# pc-rs

Predictive coding in Rust.

```rust
mod gaussian;
mod graph;
mod linalg;

use crate::gaussian::{function::GaussianFunction, variable::GaussianVariable};
use graph::Graph;
use linalg::{math::Activation, matrix::Matrix, vector::Vector};

const MU_SIZE: usize = 3;
const DATA_A_SIZE: usize = 6;
const DATA_B_SIZE: usize = 4;
const BATCH_SIZE: usize = 1;
const NUM_EPOCHS: usize = 100;
const NUM_ITERATIONS: usize = 100;

fn main() {
    let mu = GaussianVariable::new(Matrix::normal(BATCH_SIZE, MU_SIZE, 0.0, 0.05), false);
    let data_a = GaussianVariable::new(Matrix::ones(BATCH_SIZE, DATA_A_SIZE) * 2.0, true);
    let data_b = GaussianVariable::new(Matrix::ones(BATCH_SIZE, DATA_B_SIZE) * 4.0, true);

    let mu_data_a = GaussianFunction::new(
        Matrix::normal(MU_SIZE, DATA_A_SIZE, 0.0, 0.05),
        Vector::zeros(DATA_A_SIZE),
        Activation::Linear,
    );
    let mu_data_b = GaussianFunction::new(
        Matrix::normal(MU_SIZE, DATA_B_SIZE, 0.0, 0.05),
        Vector::zeros(DATA_B_SIZE),
        Activation::Linear,
    );

    let mut graph = Graph::<GaussianVariable, GaussianFunction>::new();
    let mu_index = graph.add_node(mu);
    let data_a_index = graph.add_node(data_a);
    let data_b_index = graph.add_node(data_b);

    graph.add_edges(vec![
        (mu_index, data_a_index, mu_data_a),
        (mu_index, data_b_index, mu_data_b),
    ]);

    for _ in 0..NUM_EPOCHS {
        let mu = graph.get_node_mut(mu_index).unwrap();
        mu.set_data(Matrix::normal(BATCH_SIZE, MU_SIZE, 0.0, 0.05));

        for _ in 0..NUM_ITERATIONS {
            graph.infer();
        }
        graph.learn();
    }

    let preds = graph.forward();
    let mu_data_a_pred = preds.get(&(mu_index, data_a_index)).unwrap();
    let mu_data_b_pred = preds.get(&(mu_index, data_b_index)).unwrap();
    println!("mu_data_a_pred: {:?}", mu_data_a_pred);
    println!("mu_data_b_pred: {:?}", mu_data_b_pred);
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
