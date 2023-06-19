mod graph;
mod infer;
mod linalg;

use graph::Graph;
use infer::gaussian::{GaussianFunction, GaussianVariable};
use linalg::{math::Activation, matrix::Matrix};

fn main() {
    let prior_values = Matrix::new(vec![vec![5.0, 5.0, 5.0, 5.0]]);
    let mu_values = Matrix::new(vec![vec![1.0, 1.0, 1.0]]);
    let data_values = Matrix::new(vec![vec![10.0, 10.0, 10.0]]);

    let prior = GaussianVariable::new("prior", prior_values, true);
    let mu = GaussianVariable::new("mu", mu_values, false);
    let data = GaussianVariable::new("data", data_values, true);

    let prior_mu_values = Matrix::new(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0],
    ]);
    let mu_data_values = Matrix::new(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let prior_mu_transform = GaussianFunction::new(prior_mu_values, Activation::Linear, false);
    let mu_data_transform = GaussianFunction::new(mu_data_values, Activation::Linear, true);

    let mut graph = Graph::<GaussianVariable, GaussianFunction>::new();
    let prior_index = graph.add_node(prior);
    let mu_index = graph.add_node(mu);
    let data_index = graph.add_node(data);
    graph.add_edges(vec![
        (prior_index, mu_index, prior_mu_transform),
        (mu_index, data_index, mu_data_transform),
    ]);

    for _ in 0..100 {
        infer::forward(&mut graph);
    }

    let mu = graph.get_node(mu_index).unwrap();
    println!("mu: {:?}", mu);
}
