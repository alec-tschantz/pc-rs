mod graph;
mod infer;
mod linalg;

use graph::Graph;
use infer::{apply_deltas, forward, transform::Transform, variable::Variable};
use linalg::{math::Activation, matrix::Matrix};

fn main() {
    let prior_1_values = Matrix::new(vec![vec![20.0, 20.0, 20.0, 20.0]]);
    let prior_2_values = Matrix::new(vec![vec![20.0, 20.0, 20.0]]);
    let mu_values = Matrix::new(vec![vec![1.0, 1.0, 1.0]]);
    let data_values = Matrix::new(vec![vec![10.0, 10.0, 10.0]]);

    let prior_1 = Variable::new("prior_1", prior_1_values, true);
    let prior_2 = Variable::new("prior_2", prior_2_values, true);
    let mu = Variable::new("mu", mu_values, false);
    let data = Variable::new("data", data_values, true);

    let prior_1_mu_values = Matrix::new(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0],
    ]);
    let prior_2_mu_values = Matrix::new(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let mu_data_values = Matrix::new(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let prior_1_mu_transform = Transform::new(prior_1_mu_values, Activation::Linear);
    let prior_2_mu_transform = Transform::new(prior_2_mu_values, Activation::Linear);
    let mu_data_transform = Transform::new(mu_data_values, Activation::Linear);

    let mut graph = Graph::<Variable, Transform>::new();
    let prior_1_index = graph.add_node(prior_1);
    let prior_2_index = graph.add_node(prior_2);
    let mu_index = graph.add_node(mu);
    let data_index = graph.add_node(data);
    graph.add_edge(prior_1_index, mu_index, prior_1_mu_transform);
    graph.add_edge(prior_2_index, mu_index, prior_2_mu_transform);
    graph.add_edge(mu_index, data_index, mu_data_transform);

    for _ in 0..100 {
        let deltas = forward(&graph);
        apply_deltas(&mut graph, deltas);
    }

    let mu = graph.get_node(mu_index).unwrap();
    println!("mu: {:?}", mu);
}
