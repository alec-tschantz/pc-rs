mod graph;
mod infer;
mod linalg;

use graph::Graph;
use infer::{transform::Transform, variable::Variable};
use linalg::{math::Activation, matrix::Matrix};

fn reset_graph(graph: &mut Graph<Variable, Transform>, mu_index: usize) {
    let mu = graph.get_node_mut(mu_index).unwrap();
    mu.data = Matrix::new(vec![vec![1.0, 1.0, 1.0]]);
}

fn main() {
    let prior_values = Matrix::new(vec![vec![1.0, 1.0, 1.0, 1.0]]);
    let mu_values = Matrix::new(vec![vec![1.0, 1.0, 1.0]]);
    let data_values = Matrix::new(vec![vec![10.0, 10.0, 10.0]]);

    let prior = Variable::new("prior", prior_values, true);
    let mu = Variable::new("mu", mu_values, false);
    let data = Variable::new("data", data_values, true);

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
    let prior_mu_transform = Transform::new(prior_mu_values, Activation::Linear, false);
    let mu_data_transform = Transform::new(mu_data_values, Activation::Linear, true);

    let mut graph = Graph::<Variable, Transform>::new();
    let prior_index = graph.add_node(prior);
    let mu_index = graph.add_node(mu);
    let data_index = graph.add_node(data);
    graph.add_edge(prior_index, mu_index, prior_mu_transform);
    graph.add_edge(mu_index, data_index, mu_data_transform);

    for _ in 0..100 {
        reset_graph(&mut graph, mu_index);
        for _ in 0..100 {
            let deltas = infer::forward(&graph);
            infer::apply_deltas(&mut graph, deltas);
        }
        let param_deltas = infer::forward_params(&graph);
        infer::apply_param_deltas(&mut graph, param_deltas);
    }
}
