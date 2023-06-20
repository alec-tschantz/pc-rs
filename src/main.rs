mod gaussian;
mod graph;
mod linalg;

use crate::gaussian::{function::GaussianFunction, variable::GaussianVariable};
use graph::Graph;
use linalg::{math::Activation, matrix::Matrix};

fn main() {
    let prior = GaussianVariable::new("prior", Matrix::ones(1, 32) * 4.0, true);
    let mu = GaussianVariable::new("mu", Matrix::random(1, 32), false);
    let data = GaussianVariable::new("data", Matrix::ones(1, 32) * 2.0, true);

    let prior_mu_function = GaussianFunction::new(Matrix::identity(32), Activation::Linear);
    let mu_data_function = GaussianFunction::new(Matrix::identity(32), Activation::Linear);

    let mut graph = Graph::<GaussianVariable, GaussianFunction>::new();
    let prior_index = graph.add_node(prior);
    let mu_index = graph.add_node(mu);
    let data_index = graph.add_node(data);
    
    graph.add_edges(vec![
        (prior_index, mu_index, prior_mu_function),
        (mu_index, data_index, mu_data_function),
    ]);

    for _ in 0..500 {
        graph.infer();
    }
    graph.learn();

    let mu = graph.get_node(mu_index).unwrap();
    println!("mu: {:?}", mu);

    let preds = graph.forward();
    let mu_data_pred = preds.get(&(mu_index, data_index)).unwrap();
    println!("mu_data_pred: {:?}", mu_data_pred);
}
