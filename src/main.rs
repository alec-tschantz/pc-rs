
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