pub mod variable;
pub mod transform;

use std::collections::HashMap;

use crate::{graph::Graph, linalg::matrix::Matrix};

use self::{variable::Variable, transform::Transform};


pub fn forward(graph: &Graph<Variable, Transform>) -> HashMap<usize, Matrix> {
    let mut deltas = HashMap::<usize, Matrix>::new();

    for edge in graph.get_edges() {
        let source = graph.get_node(edge.source).unwrap();
        let target = graph.get_node(edge.target).unwrap();
        let transform = &edge.value;
        let prediction = transform.forward(source);
        let error = &target.data - &prediction;
        let delta_for_target = error.clone();
        let delta_for_source = transform.backward(source, &error);
        

        if let Some(e_delta) = deltas.get_mut(&edge.target) {
            *e_delta -= &delta_for_target;
        } else {
            deltas.insert(edge.target, -delta_for_target);
        }

        if let Some(e_delta) = deltas.get_mut(&edge.source) {
            *e_delta += &delta_for_source;
        } else {
            deltas.insert(edge.source, delta_for_source);
        }
    }

    return deltas;
}

pub fn apply_deltas(graph: &mut Graph<Variable, Transform>, deltas: HashMap<usize, Matrix>) {
    for (index, delta) in deltas {
        let node = graph.get_node_mut(index).unwrap();
        if !node.fixed {
            node.data += &delta.apply(|v| v * 0.1);
        }
    }
}
