pub mod transform;
pub mod variable;

use std::collections::HashMap;

use crate::{graph::Graph, linalg::matrix::Matrix};

use self::{transform::Transform, variable::Variable};

pub type VariableDeltas = HashMap<usize, Matrix>;
pub type TransformDeltas = HashMap<usize, Matrix>;

pub fn forward(graph: &Graph<Variable, Transform>) -> VariableDeltas {
    let mut deltas = VariableDeltas::new();

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

pub fn forward_params(graph: &Graph<Variable, Transform>) -> TransformDeltas {
    let mut deltas = TransformDeltas::new();

    for (index, edge) in graph.get_edges().enumerate() {
        let source = graph.get_node(edge.source).unwrap();
        let target = graph.get_node(edge.target).unwrap();
        let transform = &edge.value;
        let prediction = transform.forward(source);
        let error = &target.data - &prediction;
        let delta = transform.backward_params(source, &error);

        deltas.insert(index, delta);
    }

    return deltas;
}

pub fn apply_deltas(graph: &mut Graph<Variable, Transform>, deltas: VariableDeltas) {
    for (index, delta) in deltas {
        let node = graph.get_node_mut(index).unwrap();
        if !node.fixed {
            node.data += &delta.apply(|v| v * 0.01);
        }
    }
}

pub fn apply_param_deltas(graph: &mut Graph<Variable, Transform>, deltas: TransformDeltas) {
    for (index, delta) in deltas {
        let edge = graph.get_edge_mut(index).unwrap();
        let transform = &mut edge.value;
        if !transform.fixed {
            transform.params += &delta.apply(|v| v * 0.01);
        }
    }
}
