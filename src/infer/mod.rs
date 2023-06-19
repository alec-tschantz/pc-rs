pub mod gaussian;

use crate::graph::{Graph, Function, Variable};

use self::gaussian::{GaussianDerivative, GaussianFunction, GaussianVariable};
use std::collections::HashMap;

pub type VariableDerivatives = HashMap<usize, Vec<GaussianDerivative>>;

pub fn forward(graph: &mut Graph<GaussianVariable, GaussianFunction, GaussianDerivative>) {
    let mut deltas: VariableDerivatives = HashMap::new();

    for (node_index, _) in graph.get_nodes().enumerate() {
        deltas.insert(node_index, Vec::new());
    }

    for edge in graph.get_edges() {
        let source = &graph.get_node(edge.source).unwrap();
        let target = &graph.get_node(edge.target).unwrap();
        let function = &edge.function;
        let (source_deriv, target_deriv) = function.backward(source, target);

        deltas.get_mut(&edge.target).unwrap().push(target_deriv);
        deltas.get_mut(&edge.source).unwrap().push(source_deriv);
    }

    for (node_index, deltas) in deltas.iter_mut() {
        let node = graph.get_node_mut(*node_index).unwrap();
        node.update(deltas);
    }
}
