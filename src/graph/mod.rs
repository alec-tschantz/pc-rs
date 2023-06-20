use std::collections::HashMap;

use crate::linalg::matrix::Matrix;

pub trait Function<T: Variable> {
    fn forward(&self, input: &T) -> Matrix;
    fn backward(&self, input: &T, target: &T) -> (Matrix, Matrix);
    fn backward_params(&self, input: &T, target: &T) -> Matrix;
    fn update(&mut self, derivative: Matrix);
}

pub trait Variable {
    fn update(&mut self, derivatives: &Vec<Matrix>);
}

pub struct Edge<F> {
    pub source: usize,
    pub target: usize,
    pub function: F,
}

pub struct Graph<T: Variable, F: Function<T>> {
    nodes: Vec<T>,
    edges: Vec<Edge<F>>,
}

impl<T: Variable, F: Function<T>> Graph<T, F> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn forward(&mut self) -> HashMap<(usize, usize), Matrix> {
        let mut preds: HashMap<(usize, usize), Matrix> = HashMap::new();

        for edge in self.get_edges() {
            let source = &self.get_node(edge.source).unwrap();
            let function = &edge.function;
            let pred = function.forward(source);
            preds.insert((edge.source, edge.target), pred);
        }

        return preds;
    }

    pub fn infer(&mut self) {
        let mut deltas: HashMap<usize, Vec<Matrix>> = HashMap::new();
        for (node_index, _) in self.get_nodes().enumerate() {
            deltas.insert(node_index, Vec::new());
        }

        for edge in self.get_edges() {
            let source = &self.get_node(edge.source).unwrap();
            let target = &self.get_node(edge.target).unwrap();
            let function = &edge.function;
            let (source_deriv, target_deriv) = function.backward(source, target);

            deltas.get_mut(&edge.target).unwrap().push(target_deriv);
            deltas.get_mut(&edge.source).unwrap().push(source_deriv);
        }

        for (node_index, deltas) in deltas.iter_mut() {
            let node = self.get_node_mut(*node_index).unwrap();
            node.update(deltas);
        }
    }

    pub fn learn(&mut self) {
        for i in 0..self.edges.len() {
            let source = &self.get_node(self.edges[i].source).unwrap();
            let target = &self.get_node(self.edges[i].target).unwrap();

            let derivative = self.edges[i].function.backward_params(source, target);
            self.edges[i].function.update(derivative);
        }
    }

    pub fn add_node(&mut self, value: T) -> usize {
        self.nodes.push(value);
        self.nodes.len() - 1
    }

    pub fn add_edge(&mut self, source: usize, target: usize, function: F) -> usize {
        let edge = Edge {
            function,
            source,
            target,
        };
        self.edges.push(edge);
        self.edges.len() - 1
    }

    pub fn add_edges(&mut self, edges_info: Vec<(usize, usize, F)>) -> Vec<usize> {
        let mut indices = Vec::new();
        for (source, target, function) in edges_info {
            indices.push(self.add_edge(source, target, function));
        }
        indices
    }

    pub fn get_node(&self, index: usize) -> Option<&T> {
        self.nodes.get(index)
    }

    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut T> {
        self.nodes.get_mut(index)
    }

    pub fn get_nodes(&self) -> std::slice::Iter<T> {
        self.nodes.iter()
    }

    pub fn get_edges(&self) -> std::slice::Iter<Edge<F>> {
        self.edges.iter()
    }
}