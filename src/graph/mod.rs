use crate::linalg::matrix::Matrix;

pub trait Function<T: Variable> {
    fn forward(&self, input: &T) -> Matrix;
    fn backward(&self, input: &T, target: &T) -> (Matrix, Matrix);
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
