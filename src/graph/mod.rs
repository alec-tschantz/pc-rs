pub struct Graph<N, E> {
    nodes: Vec<N>,
    edges: Vec<Edge<E>>,
}

pub struct Edge<E> {
    pub value: E,
    pub source: usize,
    pub target: usize,
}

impl<N, E> Graph<N, E>
where
    N: PartialEq,
{
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, value: N) -> usize {
        self.nodes.push(value);
        self.nodes.len() - 1
    }

    pub fn add_edge(&mut self, source: usize, target: usize, value: E) -> usize {
        let edge = Edge {
            value,
            source,
            target,
        };
        self.edges.push(edge);
        self.edges.len() - 1
    }

    pub fn get_node(&self, index: usize) -> Option<&N> {
        self.nodes.get(index)
    }

    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut N> {
        self.nodes.get_mut(index)
    }

    pub fn get_edge(&self, index: usize) -> Option<&Edge<E>> {
        self.edges.get(index)
    }

    pub fn get_edge_mut(&mut self, index: usize) -> Option<&mut Edge<E>> {
        self.edges.get_mut(index)
    }

    pub fn get_edges(&self) -> std::slice::Iter<Edge<E>> {
        self.edges.iter()
    }
}
