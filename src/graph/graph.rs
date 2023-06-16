use std::collections::{HashMap, HashSet};
use crate::graph::transform::Transform;
use crate::graph::variable::Variable;
use crate::linalg::matrix::Matrix;

pub struct Prediction {
    pub variable: Variable,
    pub value: Matrix,
}

pub struct Triplet {
    pub source: Variable,
    pub target: Variable,
    pub transform: Transform,
}

pub struct Graph {
    pub variables: HashSet<Variable>,
    pub triplets: Vec<Triplet>,
    pub preds: HashMap<Variable, Prediction>,
}

impl Graph {
    pub fn new() -> Self {
        let variables = HashSet::new();
        let triplets = Vec::new();
        let preds = HashMap::new();
        Self {
            variables,
            triplets,
            preds,
        }
    }

    pub fn add_variable(&mut self, variable: Variable) {
        self.variables.insert(variable);
    }

    pub fn add_transform(&mut self, source: Variable, target: Variable, transform: Transform) {
        self.triplets.push(Triplet { source, target, transform });
    }

    pub fn forward(&mut self) {
        for triplet in self.triplets.iter() {
            let pred = triplet.transform.forward(&triplet.source);
            let pred_data = Prediction {
                variable: triplet.target.clone(),
                value: pred.clone(),
            };
            self.preds.insert(triplet.target.clone(), pred_data);
        }
    }

    pub fn compute_error(&self, target_variables: &HashSet<Variable>) -> HashMap<Variable, Matrix> {
        let mut errors = HashMap::new();
        for tgt in target_variables.iter() {
            if let Some(pred) = self.preds.get(tgt) {
                let err = &tgt.data - &pred.value;
                errors.insert(pred.variable.clone(), err);
            }
        }
        errors
    }
}