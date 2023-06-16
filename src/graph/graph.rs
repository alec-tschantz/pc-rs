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
    pub error: Option<Matrix>, 
}

pub struct Graph {
    pub variables: HashSet<Variable>,
    pub triplets: Vec<Triplet>,
    pub preds: HashMap<Variable, Vec<Prediction>>, 
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
        self.triplets.push(Triplet { source, target, transform, error: None }); // Add a None value for the error field when adding a new transform
    }

    pub fn forward(&mut self) {
        for triplet in self.triplets.iter_mut() {
            let pred = triplet.transform.forward(&triplet.source);
            let pred_data = Prediction {
                variable: triplet.target.clone(),
                value: pred.clone(),
            };
            // Change the preds entry for the target variable to a Vec, and push the new prediction
            self.preds.entry(triplet.target.clone()).or_insert_with(Vec::new).push(pred_data);
        }
    }

    // Modify compute_error to compute errors for every triplet
    pub fn compute_error(&mut self) {
        for triplet in self.triplets.iter_mut() {
            if let Some(preds) = self.preds.get(&triplet.target) {
                for pred in preds {
                    let err = &triplet.target.data - &pred.value;
                    triplet.error = Some(err); // Store the error directly in the triplet
                }
            }
        }
    }

    // Helper function to get all errors from a specific source variable
    pub fn get_errors_from_source(&self, source_variable: &Variable) -> Vec<Option<Matrix>> {
        self.triplets
            .iter()
            .filter(|triplet| triplet.source == *source_variable)
            .map(|triplet| triplet.error.clone())
            .collect()
    }

    // Helper function to get all errors that were a result of a specific target variable
    pub fn get_errors_from_target(&self, target_variable: &Variable) -> Vec<Option<Matrix>> {
        self.triplets
            .iter()
            .filter(|triplet| triplet.target == *target_variable)
            .map(|triplet| triplet.error.clone())
            .collect()
    }
}