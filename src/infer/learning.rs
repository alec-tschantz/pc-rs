// pub fn backward_params(&self, inp: &Variable, err: &Matrix) -> Matrix {
//     assert_eq!(inp.size, self.params.rows);
//     let product = &inp.data.matmul(&self.params);
//     let fn_deriv = self.function.backward(&product);
//     let err_deriv = err * &fn_deriv;
//     let out = inp.data.transpose().matmul(&err_deriv);
//     out
// }

// pub fn forward_params(graph: &Graph<Variable, Transform>) -> TransformDeltas {
//     let mut deltas = TransformDeltas::new();

//     for (index, edge) in graph.get_edges().enumerate() {
//         let source = graph.get_node(edge.source).unwrap();
//         let target = graph.get_node(edge.target).unwrap();
//         let transform = &edge.value;
//         let prediction = transform.forward(source);
//         let error = &target.data - &prediction;
//         let delta = transform.backward_params(source, &error);

//         deltas.insert(index, delta);
//     }

//     return deltas;
// }

// pub fn apply_param_deltas(graph: &mut Graph<Variable, Transform>, deltas: TransformDeltas) {
//     for (index, delta) in deltas {
//         let edge = graph.get_edge_mut(index).unwrap();
//         let transform = &mut edge.value;
//         if !transform.fixed {
//             transform.params += &delta.apply(|v| v * 0.01);
//         }
//     }
// }