# pc-rs

Predictive coding in Rust.

```rust
mod linalg;
use crate::linalg::matrix::Matrix;

mod transform;
use crate::transform::Transform;

fn main() {
    let input = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let target = Matrix::new(vec![vec![2.0, 3.0], vec![4.0, 5.0]]);

    let transform = Transform::new(input.cols, target.cols);

    let pred = transform.forward(&input);
    let err = &target - &pred;
    let out = transform.backward(&input, &err);

    println!("Input: {:?}", input);
    println!("Target: {:?}", target);
    println!("Prediction: {:?}", pred);
    println!("Error: {:?}", err);
    println!("Output: {:?}", out);
}
```

To run a demo:

```bash
cargo run
```

To run tests:

```bash
cargo test
```
