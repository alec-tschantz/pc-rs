# pc-rs

Predictive coding in Rust.

```rust
mod linalg;
use linalg::Matrix;

fn main() {
    let a = Matrix::ones(2, 2);
    let b = Matrix::ones(2, 2);
    let c = a.add(&b);
    let d = c.matmul(&b);
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