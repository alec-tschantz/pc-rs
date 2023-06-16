mod linalg;
use crate::linalg::matrix::Matrix;

fn main() {
    let a = Matrix::ones(2, 2);
    let b = Matrix::new(vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    let c = &a + &b;
    let d = &c * &b;

    println!("Matrix a: {:?}", a);
    println!("Matrix b: {:?}", b);
    println!("Matrix c: {:?}", c);
    println!("Matrix d: {:?}", d);
}
