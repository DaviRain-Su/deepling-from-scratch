use ndarray::Array1;
fn main() {
    let result = and(1.0, 1.0);
    println!("and(1.0, 1.0) - Result: {}", result);
    let result = and(1.0, 0.0);
    println!("and(1.0, 0.0) - Result: {}", result);
    let result = and(0.0, 1.0);
    println!("and(0.0, 1.0) - Result: {}", result);
    let result = and(0.0, 0.0);
    println!("and(0.0, 0.0) - Result: {}", result);

    let result = and_ndarray(1.0, 1.0);
    println!("and_ndarray(1.0, 1.0) - Result: {}", result);
    let result = and_ndarray(1.0, 0.0);
    println!("and_ndarray(1.0, 0.0) - Result: {}", result);
    let result = and_ndarray(0.0, 1.0);
    println!("and_ndarray(0.0, 1.0) - Result: {}", result);
    let result = and_ndarray(0.0, 0.0);
    println!("and_ndarray(0.0, 0.0) - Result: {}", result);
}

fn and(x1: f64, x2: f64) -> f64 {
    let (w1, w2, b) = (0.5, 0.5, -0.7);
    let tmp = x1 * w1 + x2 * w2 + b;
    if tmp <= 0.0 { 0.0 } else { 1.0 }
}

// use ndarray impl and function
fn and_ndarray(x1: f64, x2: f64) -> f64 {
    let x = Array1::from(vec![x1, x2]);
    let w = Array1::from(vec![0.5, 0.5]);
    let b = -0.7;
    let tmp = x.dot(&w) + b;
    dbg!(tmp);
    if tmp <= 0.0 { 0.0 } else { 1.0 }
}
