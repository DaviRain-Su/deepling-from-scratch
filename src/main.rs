fn main() {
    let result = and(1.0, 1.0);
    println!("and(1.0, 1.0) - Result: {}", result);
    let result = and(1.0, 0.0);
    println!("and(1.0, 0.0) - Result: {}", result);
    let result = and(0.0, 1.0);
    println!("and(0.0, 1.0) - Result: {}", result);
    let result = and(0.0, 0.0);
    println!("and(0.0, 0.0) - Result: {}", result);
}

fn and(x1: f64, x2: f64) -> f64 {
    let (w1, w2, theta) = (0.5, 0.5, 0.7);
    let tmp = x1 * w1 + x2 * w2;
    if tmp <= theta { 0.0 } else { 1.0 }
}
