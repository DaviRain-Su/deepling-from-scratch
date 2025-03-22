mod minsit_loader;
use crate::minsit_loader::*;

use ndarray::Array1;
fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let result = nand(1.0, 1.0);
    println!("nand(1.0, 1.0) - Result: {}", result);
    let result = nand(1.0, 0.0);
    println!("nand(1.0, 0.0) - Result: {}", result);
    let result = nand(0.0, 1.0);
    println!("nand(0.0, 1.0) - Result: {}", result);
    let result = nand(0.0, 0.0);
    println!("nand(0.0, 0.0) - Result: {}", result);

    let result = or(1.0, 1.0);
    println!("or(1.0, 1.0) - Result: {}", result);
    let result = or(1.0, 0.0);
    println!("or(1.0, 0.0) - Result: {}", result);
    let result = or(0.0, 1.0);
    println!("or(0.0, 1.0) - Result: {}", result);
    let result = or(0.0, 0.0);
    println!("or(0.0, 0.0) - Result: {}", result);

    let result = xor(1.0, 1.0);
    println!("xor(1.0, 1.0) - Result: {}", result);
    let result = xor(1.0, 0.0);
    println!("xor(1.0, 0.0) - Result: {}", result);
    let result = xor(0.0, 1.0);
    println!("xor(0.0, 1.0) - Result: {}", result);
    let result = xor(0.0, 0.0);
    println!("xor(0.0, 0.0) - Result: {}", result);

    let result = step_function(Array1::from(vec![1.0, 2.0]));
    println!("step_function([1.0, 0.0]) - Result: {:?}", result);

    let result = sigmoid(Array1::from(vec![-1.0, 1.0, 2.0]));
    println!("sigmoid([-1.0, 1.0, 2.0]) - Result: {:?}", result);

    let result = relu(Array1::from(vec![-1.0, 1.0, 2.0]));
    println!("relu([-1.0, 1.0, 2.0]) - Result: {:?}", result);

    // 加载 MNIST 数据集
    let mut mnist = MnistData::load("./mnist_data")?;

    // 预处理图像
    preprocess_images(&mut mnist.train_images);
    preprocess_images(&mut mnist.test_images);

    // 创建一个 mini-batch
    let batch_indices: Vec<usize> = (0..32).collect();
    let (batch_images, batch_labels) = mnist.create_batch(&batch_indices, 32);

    mnist.print_summary();

    // 显示第一个数字
    display_digit(&batch_images[0..MNIST_IMG_SIZE], batch_labels[0] as u8);

    Ok(())
}

fn and(x1: f64, x2: f64) -> f64 {
    let (w1, w2, b) = (0.5, 0.5, -0.7);
    let tmp = x1 * w1 + x2 * w2 + b;
    if tmp <= 0.0 { 0.0 } else { 1.0 }
}

// (w1, w2, theta) = (0.5, 0.5, 0.7)
// use ndarray impl and function
fn and_ndarray(x1: f64, x2: f64) -> f64 {
    let x = Array1::from(vec![x1, x2]);
    let w = Array1::from(vec![0.5, 0.5]);
    let b = -0.7;
    let tmp = x.dot(&w) + b;
    //dbg!(tmp);
    if tmp <= 0.0 { 0.0 } else { 1.0 }
}

// (w1, w2, theta) = (-0.5, -0.5, -0.7)
fn nand(x1: f64, x2: f64) -> f64 {
    let x = Array1::from(vec![x1, x2]);
    let w = Array1::from(vec![-0.5, -0.5]);
    let b = 0.7;
    let tmp = x.dot(&w) + b;
    //dbg!(tmp);
    if tmp <= 0.0 { 0.0 } else { 1.0 }
}

// (w1, w2, theta) = (0.5, 0.5, 0.2)
fn or(x1: f64, x2: f64) -> f64 {
    let x = Array1::from(vec![x1, x2]);
    let w = Array1::from(vec![0.5, 0.5]);
    let b = -0.2;
    let tmp = x.dot(&w) + b;
    //dbg!(tmp);
    if tmp <= 0.0 { 0.0 } else { 1.0 }
}

fn xor(x1: f64, x2: f64) -> f64 {
    let s1 = nand(x1, x2);
    let s2 = or(x1, x2);
    and(s1, s2)
}

fn step_function(x: Array1<f64>) -> Array1<f64> {
    // y = x > 0
    x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

// sigmoid function
fn sigmoid(x: Array1<f64>) -> Array1<f64> {
    x.mapv(|element| 1.0 / (1.0 + (-element).exp()))
}

// relu function (Rectified linear function)
fn relu(x: Array1<f64>) -> Array1<f64> {
    x.mapv(|element| if element > 0.0 { element } else { 0.0 })
}
