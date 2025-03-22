#![allow(dead_code)]

mod minsit_loader;
use crate::minsit_loader::*;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
