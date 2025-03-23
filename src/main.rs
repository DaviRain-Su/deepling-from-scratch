#![allow(dead_code)]

mod mnsit_loader;
use crate::mnsit_loader::*;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 加载 MNIST 数据集
    let mut mnist = MnistData::load("./mnist_data").map_err(|e| anyhow::anyhow!(e))?;

    // 预处理图像
    mnist.preprocess_images(false);

    mnist.print_summary();

    Ok(())
}
