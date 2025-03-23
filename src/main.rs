#![allow(dead_code)]

mod mnsit_loader;
use crate::mnsit_loader::*;
mod sample_network;
use crate::sample_network::Network;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 加载 MNIST 数据集
    let mut mnist = MnistData::load("./mnist_data").map_err(|e| anyhow::anyhow!(e))?;

    // 预处理图像
    mnist.preprocess_images(false);

    mnist.print_summary();

    let network = Network::init_network("./data/sample_weight.json")?;
    network.print_params_info();

    println!("\nNetwork loaded successfully!");

    Ok(())
}
