#![allow(dead_code)]

mod mnsit_loader;
use crate::mnsit_loader::*;
mod sample_network;
use crate::sample_network::Network;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 加载 MNIST 数据集
    println!("Loading MNIST dataset...");
    let mut mnist = MnistData::load("./mnist_data")?;

    // 预处理图像
    println!("Preprocessing images...");
    mnist.preprocess_images(false);

    // 验证预处理后的数据范围
    if let Some(first_image) = mnist.test_images.chunks(mnist.image_size).next() {
        let min_val = first_image.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = first_image.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        println!("Image value range: {} to {}", min_val, max_val);
    }

    mnist.print_summary();

    // 加载网络
    println!("Loading network...");
    let network = Network::init_network("./data/sample_weight.json")?;
    network.print_params_info();

    // 评估模型
    println!("\nEvaluating model...");
    let mut accuracy_cnt = 0;

    // 测试前几个样本并打印详细信息
    for i in 0..10 {
        let start_idx = i * mnist.image_size;
        let end_idx = start_idx + mnist.image_size;
        let image = &mnist.test_images[start_idx..end_idx];
        let label = mnist.test_labels[i];

        let predicted_class = network.predict_class(image);
        println!(
            "Sample {}: True label = {}, Predicted = {}",
            i, label, predicted_class
        );

        if predicted_class == label as usize {
            accuracy_cnt += 1;
        }
    }

    // 遍历每个测试样本
    for i in 10..mnist.num_test {
        // 获取单个图像的切片（784个像素）
        let start_idx = i * mnist.image_size;
        let end_idx = start_idx + mnist.image_size;
        let image = &mnist.test_images[start_idx..end_idx];
        let label = mnist.test_labels[i];

        // 预测
        let predicted_class = network.predict_class(image);
        if predicted_class == label as usize {
            accuracy_cnt += 1;
        }

        // 显示进度
        if (i + 1) % 100 == 0 {
            println!("Processing {}/{}", i + 1, mnist.num_test);
        }
    }

    // 计算并显示准确率
    let accuracy = accuracy_cnt as f64 / mnist.num_test as f64;
    println!("\nTest Results:");
    println!("Total samples: {}", mnist.num_test);
    println!("Correct predictions: {}", accuracy_cnt);
    println!("Accuracy: {:.2}%", accuracy * 100.0);
    Ok(())
}
