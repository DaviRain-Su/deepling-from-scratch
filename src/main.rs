#![allow(dead_code)]

use ndarray::{Array1, Array2, s};
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
    let batch_size = 100;
    let mut accuracy_cnt = 0;

    // 计算总批次数
    let num_batches = mnist.num_test / batch_size;

    // 按批次处理数据
    for batch_idx in 0..num_batches {
        // 创建一个存储整个批次数据的矩阵 (batch_size, 784)
        let mut batch_data = Array2::zeros((batch_size, 784));
        let batch_start = batch_idx * batch_size;

        // 填充批次数据
        for i in 0..batch_size {
            let img_idx = batch_start + i;
            let img_start = img_idx * mnist.image_size;
            let img_end = img_start + mnist.image_size;

            // 将图像数据复制到批次矩阵中
            let image_slice = &mnist.test_images[img_start..img_end];
            batch_data
                .slice_mut(s![i, ..])
                .assign(&Array1::from_vec(image_slice.to_vec()));
        }

        // 对整个批次进行预测
        let predictions = network.predict_batch(&batch_data);

        // 处理预测结果
        for i in 0..batch_size {
            let img_idx = batch_start + i;

            // 获取预测类别（找到概率最大的索引）
            let predicted_class = predictions
                .slice(s![i, ..])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            // 检查预测是否正确
            if predicted_class == mnist.test_labels[img_idx] as usize {
                accuracy_cnt += 1;
            }
        }

        println!("Processing batch {}/{}", batch_idx + 1, num_batches);
    }

    // 处理剩余的样本（如果有的话）
    let remaining = mnist.num_test % batch_size;
    if remaining > 0 {
        let mut batch_data = Array2::zeros((remaining, 784));
        let start_idx = (mnist.num_test - remaining) * mnist.image_size;

        for i in 0..remaining {
            let img_start = start_idx + i * mnist.image_size;
            let img_end = img_start + mnist.image_size;
            let image_slice = &mnist.test_images[img_start..img_end];
            batch_data
                .slice_mut(s![i, ..])
                .assign(&Array1::from_vec(image_slice.to_vec()));
        }

        let predictions = network.predict_batch(&batch_data);

        for i in 0..remaining {
            let img_idx = mnist.num_test - remaining + i;
            let predicted_class = predictions
                .slice(s![i, ..])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if predicted_class == mnist.test_labels[img_idx] as usize {
                accuracy_cnt += 1;
            }
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
