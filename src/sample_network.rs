use crate::utils::*;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Deserialize)]
struct ArrayData {
    shape: Vec<usize>,
    data: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct NetworkParams {
    #[serde(rename = "W1")]
    w1: ArrayData,
    b1: ArrayData,
    #[serde(rename = "W2")]
    w2: ArrayData,
    b2: ArrayData,
    #[serde(rename = "W3")]
    w3: ArrayData,
    b3: ArrayData,
}

#[derive(Debug)]
pub struct Network {
    pub w1: Array2<f32>,
    pub b1: Array2<f32>,
    pub w2: Array2<f32>,
    pub b2: Array2<f32>,
    pub w3: Array2<f32>,
    pub b3: Array2<f32>,
}

impl Network {
    pub fn init_network(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // 读取JSON文件
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let params: NetworkParams = serde_json::from_reader(reader)?;

        // 转换参数为Array2
        let w1 = convert_to_array2(&params.w1)?;
        let b1 = convert_to_array2(&params.b1)?;
        let w2 = convert_to_array2(&params.w2)?;
        let b2 = convert_to_array2(&params.b2)?;
        let w3 = convert_to_array2(&params.w3)?;
        let b3 = convert_to_array2(&params.b3)?;

        Ok(Network {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        })
    }

    pub fn predict(&self, x: &[f32]) -> Array2<f32> {
        // 将输入转换为列向量 (784, 1)
        let x = Array2::from_shape_vec((1, 784), x.to_vec()).unwrap();
        // 打印输入数据范围（调试用）
        if let (Some(&min), Some(&max)) = (
            x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
            x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
        ) {
            println!("Input range: {} to {}", min, max);
        }

        // W1: (784, 50), x: (1, 784)
        // X(1, 784) * W1(784,50) + B(1,50)(这是转置后的)
        let a1 = x.dot(&self.w1) + &self.b1.t();
        let z1 = sigmoid2(a1);

        // W2: (50, 100), z1: (1, 50)
        // Z1(1,50) * W2(50,100) + B(1,100)(这是转置后的)
        let a2 = z1.dot(&self.w2) + &self.b2.t();
        let z2 = sigmoid2(a2);

        // W3: (100, 10), z2: (1, 100)
        // Z2(1,100) * W3(100,10) + B(1,10)(这是转置后的)
        let a3 = z2.dot(&self.w3) + &self.b3.t();
        softmax2(a3)
    }

    pub fn predict_batch(&self, x: &Array2<f32>) -> Array2<f32> {
        // x的形状为 (batch_size, 784)
        let a1 = x.dot(&self.w1) + &self.b1.t();
        let z1 = sigmoid2(a1);

        let a2 = z1.dot(&self.w2) + &self.b2.t();
        let z2 = sigmoid2(a2);

        let a3 = z2.dot(&self.w3) + &self.b3.t();

        softmax_batch(&a3)
    }

    pub fn predict_class(&self, x: &[f32]) -> usize {
        let prediction = self.predict(x);

        // 打印预测结果（调试用）
        println!("Prediction values: {:?}", prediction);

        // 获取最大值的索引
        prediction
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    pub fn print_params_info(&self) {
        println!("Network Parameters Info:");
        println!("W1 shape: {:?}", self.w1.shape());
        println!("b1 shape: {:?}", self.b1.shape());
        println!("W2 shape: {:?}", self.w2.shape());
        println!("b2 shape: {:?}", self.b2.shape());
        println!("W3 shape: {:?}", self.w3.shape());
        println!("b3 shape: {:?}", self.b3.shape());
    }
}

fn convert_to_array2(array_data: &ArrayData) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut flattened = Vec::new();
    for row in &array_data.data {
        flattened.extend(row);
    }

    Ok(Array2::from_shape_vec(
        (array_data.shape[0], array_data.shape[1]),
        flattened,
    )?)
}

#[test]
fn test_network() {
    let network = Network::init_network("./data/sample_weight.json").unwrap();
    network.print_params_info();

    println!("\nNetwork loaded successfully!");
}
