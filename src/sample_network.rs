use ndarray::Array2;
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
