use ndarray::Axis;
use ndarray::{Array1, Array2};

// x1*w1 + x2*w2 <> b
// x1*w1 + x2*w2 - b <> 0
// (w1, w2, theta) = (0.5, 0.5, -0.7)
pub fn and(x1: Array1<f64>, x2: Array1<f64>) -> Array1<f64> {
    let (w1, w2, b) = (0.5, 0.5, 0.7);
    let new_x1 = x1.mapv(|x| x * w1);
    let new_x2 = x2.mapv(|x| x * w2);
    (new_x1 + new_x2 - b).mapv(|x| if x <= 0.0 { 0.0 } else { 1.0 })
}

// x1*w1 + x2*w2 -b <> 0
// (w1, w2, theta) = (-0.5, -0.5, -0.7)
pub fn nand(x1: Array1<f64>, x2: Array1<f64>) -> Array1<f64> {
    let (w1, w2, b) = (-0.5, -0.5, -0.7);
    let new_x1 = x1.mapv(|x| x * w1);
    let new_x2 = x2.mapv(|x| x * w2);
    (new_x1 + new_x2 - b).mapv(|x| if x <= 0.0 { 0.0 } else { 1.0 })
}

// (w1, w2, theta) = (0.5, 0.5, 0.2)
pub fn or(x1: Array1<f64>, x2: Array1<f64>) -> Array1<f64> {
    let (w1, w2, b) = (0.5, 0.5, 0.2);
    let new_x1 = x1.mapv(|x| x * w1);
    let new_x2 = x2.mapv(|x| x * w2);
    (new_x1 + new_x2 - b).mapv(|x| if x <= 0.0 { 0.0 } else { 1.0 })
}

pub fn xor(x1: Array1<f64>, x2: Array1<f64>) -> Array1<f64> {
    let s1 = nand(x1.clone(), x2.clone());
    let s2 = or(x1.clone(), x2.clone());
    and(s1, s2)
}

pub fn step_function(x: Array1<f32>) -> Array1<f32> {
    // y = x > 0
    x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

pub fn sigmoid(x: Array1<f32>) -> Array1<f32> {
    x.mapv(|element| 1.0 / (1.0 + (-element).exp()))
}

// sigmoid function
pub fn sigmoid2(x: Array2<f32>) -> Array2<f32> {
    x.mapv(|element| 1.0 / (1.0 + (-element).exp()))
}

pub fn softmax(x: Array1<f32>) -> Array1<f32> {
    let exp_x = x.mapv(|element| element.exp());
    let sum_exp_x = exp_x.sum_axis(Axis(1));
    exp_x / sum_exp_x
}

pub fn softmax2(x: Array2<f32>) -> Array2<f32> {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.map(|&v| ((v - max_val) as f64).exp() as f32);
    let sum_exp = exp_x.sum();
    exp_x.map(|&v| v / sum_exp)
}

// 批处理版本的softmax函数
pub fn softmax_batch(x: &Array2<f32>) -> Array2<f32> {
    let mut result = x.clone();

    // 对每一行（每个样本）分别计算softmax
    for mut row in result.rows_mut() {
        let max_val = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x: Array1<f32> = row.map(|&v| ((v - max_val) as f64).exp() as f32);
        let sum_exp = exp_x.sum();
        row.assign(&exp_x.map(|&v| v / sum_exp));
    }

    result
}

// relu function (Rectified linear function)
pub fn relu(x: Array1<f32>) -> Array1<f32> {
    x.mapv(|element| if element > 0.0 { element } else { 0.0 })
}

pub fn argmax(x: &Array1<f32>) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

pub fn argmax2(x: &Array2<f32>) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_step_function() {
        let x = Array1::from(vec![0.0, 1.0, 2.0]);
        let y = step_function(x);
        assert_eq!(y, Array1::from(vec![0.0, 1.0, 1.0]));
    }

    #[test]
    fn test_and() {
        let x1 = Array1::from(vec![0.0, 1.0, 0.0, 1.0]);
        let x2 = Array1::from(vec![0.0, 0.0, 1.0, 1.0]);
        let y = and(x1, x2);
        assert_eq!(y, Array1::from(vec![0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_nand() {
        let x1 = Array1::from(vec![0.0, 1.0, 0.0, 1.0]);
        let x2 = Array1::from(vec![0.0, 0.0, 1.0, 1.0]);
        let y = nand(x1, x2);
        assert_eq!(y, Array1::from(vec![1.0, 1.0, 1.0, 0.0]));
    }

    #[test]
    fn test_or() {
        let x1 = Array1::from(vec![0.0, 1.0, 0.0, 1.0]);
        let x2 = Array1::from(vec![0.0, 0.0, 1.0, 1.0]);
        let y = or(x1, x2);
        assert_eq!(y, Array1::from(vec![0.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_xor() {
        let x1 = Array1::from(vec![0.0, 1.0, 0.0, 1.0]);
        let x2 = Array1::from(vec![0.0, 0.0, 1.0, 1.0]);
        let y = xor(x1.clone(), x2.clone());
        assert_eq!(y, Array1::from(vec![0.0, 1.0, 1.0, 0.0]));
    }
}
