use ndarray::Array1;

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

pub fn step_function(x: Array1<f64>) -> Array1<f64> {
    // y = x > 0
    x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

// sigmoid function
pub fn sigmoid(x: Array1<f64>) -> Array1<f64> {
    x.mapv(|element| 1.0 / (1.0 + (-element).exp()))
}

// relu function (Rectified linear function)
pub fn relu(x: Array1<f64>) -> Array1<f64> {
    x.mapv(|element| if element > 0.0 { element } else { 0.0 })
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
