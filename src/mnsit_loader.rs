use byteorder::{BigEndian, ReadBytesExt};
use rand::seq::SliceRandom;
use std::fmt::Display;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// Constants for MNIST dataset
pub const MNIST_TRAIN_SIZE: usize = 60000;
pub const MNIST_TEST_SIZE: usize = 10000;
pub const MNIST_IMG_SIZE: usize = 784; // 28x28 pixels
pub const MNIST_NUM_CLASSES: usize = 10;

/// Custom error type for MNIST operations
#[derive(Debug, thiserror::Error)]
pub enum MnistError {
    Io(io::Error),
    InvalidMagicNumber { expected: u32, got: u32 },
    InvalidData(String),
}

impl Display for MnistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MnistError::Io(err) => write!(f, "I/O error: {}", err),
            MnistError::InvalidMagicNumber { expected, got } => {
                write!(
                    f,
                    "Invalid magic number: expected {}, got {}",
                    expected, got
                )
            }
            MnistError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl From<io::Error> for MnistError {
    fn from(err: io::Error) -> MnistError {
        MnistError::Io(err)
    }
}

/// MNIST dataset structure
#[derive(Debug)]
pub struct MnistData {
    pub train_images: Vec<f32>, // Training images [60000 x 784]
    pub train_labels: Vec<u8>,  // Training labels [60000]
    pub test_images: Vec<f32>,  // Test images [10000 x 784]
    pub test_labels: Vec<u8>,   // Test labels [10000]
    pub num_train: usize,       // Number of training samples
    pub num_test: usize,        // Number of test samples
    pub image_size: usize,      // Size of each image (pixels)
    pub num_classes: usize,     // Number of classes
}

impl MnistData {
    /// Create a new empty MNIST dataset
    pub fn new() -> Self {
        MnistData {
            train_images: Vec::new(),
            train_labels: Vec::new(),
            test_images: Vec::new(),
            test_labels: Vec::new(),
            num_train: MNIST_TRAIN_SIZE,
            num_test: MNIST_TEST_SIZE,
            image_size: MNIST_IMG_SIZE,
            num_classes: MNIST_NUM_CLASSES,
        }
    }

    /// Load MNIST dataset from files
    pub fn load<P: AsRef<Path>>(base_path: P) -> Result<Self, MnistError> {
        let base_path = base_path.as_ref();
        let mut data = MnistData::new();

        // Allocate memory
        data.train_images = vec![0.0; data.num_train * data.image_size];
        data.test_images = vec![0.0; data.num_test * data.image_size];
        data.train_labels = vec![0; data.num_train];
        data.test_labels = vec![0; data.num_test];

        // Load training data
        let train_images_path = base_path.join("train-images-idx3-ubyte");
        let train_labels_path = base_path.join("train-labels-idx1-ubyte");
        let test_images_path = base_path.join("t10k-images-idx3-ubyte");
        let test_labels_path = base_path.join("t10k-labels-idx1-ubyte");

        // Load raw data
        let mut train_images_raw = vec![0u8; data.num_train * data.image_size];
        let mut test_images_raw = vec![0u8; data.num_test * data.image_size];

        read_idx_file(&train_images_path, &mut train_images_raw, 0x00000803)?;
        read_idx_file(&train_labels_path, &mut data.train_labels, 0x00000801)?;
        read_idx_file(&test_images_path, &mut test_images_raw, 0x00000803)?;
        read_idx_file(&test_labels_path, &mut data.test_labels, 0x00000801)?;

        // Convert raw image data to f32 and normalize
        for (i, &pixel) in train_images_raw.iter().enumerate() {
            data.train_images[i] = pixel as f32 / 255.0;
        }
        for (i, &pixel) in test_images_raw.iter().enumerate() {
            data.test_images[i] = pixel as f32 / 255.0;
        }

        // Apply preprocessing
        data.preprocess_images(true);

        Ok(data)
    }

    /// Preprocess MNIST images
    pub fn preprocess_images(&mut self, _use_neon: bool) {
        self.preprocess_images_standard();
    }

    fn preprocess_images_standard(&mut self) {
        for chunk in self.train_images.chunks_mut(self.image_size) {
            Self::normalize_image_chunk(chunk);
        }
        for chunk in self.test_images.chunks_mut(self.image_size) {
            Self::normalize_image_chunk(chunk);
        }
    }

    fn normalize_image_chunk(chunk: &mut [f32]) {
        // 1. 计算均值和最大最小值
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mean = {
            let mut sum = 0.0;
            for &x in chunk.iter() {
                sum += x;
                min_val = min_val.min(x);
                max_val = max_val.max(x);
            }
            sum / chunk.len() as f32
        };

        // 2. 中心化数据并计算方差
        let variance = {
            let mut sum_sq = 0.0;
            for x in chunk.iter_mut() {
                *x -= mean;
                sum_sq += *x * *x;
            }
            sum_sq / chunk.len() as f32
        };

        // 3. 标准化 (添加小的epsilon值防止除零)
        let std_dev = (variance + 1e-8).sqrt();
        if std_dev > 1e-8 {
            for x in chunk.iter_mut() {
                *x /= std_dev;
            }
        }

        // 4. 重新缩放到 [0, 1] 范围
        let range = max_val - min_val;
        if range > 1e-8 {
            let scale = 1.0 / range;
            for x in chunk.iter_mut() {
                *x = (*x - min_val) * scale;
            }
        }
    }

    /// Create a mini-batch from training data
    pub fn create_batch(
        &self,
        batch_indices: &[usize],
        batch_size: usize,
    ) -> (Vec<Vec<f32>>, Vec<u8>) {
        let mut batch_images = Vec::with_capacity(batch_size);
        let mut batch_labels = Vec::with_capacity(batch_size);

        for &idx in batch_indices.iter().take(batch_size) {
            let start = idx * self.image_size;
            let end = start + self.image_size;
            batch_images.push(self.train_images[start..end].to_vec());
            batch_labels.push(self.train_labels[idx]);
        }

        (batch_images, batch_labels)
    }

    /// Display summary statistics
    pub fn print_summary(&self) {
        println!("MNIST Dataset Summary");
        println!("---------------------");
        println!("Training samples: {}", self.num_train);
        println!("Test samples: {}", self.num_test);
        println!("Image size: {} pixels", self.image_size);
        println!("Number of classes: {}", self.num_classes);

        // Count samples per class
        let mut train_counts = [0; 10];
        let mut test_counts = [0; 10];

        for &label in &self.train_labels {
            train_counts[label as usize] += 1;
        }
        for &label in &self.test_labels {
            test_counts[label as usize] += 1;
        }

        println!("\nClass distribution:");
        println!("Digit   Training    Test");
        for i in 0..10 {
            println!("  {}     {:5}     {:4}", i, train_counts[i], test_counts[i]);
        }

        println!("\nSample digit:");
        self.display_digit(0);
    }

    /// Display a digit as ASCII art
    pub fn display_digit(&self, index: usize) {
        if index >= self.num_train {
            println!("Invalid index!");
            return;
        }

        let start = index * self.image_size;
        let end = start + self.image_size;
        let image = &self.train_images[start..end];
        let label = self.train_labels[index];

        println!("Digit: {}", label);
        for i in 0..28 {
            for j in 0..28 {
                let pixel = image[i * 28 + j];
                let symbol = match pixel {
                    x if x < 0.2 => "  ",
                    x if x < 0.5 => "· ",
                    x if x < 0.8 => "o ",
                    _ => "@ ",
                };
                print!("{}", symbol);
            }
            println!();
        }
    }
}

/// Read IDX file format
fn read_idx_file<P: AsRef<Path>>(
    path: P,
    data: &mut [u8],
    magic_expected: u32,
) -> Result<(), MnistError> {
    let mut file = BufReader::new(File::open(path)?);

    let magic = file.read_u32::<BigEndian>()?;
    if magic != magic_expected {
        return Err(MnistError::InvalidMagicNumber {
            expected: magic_expected,
            got: magic,
        });
    }

    let dim_count = if magic_expected == 0x00000803 { 3 } else { 1 };
    for _ in 0..dim_count {
        let _ = file.read_u32::<BigEndian>()?;
    }

    file.read_exact(data)?;
    Ok(())
}

/// Shuffle indices
pub fn shuffle_indices(indices: &mut [usize]) {
    let mut rng = rand::rng();
    indices.shuffle(&mut rng);
}
