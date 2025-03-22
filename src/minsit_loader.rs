//! MNIST dataset loader and preprocessing utilities
use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Constants for MNIST dataset
pub const MNIST_TRAIN_SIZE: usize = 60000;
pub const MNIST_TEST_SIZE: usize = 10000;
pub const MNIST_IMG_SIZE: usize = 784; // 28x28 pixels
pub const MNIST_NUM_CLASSES: usize = 10;

/// Structure representing the MNIST dataset
#[derive(Debug)]
pub struct MnistData {
    pub train_images: Vec<f32>, // Training images [60000 x 784]
    pub train_labels: Vec<f32>, // Training labels [60000]
    pub test_images: Vec<f32>,  // Test images [10000 x 784]
    pub test_labels: Vec<f32>,  // Test labels [10000]
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

    /// Load MNIST dataset from files at the specified path
    pub fn load<P: AsRef<Path>>(base_path: P) -> io::Result<Self> {
        let mut data = MnistData::new();

        // Load training data
        data.train_images = read_idx_file(
            base_path.as_ref().join("train-images-idx3-ubyte"),
            0x0803,
            MNIST_TRAIN_SIZE * MNIST_IMG_SIZE,
        )?;

        data.train_labels = read_idx_file(
            base_path.as_ref().join("train-labels-idx1-ubyte"),
            0x0801,
            MNIST_TRAIN_SIZE,
        )?;

        // Load test data
        data.test_images = read_idx_file(
            base_path.as_ref().join("t10k-images-idx3-ubyte"),
            0x0803,
            MNIST_TEST_SIZE * MNIST_IMG_SIZE,
        )?;

        data.test_labels = read_idx_file(
            base_path.as_ref().join("t10k-labels-idx1-ubyte"),
            0x0801,
            MNIST_TEST_SIZE,
        )?;

        Ok(data)
    }

    /// Create a mini-batch from training data
    pub fn create_batch(&self, batch_indices: &[usize], batch_size: usize) -> (Vec<f32>, Vec<f32>) {
        let mut batch_images = Vec::with_capacity(batch_size * self.image_size);
        let mut batch_labels = Vec::with_capacity(batch_size);

        for &idx in batch_indices.iter().take(batch_size) {
            let start = idx * self.image_size;
            let end = start + self.image_size;
            batch_images.extend_from_slice(&self.train_images[start..end]);
            batch_labels.push(self.train_labels[idx]);
        }

        (batch_images, batch_labels)
    }

    /// Print summary statistics of the dataset
    pub fn print_summary(&self) {
        println!("MNIST Dataset Summary:");
        println!("Training samples: {}", self.num_train);
        println!("Test samples: {}", self.num_test);
        println!("Image size: {} pixels", self.image_size);
        println!("Number of classes: {}", self.num_classes);
    }
}

/// Helper function to read IDX file format
fn read_idx_file<P: AsRef<Path>>(
    path: P,
    expected_magic: u32,
    data_size: usize,
) -> io::Result<Vec<f32>> {
    let mut file = File::open(path)?;

    // Read and verify magic number
    let magic = file.read_u32::<BigEndian>()?;
    if magic != expected_magic {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid magic number in IDX file",
        ));
    }

    // Read number of items
    let _num_items = file.read_u32::<BigEndian>()? as usize;

    // For images, read dimensions
    if expected_magic == 0x0803 {
        let _rows = file.read_u32::<BigEndian>()?;
        let _cols = file.read_u32::<BigEndian>()?;
    }

    // Read data
    let mut buffer = vec![0u8; data_size];
    file.read_exact(&mut buffer)?;

    // Convert to f32 and normalize to [0, 1]
    Ok(buffer.into_iter().map(|x| (x as f32 / 255.0)).collect())
}

/// Preprocess MNIST images (normalize, enhance contrast)
pub fn preprocess_images(images: &mut [f32]) {
    for image in images.chunks_mut(MNIST_IMG_SIZE) {
        // Calculate mean and standard deviation
        let mean = image.iter().sum::<f32>() / MNIST_IMG_SIZE as f32;
        let std_dev =
            (image.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / MNIST_IMG_SIZE as f32).sqrt();

        // Normalize
        if std_dev > 1e-6 {
            for pixel in image.iter_mut() {
                *pixel = (*pixel - mean) / std_dev;
            }
        }
    }
}

/// Display MNIST digit as ASCII art
pub fn display_digit(image: &[f32], label: u8) {
    println!("Label: {}", label);
    for row in 0..28 {
        for col in 0..28 {
            let pixel = image[row * 28 + col];
            let char = match pixel {
                x if x < 0.2 => ' ',
                x if x < 0.4 => '.',
                x if x < 0.6 => '*',
                x if x < 0.8 => '#',
                _ => '@',
            };
            print!("{}", char);
        }
        println!();
    }
}

/// Shuffle a slice in place
pub fn shuffle_slice<T>(slice: &mut [T]) {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    slice.shuffle(&mut rng);
}
