//! MNIST dataset loader and preprocessing utilities
use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, BufReader, Read};
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
            MNIST_TRAIN_SIZE * MNIST_IMG_SIZE,
            0x00000803,
        )?;

        data.train_labels = read_idx_file(
            base_path.as_ref().join("train-labels-idx1-ubyte"),
            MNIST_TRAIN_SIZE,
            0x00000801,
        )?;

        // Load test data
        data.test_images = read_idx_file(
            base_path.as_ref().join("t10k-images-idx3-ubyte"),
            MNIST_TEST_SIZE * MNIST_IMG_SIZE,
            0x00000803,
        )?;

        data.test_labels = read_idx_file(
            base_path.as_ref().join("t10k-labels-idx1-ubyte"),
            MNIST_TEST_SIZE,
            0x00000801,
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

/// Read data from IDX file format
fn inner_sread_idx_file<P: AsRef<Path>>(
    filename: P,
    data: &mut [u8],
    size: usize,
    magic_expected: u32,
) -> io::Result<()> {
    // Open file
    let file = File::open(filename.as_ref())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to open file: {}", e)))?;

    let mut reader = BufReader::new(file);

    // Read magic number
    let magic = reader.read_u32::<BigEndian>().map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to read magic number: {}", e),
        )
    })?;

    if magic != magic_expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Invalid magic number (expected 0x{:08x}, got 0x{:08x})",
                magic_expected, magic
            ),
        ));
    }

    // Read dimensions
    let dim_count = if magic_expected == 0x00000803 { 3 } else { 1 };
    let mut dimensions = vec![0u32; dim_count];

    for dim in dimensions.iter_mut() {
        *dim = reader.read_u32::<BigEndian>().map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to read dimensions: {}", e),
            )
        })?;
    }

    // Read the actual data
    reader
        .read_exact(&mut data[..size])
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to read data: {}", e)))?;

    Ok(())
}

/// Helper function to read IDX file and return a Vec<f32>
fn read_idx_file<P: AsRef<Path>>(
    filename: P,
    size: usize,
    magic_expected: u32,
) -> io::Result<Vec<f32>> {
    let mut buffer = vec![0u8; size];
    inner_sread_idx_file(filename, &mut buffer, size, magic_expected)?;

    // Convert u8 to normalized f32
    Ok(buffer.into_iter().map(|x| x as f32 / 255.0).collect())
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
