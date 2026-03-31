/**
 * ============================================================================
 * GridMind - Synthetic Data Generator
 * ============================================================================
 * 
 * This module generates synthetic MNIST-like data for federated learning
 * demonstrations. Each client generates its own local "private" dataset,
 * simulating the federated learning scenario where data stays on-device.
 * 
 * DATA FORMAT:
 * - Images: 28x28 grayscale (values 0-1), shape [N, 28, 28, 1]
 * - Labels: One-hot encoded, shape [N, 10]
 * 
 * SYNTHETIC DATA STRATEGY:
 * We generate simple geometric patterns for each digit class:
 * - 0: Circle/ring pattern
 * - 1: Vertical line
 * - 2: Two horizontal strokes
 * - 3: Three horizontal strokes
 * - 4: Cross pattern (vertical + horizontal)
 * - 5: Square pattern
 * - 6: Circle with dot
 * - 7: Diagonal line
 * - 8: Double circle
 * - 9: Circle with vertical line
 * 
 * Each pattern has random variations (position, size, noise) to simulate
 * real data variability and ensure the model learns features.
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const DATA_CONFIG = {
  imageSize: 28,
  numClasses: 10,
  noiseLevel: 0.1,    // Random noise added to images
  strokeWidth: 2,     // Width of drawn strokes
};

// =============================================================================
// PATTERN GENERATORS
// =============================================================================

/**
 * Draw a circle pattern (digit 0)
 * @param {Float32Array} pixels - Pixel array to draw on
 * @param {number} cx - Center X
 * @param {number} cy - Center Y
 * @param {number} radius - Circle radius
 */
function drawCircle(pixels, cx, cy, radius) {
  const size = DATA_CONFIG.imageSize;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      if (Math.abs(dist - radius) < DATA_CONFIG.strokeWidth) {
        pixels[y * size + x] = 1.0;
      }
    }
  }
}

/**
 * Draw a filled circle
 * @param {Float32Array} pixels - Pixel array to draw on
 * @param {number} cx - Center X
 * @param {number} cy - Center Y
 * @param {number} radius - Circle radius
 */
function drawFilledCircle(pixels, cx, cy, radius) {
  const size = DATA_CONFIG.imageSize;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      if (dist <= radius) {
        pixels[y * size + x] = 1.0;
      }
    }
  }
}

/**
 * Draw a vertical line (digit 1)
 * @param {Float32Array} pixels - Pixel array
 * @param {number} x - X position
 * @param {number} y1 - Start Y
 * @param {number} y2 - End Y
 */
function drawVerticalLine(pixels, x, y1, y2) {
  const size = DATA_CONFIG.imageSize;
  const width = DATA_CONFIG.strokeWidth;
  for (let y = Math.min(y1, y2); y <= Math.max(y1, y2); y++) {
    for (let dx = -width; dx <= width; dx++) {
      const px = x + dx;
      if (px >= 0 && px < size && y >= 0 && y < size) {
        pixels[y * size + px] = 1.0;
      }
    }
  }
}

/**
 * Draw a horizontal line
 * @param {Float32Array} pixels - Pixel array
 * @param {number} y - Y position
 * @param {number} x1 - Start X
 * @param {number} x2 - End X
 */
function drawHorizontalLine(pixels, y, x1, x2) {
  const size = DATA_CONFIG.imageSize;
  const width = DATA_CONFIG.strokeWidth;
  for (let x = Math.min(x1, x2); x <= Math.max(x1, x2); x++) {
    for (let dy = -width; dy <= width; dy++) {
      const py = y + dy;
      if (x >= 0 && x < size && py >= 0 && py < size) {
        pixels[py * size + x] = 1.0;
      }
    }
  }
}

/**
 * Draw a diagonal line (digit 7)
 * @param {Float32Array} pixels - Pixel array
 * @param {number} x1 - Start X
 * @param {number} y1 - Start Y
 * @param {number} x2 - End X
 * @param {number} y2 - End Y
 */
function drawDiagonalLine(pixels, x1, y1, x2, y2) {
  const size = DATA_CONFIG.imageSize;
  const width = DATA_CONFIG.strokeWidth;
  const steps = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1));
  
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = Math.round(x1 + (x2 - x1) * t);
    const y = Math.round(y1 + (y2 - y1) * t);
    
    for (let dx = -width; dx <= width; dx++) {
      for (let dy = -width; dy <= width; dy++) {
        const px = x + dx;
        const py = y + dy;
        if (px >= 0 && px < size && py >= 0 && py < size) {
          pixels[py * size + px] = 1.0;
        }
      }
    }
  }
}

/**
 * Add random noise to image
 * @param {Float32Array} pixels - Pixel array
 * @param {number} level - Noise level (0-1)
 */
function addNoise(pixels, level) {
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = Math.max(0, Math.min(1, pixels[i] + (Math.random() - 0.5) * level));
  }
}

/**
 * Apply slight random translation and scaling
 * @param {Float32Array} pixels - Original pixels
 * @returns {Float32Array} Transformed pixels
 */
function applyRandomTransform(pixels) {
  const size = DATA_CONFIG.imageSize;
  const transformed = new Float32Array(size * size);
  
  // Random translation (-2 to +2 pixels)
  const tx = Math.floor(Math.random() * 5) - 2;
  const ty = Math.floor(Math.random() * 5) - 2;
  
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const srcX = x - tx;
      const srcY = y - ty;
      
      if (srcX >= 0 && srcX < size && srcY >= 0 && srcY < size) {
        transformed[y * size + x] = pixels[srcY * size + srcX];
      }
    }
  }
  
  return transformed;
}

// =============================================================================
// DIGIT PATTERN GENERATORS
// =============================================================================

/**
 * Generate a synthetic image for a specific digit
 * 
 * @param {number} digit - Digit class (0-9)
 * @returns {Float32Array} 28x28 grayscale image
 */
function generateDigitImage(digit) {
  const size = DATA_CONFIG.imageSize;
  let pixels = new Float32Array(size * size);
  
  // Random center and size variations
  const cx = 14 + (Math.random() - 0.5) * 4;
  const cy = 14 + (Math.random() - 0.5) * 4;
  const scale = 0.8 + Math.random() * 0.4;
  
  switch (digit) {
    case 0: // Circle
      drawCircle(pixels, cx, cy, 8 * scale);
      break;
      
    case 1: // Vertical line
      drawVerticalLine(pixels, cx, 4, 24);
      break;
      
    case 2: // Two horizontal lines
      drawHorizontalLine(pixels, 8, 6, 22);
      drawHorizontalLine(pixels, 20, 6, 22);
      drawDiagonalLine(pixels, 22, 8, 6, 20);
      break;
      
    case 3: // Three horizontal lines
      drawHorizontalLine(pixels, 6, 8, 20);
      drawHorizontalLine(pixels, 14, 8, 20);
      drawHorizontalLine(pixels, 22, 8, 20);
      drawVerticalLine(pixels, 20, 6, 22);
      break;
      
    case 4: // Cross pattern
      drawVerticalLine(pixels, 18, 4, 24);
      drawHorizontalLine(pixels, 16, 6, 22);
      drawVerticalLine(pixels, 8, 4, 16);
      break;
      
    case 5: // Square-ish (S shape)
      drawHorizontalLine(pixels, 5, 6, 20);
      drawVerticalLine(pixels, 6, 5, 14);
      drawHorizontalLine(pixels, 14, 6, 20);
      drawVerticalLine(pixels, 20, 14, 23);
      drawHorizontalLine(pixels, 23, 6, 20);
      break;
      
    case 6: // Circle with stem
      drawCircle(pixels, 14, 16, 7 * scale);
      drawVerticalLine(pixels, 7, 4, 16);
      break;
      
    case 7: // Diagonal line with top
      drawHorizontalLine(pixels, 5, 6, 22);
      drawDiagonalLine(pixels, 22, 5, 10, 24);
      break;
      
    case 8: // Double circle (figure 8)
      drawCircle(pixels, 14, 8, 5 * scale);
      drawCircle(pixels, 14, 20, 5 * scale);
      break;
      
    case 9: // Circle with vertical line
      drawCircle(pixels, 14, 10, 6 * scale);
      drawVerticalLine(pixels, 20, 10, 24);
      break;
  }
  
  // Apply random transformation
  pixels = applyRandomTransform(pixels);
  
  // Add noise
  addNoise(pixels, DATA_CONFIG.noiseLevel);
  
  return pixels;
}

// =============================================================================
// BATCH DATA GENERATION
// =============================================================================

/**
 * Generate a batch of synthetic training data
 * 
 * TENSOR CREATION NOTES:
 * - tf.tensor4d() creates a 4D tensor [batch, height, width, channels]
 * - tf.oneHot() converts integer labels to one-hot vectors
 * - Data is wrapped in tf.tidy() to prevent memory leaks
 * 
 * @param {number} batchSize - Number of samples to generate
 * @param {number[]} classDistribution - Optional: which classes to include (for non-IID simulation)
 * @returns {Object} { xs: tf.Tensor4D, ys: tf.Tensor2D }
 */
function generateBatch(batchSize, classDistribution = null) {
  console.log(`[DataGenerator] Generating batch of ${batchSize} samples...`);
  
  // Determine which classes this client has access to
  const availableClasses = classDistribution || [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  
  // Generate images and labels
  const images = [];
  const labels = [];
  
  for (let i = 0; i < batchSize; i++) {
    // Randomly select a class from available classes
    const digit = availableClasses[Math.floor(Math.random() * availableClasses.length)];
    
    // Generate the image
    const imageData = generateDigitImage(digit);
    images.push(imageData);
    labels.push(digit);
  }
  
  // Convert to tensors inside tf.tidy() to manage memory
  return tf.tidy(() => {
    // -------------------------------------------------------------------------
    // TENSOR CREATION: Convert raw arrays to TF.js tensors
    // 
    // Image tensor shape: [batchSize, 28, 28, 1]
    // - batchSize: number of images
    // - 28, 28: image dimensions
    // - 1: single channel (grayscale)
    // -------------------------------------------------------------------------
    const size = DATA_CONFIG.imageSize;
    const imageBuffer = new Float32Array(batchSize * size * size);
    
    for (let i = 0; i < batchSize; i++) {
      imageBuffer.set(images[i], i * size * size);
    }
    
    const xs = tf.tensor4d(imageBuffer, [batchSize, size, size, 1]);
    
    // -------------------------------------------------------------------------
    // LABEL TENSOR: One-hot encoding
    // 
    // Convert [3, 7, 1, ...] to:
    // [[0,0,0,1,0,0,0,0,0,0],
    //  [0,0,0,0,0,0,0,1,0,0],
    //  [0,1,0,0,0,0,0,0,0,0], ...]
    // -------------------------------------------------------------------------
    const labelTensor = tf.tensor1d(labels, 'int32');
    const ys = tf.oneHot(labelTensor, DATA_CONFIG.numClasses);
    labelTensor.dispose(); // Clean up intermediate tensor
    
    console.log(`[DataGenerator] Generated: xs ${xs.shape}, ys ${ys.shape}`);
    
    return { xs, ys };
  });
}

/**
 * Generate training and validation datasets
 * 
 * @param {number} trainSize - Number of training samples
 * @param {number} valSize - Number of validation samples
 * @param {number[]} classDistribution - Optional class restriction for non-IID
 * @returns {Object} { train: {xs, ys}, val: {xs, ys} }
 */
function generateDatasets(trainSize = 500, valSize = 100, classDistribution = null) {
  console.log(`[DataGenerator] Generating datasets: train=${trainSize}, val=${valSize}`);
  
  const train = generateBatch(trainSize, classDistribution);
  const val = generateBatch(valSize, classDistribution);
  
  return { train, val };
}

/**
 * Simulate non-IID data distribution across clients
 * 
 * In real federated learning, different clients have different data distributions.
 * This function assigns each client a subset of classes to simulate this.
 * 
 * @param {number} clientId - Client identifier (0, 1, 2, ...)
 * @param {number} numClients - Total number of clients
 * @param {number} classesPerClient - How many classes each client gets
 * @returns {number[]} Array of class indices for this client
 */
function getNonIIDDistribution(clientId, numClients, classesPerClient = 3) {
  // Simple strategy: each client gets `classesPerClient` consecutive classes
  // with wraparound
  const startClass = (clientId * classesPerClient) % 10;
  const classes = [];
  
  for (let i = 0; i < classesPerClient; i++) {
    classes.push((startClass + i) % 10);
  }
  
  console.log(`[DataGenerator] Client ${clientId} non-IID classes: ${classes.join(', ')}`);
  return classes;
}

// =============================================================================
// VISUALIZATION HELPERS
// =============================================================================

/**
 * Convert a tensor image to canvas ImageData for visualization
 * 
 * @param {tf.Tensor} imageTensor - Single image tensor [28, 28, 1]
 * @returns {ImageData} Canvas-compatible image data
 */
function tensorToImageData(imageTensor) {
  const size = DATA_CONFIG.imageSize;
  const data = imageTensor.dataSync();
  
  const imageData = new ImageData(size, size);
  
  for (let i = 0; i < size * size; i++) {
    const value = Math.floor(data[i] * 255);
    imageData.data[i * 4 + 0] = value; // R
    imageData.data[i * 4 + 1] = value; // G
    imageData.data[i * 4 + 2] = value; // B
    imageData.data[i * 4 + 3] = 255;   // A
  }
  
  return imageData;
}

/**
 * Draw a batch of generated images to a canvas for debugging
 * 
 * @param {tf.Tensor4D} batchTensor - Batch of images [N, 28, 28, 1]
 * @param {HTMLCanvasElement} canvas - Target canvas
 * @param {number} cols - Number of columns in grid
 */
function drawBatchToCanvas(batchTensor, canvas, cols = 10) {
  const size = DATA_CONFIG.imageSize;
  const batchSize = batchTensor.shape[0];
  const rows = Math.ceil(batchSize / cols);
  
  canvas.width = cols * size;
  canvas.height = rows * size;
  
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  const images = tf.unstack(batchTensor);
  
  images.forEach((img, i) => {
    const row = Math.floor(i / cols);
    const col = i % cols;
    const imageData = tensorToImageData(img);
    ctx.putImageData(imageData, col * size, row * size);
    img.dispose();
  });
}

// =============================================================================
// EXPORTS (Global for browser use)
// =============================================================================

window.GridMindData = {
  generateBatch,
  generateDatasets,
  getNonIIDDistribution,
  tensorToImageData,
  drawBatchToCanvas,
  DATA_CONFIG,
};

console.log('[DataGenerator] GridMindData module loaded');
