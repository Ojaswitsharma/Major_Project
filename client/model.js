/**
 * ============================================================================
 * GridMind - TensorFlow.js Model Definition
 * ============================================================================
 * 
 * This module defines the neural network architecture used for federated
 * learning. The model is a lightweight CNN designed for MNIST-like digit
 * classification, optimized for browser-based training.
 * 
 * ARCHITECTURE (Lightweight MNIST Classifier):
 * ┌────────────────────────────────────────────────────────────┐
 * │ Input: [28, 28, 1] - Grayscale image                       │
 * ├────────────────────────────────────────────────────────────┤
 * │ Conv2D: 8 filters, 3x3, ReLU → [26, 26, 8]                 │
 * │ MaxPool2D: 2x2               → [13, 13, 8]                 │
 * ├────────────────────────────────────────────────────────────┤
 * │ Conv2D: 16 filters, 3x3, ReLU → [11, 11, 16]               │
 * │ MaxPool2D: 2x2                → [5, 5, 16]                 │
 * ├────────────────────────────────────────────────────────────┤
 * │ Flatten                       → [400]                      │
 * │ Dense: 32 units, ReLU         → [32]                       │
 * │ Dense: 10 units, Softmax      → [10] (class probabilities) │
 * └────────────────────────────────────────────────────────────┘
 * 
 * Total Parameters: ~15,000 (small enough for fast FL rounds)
 * 
 * WHY THIS ARCHITECTURE:
 * - Small enough to train quickly in browser
 * - Weights transfer quickly over WebSocket
 * - Still complex enough to demonstrate FL convergence
 */

// =============================================================================
// MODEL CONFIGURATION
// =============================================================================

const MODEL_CONFIG = {
  inputShape: [28, 28, 1],  // MNIST image dimensions
  numClasses: 10,           // Digits 0-9
  
  // Convolutional layers
  conv1Filters: 8,
  conv2Filters: 16,
  kernelSize: 3,
  poolSize: 2,
  
  // Dense layers
  denseUnits: 32,
  
  // Training hyperparameters (can be overridden)
  defaultLearningRate: 0.01,
  defaultOptimizer: 'sgd',
};

// =============================================================================
// MODEL CREATION
// =============================================================================

/**
 * Create the MNIST classification model
 * 
 * TENSORFLOW.JS NOTES:
 * - tf.sequential() creates a linear stack of layers
 * - Each layer's output becomes the next layer's input
 * - Model must be compiled before training
 * 
 * @returns {tf.Sequential} Compiled TensorFlow.js model
 */
function createModel() {
  console.log('[Model] Creating MNIST classifier...');
  
  const model = tf.sequential({
    name: 'gridmind-mnist-classifier'
  });
  
  // -------------------------------------------------------------------------
  // LAYER 1: First Convolutional Block
  // Input: [28, 28, 1] → Output: [13, 13, 8]
  // -------------------------------------------------------------------------
  model.add(tf.layers.conv2d({
    name: 'conv1',
    inputShape: MODEL_CONFIG.inputShape,
    filters: MODEL_CONFIG.conv1Filters,
    kernelSize: MODEL_CONFIG.kernelSize,
    activation: 'relu',
    // He initialization for ReLU activation
    kernelInitializer: 'heNormal',
  }));
  
  model.add(tf.layers.maxPooling2d({
    name: 'pool1',
    poolSize: MODEL_CONFIG.poolSize,
  }));
  
  // -------------------------------------------------------------------------
  // LAYER 2: Second Convolutional Block
  // Input: [13, 13, 8] → Output: [5, 5, 16]
  // -------------------------------------------------------------------------
  model.add(tf.layers.conv2d({
    name: 'conv2',
    filters: MODEL_CONFIG.conv2Filters,
    kernelSize: MODEL_CONFIG.kernelSize,
    activation: 'relu',
    kernelInitializer: 'heNormal',
  }));
  
  model.add(tf.layers.maxPooling2d({
    name: 'pool2',
    poolSize: MODEL_CONFIG.poolSize,
  }));
  
  // -------------------------------------------------------------------------
  // LAYER 3: Flatten + Dense Layers
  // Input: [5, 5, 16] → Output: [10]
  // -------------------------------------------------------------------------
  model.add(tf.layers.flatten({
    name: 'flatten',
  }));
  
  model.add(tf.layers.dense({
    name: 'dense1',
    units: MODEL_CONFIG.denseUnits,
    activation: 'relu',
    kernelInitializer: 'heNormal',
  }));
  
  model.add(tf.layers.dense({
    name: 'output',
    units: MODEL_CONFIG.numClasses,
    activation: 'softmax',
    // Glorot (Xavier) initialization for softmax output
    kernelInitializer: 'glorotNormal',
  }));
  
  console.log('[Model] Model architecture created');
  
  return model;
}

/**
 * Compile the model with optimizer and loss function
 * 
 * COMPILATION SETTINGS:
 * - Optimizer: SGD (simple, works well with federated averaging)
 * - Loss: Categorical Cross-Entropy (standard for multi-class classification)
 * - Metrics: Accuracy (for monitoring)
 * 
 * @param {tf.Sequential} model - The model to compile
 * @param {Object} options - Compilation options
 */
function compileModel(model, options = {}) {
  const learningRate = options.learningRate || MODEL_CONFIG.defaultLearningRate;
  const optimizerType = options.optimizer || MODEL_CONFIG.defaultOptimizer;
  
  let optimizer;
  switch (optimizerType) {
    case 'adam':
      optimizer = tf.train.adam(learningRate);
      break;
    case 'rmsprop':
      optimizer = tf.train.rmsprop(learningRate);
      break;
    case 'sgd':
    default:
      // SGD is preferred for federated learning due to its simplicity
      // and better convergence guarantees with weight averaging
      optimizer = tf.train.sgd(learningRate);
      break;
  }
  
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  
  console.log(`[Model] Compiled with ${optimizerType} (lr=${learningRate})`);
}

/**
 * Get model summary as a string
 * Useful for debugging and display
 * 
 * @param {tf.Sequential} model - The model
 * @returns {string} Model summary
 */
function getModelSummary(model) {
  let summary = [];
  model.summary(undefined, undefined, (line) => summary.push(line));
  return summary.join('\n');
}

/**
 * Count total trainable parameters
 * 
 * @param {tf.Sequential} model - The model
 * @returns {number} Total parameter count
 */
function countParameters(model) {
  let totalParams = 0;
  for (const layer of model.layers) {
    const weights = layer.getWeights();
    for (const w of weights) {
      totalParams += w.size;
    }
  }
  return totalParams;
}

// =============================================================================
// EXPORTS (Global for browser use)
// =============================================================================

// Make functions available globally for browser
window.GridMindModel = {
  createModel,
  compileModel,
  getModelSummary,
  countParameters,
  MODEL_CONFIG,
};

console.log('[Model] GridMindModel module loaded');
