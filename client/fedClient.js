/**
 * ============================================================================
 * GridMind - Federated Learning Client Utilities
 * ============================================================================
 * 
 * This module implements the core federated learning operations:
 * - Weight extraction from TF.js models
 * - Weight loading into TF.js models
 * - Federated Averaging (FedAvg) algorithm
 * - Weight serialization/deserialization for network transfer
 * 
 * FEDERATED AVERAGING (FedAvg) ALGORITHM:
 * =========================================
 * FedAvg is the foundational algorithm for federated learning, proposed by
 * McMahan et al. (2017). The key insight is that we can average model weights
 * instead of gradients, which reduces communication rounds.
 * 
 * Standard FedAvg formula:
 *   W_global = (1/N) * Σ W_client_i
 * 
 * Our simplified two-party averaging:
 *   W_new = (W_local + W_global) / 2
 * 
 * This is equivalent to FedAvg with two clients of equal weight.
 * 
 * TENSOR MEMORY MANAGEMENT:
 * ==========================
 * TensorFlow.js uses WebGL textures for GPU computation. Tensors must be
 * explicitly disposed to prevent memory leaks. We use tf.tidy() extensively
 * to automatically clean up intermediate tensors.
 * 
 * FUTURE-PROOFING HOOKS:
 * =======================
 * This module includes injection points for:
 * 1. Model Quantization (float32 → float16/int8)
 * 2. Differential Privacy (gradient clipping + noise)
 * 3. Secure Aggregation (encrypted weight transfer)
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * Transform pipeline configuration
 * These hooks allow for easy injection of quantization, DP, etc.
 */
const TRANSFORM_CONFIG = {
  // Quantization settings (disabled by default, enable in v2)
  quantization: {
    enabled: false,
    targetPrecision: 'float32', // 'float32' | 'float16' | 'int8'
  },
  
  // Differential Privacy settings (disabled by default, enable in v2)
  differentialPrivacy: {
    enabled: false,
    epsilon: 1.0,
    delta: 1e-5,
    clipNorm: 1.0,
  },
  
  // Compression (disabled by default)
  compression: {
    enabled: false,
    algorithm: 'none', // 'none' | 'gzip'
  },
};

// =============================================================================
// WEIGHT EXTRACTION
// =============================================================================

/**
 * Extract all trainable weights from a TF.js model
 * 
 * TENSOR STRUCTURE:
 * =================
 * A TF.js model's weights are organized as:
 * - model.getWeights() returns an array of tf.Tensor objects
 * - Each layer contributes 2 tensors: kernel (weights) and bias
 * - For Conv2D: kernel shape is [height, width, inChannels, outChannels]
 * - For Dense: kernel shape is [inputSize, outputSize]
 * 
 * MEMORY NOTES:
 * =============
 * We call tensor.dataSync() which copies GPU data to CPU. The returned
 * Float32Array is a copy, so we can safely dispose the original tensors
 * after extraction.
 * 
 * @param {tf.LayersModel} model - TF.js model to extract weights from
 * @returns {Object} { weights: Float32Array[], shapes: number[][], names: string[] }
 */
function extractWeights(model) {
  console.log('[FedClient] Extracting weights from model...');
  
  // -------------------------------------------------------------------------
  // STEP 1: Get all weight tensors from the model
  // 
  // model.getWeights() returns references to the actual tensors used by the
  // model. We need to copy their data before any modification.
  // -------------------------------------------------------------------------
  const tensors = model.getWeights();
  
  const weights = [];    // Array of Float32Array
  const shapes = [];     // Shape of each tensor for reconstruction
  const names = [];      // Names for debugging
  
  // -------------------------------------------------------------------------
  // STEP 2: Convert each tensor to Float32Array
  // 
  // tensor.dataSync() is synchronous and blocks until GPU→CPU transfer
  // is complete. For async version, use tensor.data() with await.
  // -------------------------------------------------------------------------
  for (let i = 0; i < tensors.length; i++) {
    const tensor = tensors[i];
    
    // Get tensor metadata
    const shape = tensor.shape;
    const name = `weight_${i}`;
    
    // -----------------------------------------------------------------------
    // CRITICAL: dataSync() copies data from GPU to CPU
    // This is where the actual tensor values become accessible
    // -----------------------------------------------------------------------
    const data = tensor.dataSync();
    
    // Create a copy of the data (dataSync returns a view in some cases)
    const weightsCopy = new Float32Array(data.length);
    weightsCopy.set(data);
    
    weights.push(weightsCopy);
    shapes.push(shape);
    names.push(name);
    
    console.log(`[FedClient]   Layer ${i}: shape=${JSON.stringify(shape)}, size=${data.length}`);
  }
  
  const totalParams = weights.reduce((sum, w) => sum + w.length, 0);
  console.log(`[FedClient] Extracted ${weights.length} weight tensors, ${totalParams} total parameters`);
  
  return { weights, shapes, names };
}

/**
 * Async version of weight extraction (non-blocking)
 * Preferred for large models to avoid blocking the main thread
 * 
 * @param {tf.LayersModel} model - TF.js model
 * @returns {Promise<Object>} Same as extractWeights
 */
async function extractWeightsAsync(model) {
  console.log('[FedClient] Extracting weights (async)...');
  
  const tensors = model.getWeights();
  const weights = [];
  const shapes = [];
  const names = [];
  
  for (let i = 0; i < tensors.length; i++) {
    const tensor = tensors[i];
    
    // -----------------------------------------------------------------------
    // ASYNC DATA EXTRACTION: tensor.data() returns a Promise
    // This allows other operations (UI updates) to proceed
    // -----------------------------------------------------------------------
    const data = await tensor.data();
    
    const weightsCopy = new Float32Array(data.length);
    weightsCopy.set(data);
    
    weights.push(weightsCopy);
    shapes.push(tensor.shape);
    names.push(`weight_${i}`);
  }
  
  return { weights, shapes, names };
}

// =============================================================================
// WEIGHT LOADING
// =============================================================================

/**
 * Load weights into a TF.js model
 * 
 * IMPORTANT: The weight arrays must match the model's architecture exactly.
 * Each Float32Array is converted back to a tensor with the correct shape.
 * 
 * @param {tf.LayersModel} model - Target model
 * @param {Float32Array[]} weights - Array of weight data
 * @param {number[][]} shapes - Original shapes of each weight tensor
 */
function loadWeights(model, weights, shapes) {
  console.log('[FedClient] Loading weights into model...');
  
  // -------------------------------------------------------------------------
  // STEP 1: Validate weight count matches model
  // -------------------------------------------------------------------------
  const expectedCount = model.getWeights().length;
  if (weights.length !== expectedCount) {
    throw new Error(
      `Weight count mismatch: model expects ${expectedCount}, got ${weights.length}`
    );
  }
  
  // -------------------------------------------------------------------------
  // STEP 2: Convert Float32Arrays back to tensors
  // 
  // tf.tidy() ensures that intermediate tensors are cleaned up.
  // The tensors created here will be owned by the model.
  // -------------------------------------------------------------------------
  const tensors = tf.tidy(() => {
    const tensorArray = [];
    
    for (let i = 0; i < weights.length; i++) {
      const weightData = weights[i];
      const shape = shapes[i];
      
      // Validate size matches shape
      const expectedSize = shape.reduce((a, b) => a * b, 1);
      if (weightData.length !== expectedSize) {
        throw new Error(
          `Size mismatch for weight ${i}: expected ${expectedSize}, got ${weightData.length}`
        );
      }
      
      // -----------------------------------------------------------------------
      // TENSOR CREATION: tf.tensor() creates a new tensor from data + shape
      // The data is copied to GPU memory (WebGL texture)
      // -----------------------------------------------------------------------
      const tensor = tf.tensor(weightData, shape);
      tensorArray.push(tensor);
    }
    
    return tensorArray;
  });
  
  // -------------------------------------------------------------------------
  // STEP 3: Apply weights to model
  // 
  // model.setWeights() replaces all trainable weights.
  // The old weights are automatically disposed.
  // -------------------------------------------------------------------------
  model.setWeights(tensors);
  
  // Clean up the tensor references (model now owns them)
  // Note: We don't dispose here because setWeights takes ownership
  
  console.log('[FedClient] Weights loaded successfully');
}

// =============================================================================
// FEDERATED AVERAGING (FedAvg)
// =============================================================================

/**
 * Perform Federated Averaging between local and global weights
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * FEDAVG MATH EXPLANATION
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * The Federated Averaging algorithm computes a weighted average of model
 * weights from multiple clients. In our simplified case with one local
 * client and one global model:
 * 
 *   W_new[i] = (W_local[i] + W_global[i]) / 2
 * 
 * Where:
 *   - W_local: Weights after local training
 *   - W_global: Current global weights (potentially updated by other clients)
 *   - W_new: Averaged weights to push back to server
 *   - i: Index into weight arrays (we average element-wise)
 * 
 * WEIGHTED AVERAGING (future enhancement):
 * =========================================
 * In production, you'd weight by dataset size:
 * 
 *   W_new = (n_local * W_local + n_global * W_global) / (n_local + n_global)
 * 
 * Where n_local and n_global are the number of samples each was trained on.
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * @param {Float32Array[]} localWeights - Weights from local training
 * @param {Float32Array[]} globalWeights - Latest global weights from server
 * @param {number} localWeight - Weight for local contribution (default 0.5)
 * @returns {Float32Array[]} Averaged weights
 */
function federatedAverage(localWeights, globalWeights, localWeight = 0.5) {
  console.log('[FedClient] Performing Federated Averaging...');
  console.log(`[FedClient]   Local weight: ${localWeight}, Global weight: ${1 - localWeight}`);
  
  // -------------------------------------------------------------------------
  // VALIDATION: Ensure weight arrays are compatible
  // -------------------------------------------------------------------------
  if (localWeights.length !== globalWeights.length) {
    throw new Error(
      `Layer count mismatch: local has ${localWeights.length}, global has ${globalWeights.length}`
    );
  }
  
  const globalWeight = 1 - localWeight;
  const averagedWeights = [];
  
  // -------------------------------------------------------------------------
  // ELEMENT-WISE AVERAGING
  // 
  // For each layer, we compute:
  //   averaged[j] = localWeight * local[j] + globalWeight * global[j]
  // 
  // This is done in pure JavaScript for simplicity. For very large models,
  // this could be GPU-accelerated using tf.add() and tf.mul().
  // -------------------------------------------------------------------------
  for (let i = 0; i < localWeights.length; i++) {
    const local = localWeights[i];
    const global = globalWeights[i];
    
    // Validate sizes match
    if (local.length !== global.length) {
      throw new Error(
        `Layer ${i} size mismatch: local=${local.length}, global=${global.length}`
      );
    }
    
    // -----------------------------------------------------------------------
    // CORE FEDAVG COMPUTATION
    // 
    // W_new[j] = α * W_local[j] + (1-α) * W_global[j]
    // 
    // Where α (alpha) = localWeight
    // -----------------------------------------------------------------------
    const averaged = new Float32Array(local.length);
    
    for (let j = 0; j < local.length; j++) {
      // This is the FedAvg formula for each individual weight parameter
      averaged[j] = localWeight * local[j] + globalWeight * global[j];
    }
    
    averagedWeights.push(averaged);
    
    // Log layer-wise statistics for debugging
    const localMean = local.reduce((a, b) => a + b, 0) / local.length;
    const globalMean = global.reduce((a, b) => a + b, 0) / global.length;
    const avgMean = averaged.reduce((a, b) => a + b, 0) / averaged.length;
    
    console.log(
      `[FedClient]   Layer ${i}: local_mean=${localMean.toFixed(6)}, ` +
      `global_mean=${globalMean.toFixed(6)}, avg_mean=${avgMean.toFixed(6)}`
    );
  }
  
  console.log('[FedClient] Federated Averaging complete');
  return averagedWeights;
}

/**
 * GPU-accelerated Federated Averaging using TF.js operations
 * More efficient for large models
 * 
 * @param {Float32Array[]} localWeights - Local weights
 * @param {Float32Array[]} globalWeights - Global weights
 * @param {number[][]} shapes - Tensor shapes
 * @param {number} localWeight - Local contribution weight
 * @returns {Float32Array[]} Averaged weights
 */
function federatedAverageGPU(localWeights, globalWeights, shapes, localWeight = 0.5) {
  console.log('[FedClient] GPU-accelerated FedAvg...');
  
  const globalWeight = 1 - localWeight;
  
  // Use tf.tidy to manage tensor memory
  const averaged = tf.tidy(() => {
    const result = [];
    
    for (let i = 0; i < localWeights.length; i++) {
      // Create tensors from raw data
      const localTensor = tf.tensor(localWeights[i], shapes[i]);
      const globalTensor = tf.tensor(globalWeights[i], shapes[i]);
      
      // -----------------------------------------------------------------------
      // GPU FEDAVG: W_new = α * W_local + (1-α) * W_global
      // 
      // tf.mul() performs element-wise multiplication
      // tf.add() performs element-wise addition
      // These operations run on WebGL/WebGPU
      // -----------------------------------------------------------------------
      const scaledLocal = tf.mul(localTensor, localWeight);
      const scaledGlobal = tf.mul(globalTensor, globalWeight);
      const avgTensor = tf.add(scaledLocal, scaledGlobal);
      
      // Copy back to CPU
      result.push(avgTensor.dataSync().slice());
    }
    
    return result;
  });
  
  // Convert TypedArray views to Float32Array copies
  return averaged.map(arr => new Float32Array(arr));
}

// =============================================================================
// SERIALIZATION / DESERIALIZATION (Client-Side)
// =============================================================================

/**
 * Serialize weights for network transmission
 * 
 * This function prepares weights for sending over WebSocket by:
 * 1. Converting Float32Arrays to base64 strings
 * 2. Including metadata for reconstruction
 * 
 * FUTURE: This is where quantization would be applied
 * 
 * @param {Float32Array[]} weights - Weight arrays
 * @param {number[][]} shapes - Tensor shapes
 * @returns {Object} Serialized weights ready for JSON transmission
 */
function serializeWeights(weights, shapes) {
  console.log('[FedClient] Serializing weights for transmission...');
  
  // -------------------------------------------------------------------------
  // HOOK: Apply differential privacy before transmission
  // -------------------------------------------------------------------------
  let processedWeights = weights;
  if (TRANSFORM_CONFIG.differentialPrivacy.enabled) {
    processedWeights = applyDifferentialPrivacy(weights);
  }
  
  // -------------------------------------------------------------------------
  // HOOK: Quantize weights to reduce bandwidth
  // -------------------------------------------------------------------------
  if (TRANSFORM_CONFIG.quantization.enabled) {
    processedWeights = quantizeWeights(processedWeights);
  }
  
  // -------------------------------------------------------------------------
  // BASE64 ENCODING: Convert binary data to text for JSON
  // -------------------------------------------------------------------------
  const layers = processedWeights.map((weightArray, index) => {
    // Convert Float32Array to base64
    const bytes = new Uint8Array(weightArray.buffer);
    const base64 = btoa(String.fromCharCode.apply(null, bytes));
    
    return {
      index,
      data: base64,
      length: weightArray.length,
      shape: shapes[index],
      dtype: TRANSFORM_CONFIG.quantization.enabled 
        ? TRANSFORM_CONFIG.quantization.targetPrecision 
        : 'float32',
    };
  });
  
  const totalParams = weights.reduce((sum, w) => sum + w.length, 0);
  console.log(`[FedClient] Serialized ${layers.length} layers, ${totalParams} params`);
  
  return {
    layers,
    metadata: {
      numLayers: layers.length,
      totalParams,
      timestamp: Date.now(),
    },
  };
}

/**
 * Deserialize weights received from server
 * 
 * @param {Object} payload - Serialized weight payload from server
 * @returns {Object} { weights: Float32Array[], shapes: number[][] }
 */
function deserializeWeights(payload) {
  console.log('[FedClient] Deserializing weights from server...');
  
  const weights = [];
  const shapes = [];
  
  for (const layer of payload.layers) {
    // -----------------------------------------------------------------------
    // BASE64 DECODING: Convert text back to binary
    // -----------------------------------------------------------------------
    const binary = atob(layer.data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    
    // Convert bytes back to Float32Array
    const float32 = new Float32Array(bytes.buffer);
    
    // Validate length
    if (float32.length !== layer.length) {
      console.warn(
        `[FedClient] Length mismatch in layer ${layer.index}: ` +
        `expected ${layer.length}, got ${float32.length}`
      );
    }
    
    weights.push(float32);
    shapes.push(layer.shape);
  }
  
  console.log(`[FedClient] Deserialized ${weights.length} weight tensors`);
  return { weights, shapes };
}

// =============================================================================
// DIFFERENTIAL PRIVACY HOOKS (Future Implementation)
// =============================================================================

/**
 * Apply differential privacy to weights before transmission
 * 
 * DP MECHANISM (Gaussian):
 * 1. Clip weight updates to bound sensitivity
 * 2. Add calibrated Gaussian noise
 * 
 * @param {Float32Array[]} weights - Original weights
 * @returns {Float32Array[]} Weights with DP noise
 */
function applyDifferentialPrivacy(weights) {
  // STUB: Currently returns weights unchanged
  // TODO: Implement in v2
  console.log('[FedClient] DP: Pass-through (not implemented)');
  return weights;
  
  // FUTURE IMPLEMENTATION:
  // const { epsilon, delta, clipNorm } = TRANSFORM_CONFIG.differentialPrivacy;
  // const sigma = clipNorm * Math.sqrt(2 * Math.log(1.25 / delta)) / epsilon;
  // 
  // return weights.map(layer => {
  //   const noisy = new Float32Array(layer.length);
  //   for (let i = 0; i < layer.length; i++) {
  //     const noise = gaussianRandom() * sigma;
  //     noisy[i] = layer[i] + noise;
  //   }
  //   return noisy;
  // });
}

// =============================================================================
// QUANTIZATION HOOKS (Future Implementation)
// =============================================================================

/**
 * Quantize weights to lower precision
 * 
 * @param {Float32Array[]} weights - Float32 weights
 * @returns {Float32Array[]|Int8Array[]} Quantized weights
 */
function quantizeWeights(weights) {
  // STUB: Currently returns weights unchanged
  // TODO: Implement float16/int8 quantization in v2
  console.log('[FedClient] Quantization: Pass-through (not implemented)');
  return weights;
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Calculate L2 norm of weight difference (for convergence monitoring)
 * 
 * @param {Float32Array[]} weights1 - First weight array
 * @param {Float32Array[]} weights2 - Second weight array
 * @returns {number} L2 norm of the difference
 */
function calculateWeightDifference(weights1, weights2) {
  let sumSquares = 0;
  
  for (let i = 0; i < weights1.length; i++) {
    for (let j = 0; j < weights1[i].length; j++) {
      const diff = weights1[i][j] - weights2[i][j];
      sumSquares += diff * diff;
    }
  }
  
  return Math.sqrt(sumSquares);
}

/**
 * Create a deep copy of weights array
 * 
 * @param {Float32Array[]} weights - Weights to copy
 * @returns {Float32Array[]} Deep copy
 */
function cloneWeights(weights) {
  return weights.map(layer => new Float32Array(layer));
}

// =============================================================================
// EXPORTS (Global for browser use)
// =============================================================================

window.GridMindFed = {
  // Core operations
  extractWeights,
  extractWeightsAsync,
  loadWeights,
  federatedAverage,
  federatedAverageGPU,
  
  // Serialization
  serializeWeights,
  deserializeWeights,
  
  // Utilities
  calculateWeightDifference,
  cloneWeights,
  
  // Configuration (for runtime adjustment)
  TRANSFORM_CONFIG,
};

console.log('[FedClient] GridMindFed module loaded');
