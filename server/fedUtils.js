/**
 * ============================================================================
 * GridMind - Federated Learning Utilities (Server-Side)
 * ============================================================================
 * 
 * This module provides weight serialization/deserialization with built-in
 * hooks for future enhancements like model quantization and differential privacy.
 * 
 * SERIALIZATION FORMAT:
 * Weights are transmitted as JSON with base64-encoded Float32Array data.
 * This format is chosen for WebSocket compatibility while maintaining
 * numerical precision.
 * 
 * FUTURE-PROOFING:
 * The transform pipeline is designed to easily inject:
 * 1. Quantization (float32 → float16/int8) for bandwidth reduction
 * 2. Differential Privacy noise for privacy guarantees
 * 3. Compression (gzip/brotli) for large models
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * Configuration for weight transformations
 * These settings control the privacy/compression pipeline
 */
export const transformConfig = {
  // Quantization settings (disabled by default)
  quantization: {
    enabled: false,
    precision: 'float32', // Options: 'float32', 'float16', 'int8'
    // For int8 quantization, we need scale and zero-point per tensor
    dynamicRange: true,   // Compute scale per tensor vs fixed range
  },

  // Differential Privacy settings (disabled by default)
  differentialPrivacy: {
    enabled: false,
    epsilon: 1.0,         // Privacy budget (lower = more privacy)
    delta: 1e-5,          // Probability of privacy breach
    clipNorm: 1.0,        // Max L2 norm for gradient clipping
    noiseMechanism: 'gaussian', // 'gaussian' or 'laplace'
  },

  // Compression settings (disabled by default)
  compression: {
    enabled: false,
    algorithm: 'none',    // Options: 'none', 'gzip', 'brotli'
  },
};

// =============================================================================
// SERIALIZATION / DESERIALIZATION
// =============================================================================

/**
 * Serialize weight tensors for network transmission
 * 
 * PIPELINE:
 * 1. Apply differential privacy (if enabled)
 * 2. Quantize weights (if enabled)
 * 3. Encode as base64 for JSON compatibility
 * 4. Compress (if enabled)
 * 
 * @param {Array<Float32Array>} weights - Array of weight tensors
 * @param {Object} options - Override default transform config
 * @returns {Object} Serialized weights with metadata
 */
export function serializeWeights(weights, options = {}) {
  const config = { ...transformConfig, ...options };
  
  // -------------------------------------------------------------------------
  // STEP 1: Apply Differential Privacy (HOOK - Currently Pass-Through)
  // -------------------------------------------------------------------------
  let processedWeights = weights;
  if (config.differentialPrivacy.enabled) {
    processedWeights = applyDifferentialPrivacy(weights, config.differentialPrivacy);
  }

  // -------------------------------------------------------------------------
  // STEP 2: Quantize Weights (HOOK - Currently Pass-Through)
  // -------------------------------------------------------------------------
  let quantizedWeights = processedWeights;
  let quantizationMetadata = null;
  if (config.quantization.enabled) {
    const result = quantizeWeights(processedWeights, config.quantization);
    quantizedWeights = result.weights;
    quantizationMetadata = result.metadata;
  }

  // -------------------------------------------------------------------------
  // STEP 3: Encode as Base64 for JSON transmission
  // -------------------------------------------------------------------------
  const encodedLayers = quantizedWeights.map((layer, index) => {
    // Convert Float32Array to base64 string
    const buffer = Buffer.from(layer.buffer);
    return {
      index,
      data: buffer.toString('base64'),
      length: layer.length,
      dtype: config.quantization.enabled ? config.quantization.precision : 'float32',
      // Include shape if provided in options
      shape: options.shapes ? options.shapes[index] : null,
    };
  });

  // -------------------------------------------------------------------------
  // STEP 4: Compress (HOOK - Currently Pass-Through)
  // -------------------------------------------------------------------------
  let finalPayload = {
    layers: encodedLayers,
    metadata: {
      numLayers: weights.length,
      totalParams: weights.reduce((sum, w) => sum + w.length, 0),
      quantization: quantizationMetadata,
      timestamp: Date.now(),
    },
  };

  if (config.compression.enabled) {
    finalPayload = compressPayload(finalPayload, config.compression);
  }

  return finalPayload;
}

/**
 * Deserialize weight tensors received from network
 * 
 * PIPELINE (reverse of serialization):
 * 1. Decompress (if compressed)
 * 2. Decode from base64
 * 3. Dequantize (if quantized)
 * 
 * @param {Object} payload - Serialized weight payload
 * @returns {Object} { weights: Array<Float32Array>, shapes: Array<number[]> }
 */
export function deserializeWeights(payload) {
  // -------------------------------------------------------------------------
  // STEP 1: Decompress (HOOK - Currently Pass-Through)
  // -------------------------------------------------------------------------
  let decompressedPayload = payload;
  if (payload.compressed) {
    decompressedPayload = decompressPayload(payload);
  }

  // -------------------------------------------------------------------------
  // STEP 2: Decode from Base64
  // -------------------------------------------------------------------------
  const weights = [];
  const shapes = [];
  
  for (const layerData of decompressedPayload.layers) {
    const buffer = Buffer.from(layerData.data, 'base64');
    
    let weightArray;
    // Handle different precisions
    if (layerData.dtype === 'float32' || !layerData.dtype) {
      weightArray = new Float32Array(buffer.buffer, buffer.byteOffset, layerData.length);
    } else if (layerData.dtype === 'float16') {
      // HOOK: Convert float16 back to float32
      weightArray = dequantizeFloat16ToFloat32(buffer, layerData.length);
    } else if (layerData.dtype === 'int8') {
      // HOOK: Convert int8 back to float32 using scale/zero-point
      weightArray = dequantizeInt8ToFloat32(buffer, decompressedPayload.metadata.quantization);
    } else {
      weightArray = new Float32Array(buffer.buffer, buffer.byteOffset, layerData.length);
    }
    
    // Make a copy to avoid issues with buffer views
    weights.push(new Float32Array(weightArray));
    shapes.push(layerData.shape || null);
  }

  return { weights, shapes };
}

// =============================================================================
// QUANTIZATION HOOKS (Future Implementation)
// =============================================================================

/**
 * Quantize weights to lower precision
 * 
 * QUANTIZATION MATH (for int8):
 * Given float32 weights W:
 *   scale = (max(W) - min(W)) / 255
 *   zero_point = round(-min(W) / scale)
 *   W_quantized = round(W / scale) + zero_point
 * 
 * @param {Array<Float32Array>} weights - Original float32 weights
 * @param {Object} config - Quantization configuration
 * @returns {Object} Quantized weights and metadata for dequantization
 */
function quantizeWeights(weights, config) {
  // -------------------------------------------------------------------------
  // STUB: Return weights unchanged for now
  // In v2, implement actual quantization here
  // -------------------------------------------------------------------------
  console.log('[FedUtils] Quantization requested but not yet implemented');
  
  return {
    weights: weights,
    metadata: {
      precision: 'float32',
      scales: null,
      zeroPoints: null,
    },
  };
  
  // FUTURE IMPLEMENTATION:
  // const quantizedLayers = [];
  // const scales = [];
  // const zeroPoints = [];
  // 
  // for (const layer of weights) {
  //   const min = Math.min(...layer);
  //   const max = Math.max(...layer);
  //   const scale = (max - min) / 255;
  //   const zeroPoint = Math.round(-min / scale);
  //   
  //   const quantized = new Int8Array(layer.length);
  //   for (let i = 0; i < layer.length; i++) {
  //     quantized[i] = Math.round(layer[i] / scale) + zeroPoint;
  //   }
  //   
  //   quantizedLayers.push(quantized);
  //   scales.push(scale);
  //   zeroPoints.push(zeroPoint);
  // }
  // 
  // return { weights: quantizedLayers, metadata: { scales, zeroPoints } };
}

/**
 * Convert float16 buffer back to Float32Array
 * @param {Buffer} buffer - Float16 data
 * @param {number} length - Number of elements
 * @returns {Float32Array} Dequantized weights
 */
function dequantizeFloat16ToFloat32(buffer, length) {
  // STUB: In v2, implement IEEE 754 half-precision conversion
  console.log('[FedUtils] Float16 dequantization not yet implemented');
  return new Float32Array(length);
}

/**
 * Convert int8 buffer back to Float32Array using scale/zero-point
 * @param {Buffer} buffer - Int8 data
 * @param {Object} metadata - Contains scales and zeroPoints
 * @returns {Float32Array} Dequantized weights
 */
function dequantizeInt8ToFloat32(buffer, metadata) {
  // STUB: In v2, implement int8 dequantization
  console.log('[FedUtils] Int8 dequantization not yet implemented');
  return new Float32Array(buffer.length);
}

// =============================================================================
// DIFFERENTIAL PRIVACY HOOKS (Future Implementation)
// =============================================================================

/**
 * Apply differential privacy to weights
 * 
 * DP MECHANISM:
 * 1. Clip each weight tensor's L2 norm to `clipNorm`
 * 2. Add calibrated Gaussian noise: σ = clipNorm * sqrt(2 * ln(1.25/δ)) / ε
 * 
 * @param {Array<Float32Array>} weights - Original weights
 * @param {Object} config - DP configuration (epsilon, delta, clipNorm)
 * @returns {Array<Float32Array>} Weights with DP noise added
 */
function applyDifferentialPrivacy(weights, config) {
  // -------------------------------------------------------------------------
  // STUB: Return weights unchanged for now
  // In v2, implement actual DP mechanism here
  // -------------------------------------------------------------------------
  console.log('[FedUtils] Differential Privacy requested but not yet implemented');
  console.log(`[FedUtils] Config: ε=${config.epsilon}, δ=${config.delta}, clip=${config.clipNorm}`);
  
  return weights;
  
  // FUTURE IMPLEMENTATION:
  // const { epsilon, delta, clipNorm, noiseMechanism } = config;
  // 
  // // Calculate noise scale (for Gaussian mechanism)
  // const sigma = clipNorm * Math.sqrt(2 * Math.log(1.25 / delta)) / epsilon;
  // 
  // return weights.map((layer) => {
  //   // Step 1: Clip L2 norm
  //   const l2Norm = Math.sqrt(layer.reduce((sum, w) => sum + w * w, 0));
  //   const clipFactor = Math.min(1.0, clipNorm / l2Norm);
  //   
  //   // Step 2: Add Gaussian noise
  //   const noisyLayer = new Float32Array(layer.length);
  //   for (let i = 0; i < layer.length; i++) {
  //     const clippedWeight = layer[i] * clipFactor;
  //     const noise = gaussianRandom() * sigma;
  //     noisyLayer[i] = clippedWeight + noise;
  //   }
  //   
  //   return noisyLayer;
  // });
}

// =============================================================================
// COMPRESSION HOOKS (Future Implementation)
// =============================================================================

/**
 * Compress payload for network transmission
 * @param {Object} payload - Weight payload
 * @param {Object} config - Compression configuration
 * @returns {Object} Compressed payload
 */
function compressPayload(payload, config) {
  // STUB: Return unchanged for now
  // In v2, use zlib.gzipSync() or brotli
  console.log('[FedUtils] Compression requested but not yet implemented');
  return payload;
}

/**
 * Decompress payload received from network
 * @param {Object} payload - Compressed payload
 * @returns {Object} Decompressed payload
 */
function decompressPayload(payload) {
  // STUB: Return unchanged for now
  console.log('[FedUtils] Decompression requested but not yet implemented');
  return payload;
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Calculate the byte size of serialized weights
 * Useful for bandwidth estimation
 * @param {Array<Float32Array>} weights - Weight tensors
 * @returns {number} Size in bytes
 */
export function calculateWeightSize(weights) {
  return weights.reduce((sum, layer) => sum + layer.byteLength, 0);
}

/**
 * Validate that two weight arrays have compatible shapes
 * @param {Array<Float32Array>} weights1 - First weight array
 * @param {Array<Float32Array>} weights2 - Second weight array
 * @returns {boolean} True if shapes match
 */
export function validateWeightShapes(weights1, weights2) {
  if (weights1.length !== weights2.length) {
    return false;
  }
  
  for (let i = 0; i < weights1.length; i++) {
    if (weights1[i].length !== weights2[i].length) {
      return false;
    }
  }
  
  return true;
}
