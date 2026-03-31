/**
 * ============================================================================
 * GridMind - Global Model Store (Redis-Style In-Memory Storage)
 * ============================================================================
 * 
 * This module implements an in-memory store for the global federated model
 * weights. It mimics Redis-style operations with version tracking for
 * optimistic concurrency control.
 * 
 * ARCHITECTURE NOTES:
 * - Weights are stored as an array of Float32Array (one per layer)
 * - Version number increments on each update for conflict detection
 * - In production, this would be replaced by Redis/PostgreSQL with proper
 *   locking mechanisms
 * 
 * FUTURE ENHANCEMENTS:
 * - Add Redis adapter for horizontal scaling
 * - Implement optimistic locking with version checks
 * - Add weight history for rollback capabilities
 */

// =============================================================================
// STORE STATE
// =============================================================================

/**
 * The global model state object
 * @typedef {Object} GlobalModelState
 * @property {Array<Float32Array>|null} weights - Array of weight tensors (one per layer)
 * @property {number} version - Monotonically increasing version counter
 * @property {number} lastUpdated - Unix timestamp of last update
 * @property {Object} metadata - Additional model information
 */
const globalModelState = {
  weights: null,           // Will be initialized when first client pushes weights
  version: 0,              // Increments on each update (for optimistic concurrency)
  lastUpdated: Date.now(), // Timestamp for staleness detection
  metadata: {
    layerShapes: [],       // Shape of each weight tensor for validation
    totalParams: 0,        // Total number of parameters
    contributorCount: 0,   // Number of unique contributors
  },
};

// Track connected clients for analytics
const connectedClients = new Set();

// =============================================================================
// PUBLIC API
// =============================================================================

/**
 * Get the current global model weights
 * 
 * TENSOR FORMAT:
 * Returns an array where each element corresponds to a layer's weights.
 * For a Conv2D layer, the weights tensor has shape [height, width, inChannels, outChannels]
 * For a Dense layer, the weights tensor has shape [inputSize, outputSize]
 * Biases are separate tensors with shape [numUnits]
 * 
 * @returns {Object} Object containing weights array, version, and metadata
 */
export function getWeights() {
  return {
    weights: globalModelState.weights,
    version: globalModelState.version,
    lastUpdated: globalModelState.lastUpdated,
    metadata: { ...globalModelState.metadata },
  };
}

/**
 * Update the global model weights
 * 
 * CONCURRENCY MODEL (Last-Write-Wins):
 * This implementation uses a simple last-write-wins strategy. The incoming
 * weights overwrite the current state regardless of version. This is acceptable
 * for prototyping but should be replaced with proper merging in production.
 * 
 * ALTERNATIVE STRATEGIES (for future implementation):
 * 1. Optimistic Locking: Reject if client's baseVersion != current version
 * 2. Server-Side Averaging: Merge incoming weights with current state
 * 3. Weighted Averaging: Weight contributions by dataset size or model quality
 * 
 * @param {Array<Float32Array>} newWeights - Array of weight tensors
 * @param {string} clientId - ID of the contributing client
 * @param {Object} layerShapes - Shape information for validation
 * @returns {Object} Result with new version number
 */
export function setWeights(newWeights, clientId, layerShapes = []) {
  // -------------------------------------------------------------------------
  // VALIDATION: Ensure weights array is properly formatted
  // -------------------------------------------------------------------------
  if (!Array.isArray(newWeights)) {
    throw new Error('Weights must be an array of Float32Arrays');
  }

  // Validate each weight tensor
  for (let i = 0; i < newWeights.length; i++) {
    if (!(newWeights[i] instanceof Float32Array)) {
      throw new Error(`Weight tensor at index ${i} must be a Float32Array`);
    }
  }

  // -------------------------------------------------------------------------
  // SHAPE VALIDATION: Ensure weights match expected architecture
  // -------------------------------------------------------------------------
  if (globalModelState.weights !== null) {
    // Check that the number of layers matches
    if (newWeights.length !== globalModelState.weights.length) {
      throw new Error(
        `Weight array length mismatch: expected ${globalModelState.weights.length}, got ${newWeights.length}`
      );
    }

    // Check that each layer has the correct number of parameters
    for (let i = 0; i < newWeights.length; i++) {
      if (newWeights[i].length !== globalModelState.weights[i].length) {
        throw new Error(
          `Layer ${i} size mismatch: expected ${globalModelState.weights[i].length}, got ${newWeights[i].length}`
        );
      }
    }
  }

  // -------------------------------------------------------------------------
  // UPDATE STATE: Apply the new weights
  // -------------------------------------------------------------------------
  globalModelState.weights = newWeights;
  globalModelState.version += 1;
  globalModelState.lastUpdated = Date.now();
  
  // Update metadata
  // Only update shapes if we have valid shape data
  if (layerShapes && layerShapes.length > 0 && layerShapes.some(s => s !== null)) {
    globalModelState.metadata.layerShapes = layerShapes;
  }
  globalModelState.metadata.totalParams = newWeights.reduce(
    (sum, layer) => sum + layer.length, 
    0
  );
  globalModelState.metadata.contributorCount += 1;

  console.log(`[GlobalStore] Weights updated by client ${clientId}`);
  console.log(`[GlobalStore] Version: ${globalModelState.version}, Params: ${globalModelState.metadata.totalParams}`);

  return {
    version: globalModelState.version,
    success: true,
  };
}

/**
 * Check if the global model has been initialized
 * @returns {boolean} True if weights exist
 */
export function hasWeights() {
  return globalModelState.weights !== null;
}

/**
 * Get current store statistics
 * @returns {Object} Stats about the global model
 */
export function getStats() {
  return {
    hasWeights: globalModelState.weights !== null,
    version: globalModelState.version,
    lastUpdated: globalModelState.lastUpdated,
    totalParams: globalModelState.metadata.totalParams,
    contributorCount: globalModelState.metadata.contributorCount,
    connectedClients: connectedClients.size,
  };
}

/**
 * Register a new client connection
 * @param {string} clientId - Unique client identifier
 */
export function registerClient(clientId) {
  connectedClients.add(clientId);
  console.log(`[GlobalStore] Client ${clientId} connected. Total: ${connectedClients.size}`);
}

/**
 * Unregister a client on disconnect
 * @param {string} clientId - Unique client identifier
 */
export function unregisterClient(clientId) {
  connectedClients.delete(clientId);
  console.log(`[GlobalStore] Client ${clientId} disconnected. Total: ${connectedClients.size}`);
}

/**
 * Reset the global model (useful for testing)
 */
export function reset() {
  globalModelState.weights = null;
  globalModelState.version = 0;
  globalModelState.lastUpdated = Date.now();
  globalModelState.metadata = {
    layerShapes: [],
    totalParams: 0,
    contributorCount: 0,
  };
  console.log('[GlobalStore] Store reset');
}
