/**
 * ============================================================================
 * GridMind - Main Client Orchestrator
 * ============================================================================
 * 
 * This module orchestrates the complete federated learning flow:
 * 
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                    ASYNCHRONOUS FEDERATED LEARNING FLOW                   ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
 * ║  │ STEP A: PULL                                                        │  ║
 * ║  │ Client connects and fetches current global weights (W_global_t0)    │  ║
 * ║  └────────────────────────────────┬────────────────────────────────────┘  ║
 * ║                                   ▼                                       ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
 * ║  │ STEP B: TRAIN                                                       │  ║
 * ║  │ Load weights, train locally for N epochs → produces W_local         │  ║
 * ║  │ Uses client's GPU (WebGL/WebGPU) for acceleration                   │  ║
 * ║  └────────────────────────────────┬────────────────────────────────────┘  ║
 * ║                                   ▼                                       ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
 * ║  │ STEP C: RE-FETCH                                                    │  ║
 * ║  │ Fetch latest global weights (W_global_t1) - may have changed!       │  ║
 * ║  │ Other clients may have pushed updates during our training           │  ║
 * ║  └────────────────────────────────┬────────────────────────────────────┘  ║
 * ║                                   ▼                                       ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
 * ║  │ STEP D: AGGREGATE (FedAvg)                                          │  ║
 * ║  │ W_new = (W_local + W_global_t1) / 2                                 │  ║
 * ║  │ Combines our learning with latest global state                      │  ║
 * ║  └────────────────────────────────┬────────────────────────────────────┘  ║
 * ║                                   ▼                                       ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐  ║
 * ║  │ STEP E: PUSH                                                        │  ║
 * ║  │ Send W_new to server to update global state                         │  ║
 * ║  └─────────────────────────────────────────────────────────────────────┘  ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 * 
 * SOCKET.IO EVENTS:
 * - 'pull_weights': Request current global weights
 * - 'push_weights': Submit new weights after FedAvg
 * - 'weights_updated': Notification when another client updates global model
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const CLIENT_CONFIG = {
  serverUrl: window.location.origin,  // Same origin as page
  localEpochs: 3,                     // Epochs per federated round
  batchSize: 32,                      // Training batch size
  trainingSamples: 500,               // Size of local dataset
  validationSamples: 100,             // Size of validation set
  fedAvgLocalWeight: 0.5,             // Weight for local contribution in FedAvg
};

// =============================================================================
// STATE
// =============================================================================

/**
 * Client state object
 */
const clientState = {
  socket: null,
  clientId: null,
  model: null,
  isTraining: false,
  isConnected: false,
  currentRound: 0,
  
  // Training data (generated once, reused)
  trainData: null,
  valData: null,
  
  // Weight tracking
  lastGlobalVersion: null,
  localWeights: null,
  localShapes: null,
  
  // Backend info (GPU/CPU)
  backendInfo: null,
  
  // Metrics
  trainingHistory: [],
  roundHistory: [],
};

// UI Update callbacks
let onStatusUpdate = null;
let onMetricsUpdate = null;
let onLogMessage = null;

// =============================================================================
// INITIALIZATION
// =============================================================================

/**
 * Initialize the federated learning client
 * Sets up TensorFlow.js, creates the model, and connects to server
 */
async function initialize() {
  log('Initializing GridMind client...');
  
  // -------------------------------------------------------------------------
  // STEP 1: Configure TensorFlow.js backend (WebGL/WebGPU)
  // -------------------------------------------------------------------------
  await initializeTensorFlow();
  
  // -------------------------------------------------------------------------
  // STEP 2: Create the neural network model
  // -------------------------------------------------------------------------
  log('Creating model...');
  clientState.model = GridMindModel.createModel();
  GridMindModel.compileModel(clientState.model);
  
  const paramCount = GridMindModel.countParameters(clientState.model);
  log(`Model created: ${paramCount} parameters`);
  
  // Print model summary
  clientState.model.summary();
  
  // -------------------------------------------------------------------------
  // STEP 3: Generate local training data
  // -------------------------------------------------------------------------
  log('Generating synthetic training data...');
  const datasets = GridMindData.generateDatasets(
    CLIENT_CONFIG.trainingSamples,
    CLIENT_CONFIG.validationSamples
  );
  clientState.trainData = datasets.train;
  clientState.valData = datasets.val;
  log(`Generated ${CLIENT_CONFIG.trainingSamples} training samples`);
  
  // -------------------------------------------------------------------------
  // STEP 4: Connect to server via WebSocket
  // -------------------------------------------------------------------------
  await connectToServer();
  
  updateStatus('ready');
  log('Initialization complete. Ready to train.');
}

/**
 * Initialize TensorFlow.js with optimal backend
 * Tries GPU backends first, falls back to CPU if unavailable
 */
async function initializeTensorFlow() {
  log('Configuring TensorFlow.js...');
  log('Detecting available compute backends...');
  
  let selectedBackend = null;
  let isGPU = false;
  
  // -------------------------------------------------------------------------
  // TRY 1: WebGPU (newest, fastest - requires modern browser + GPU)
  // -------------------------------------------------------------------------
  try {
    await tf.setBackend('webgpu');
    await tf.ready();
    
    // Verify it actually initialized
    if (tf.getBackend() === 'webgpu') {
      selectedBackend = 'webgpu';
      isGPU = true;
      log('WebGPU backend initialized successfully');
    }
  } catch (e) {
    log('WebGPU not available: ' + (e.message || 'Not supported'));
  }
  
  // -------------------------------------------------------------------------
  // TRY 2: WebGL (widely supported GPU acceleration)
  // -------------------------------------------------------------------------
  if (!selectedBackend) {
    try {
      await tf.setBackend('webgl');
      await tf.ready();
      
      // Verify it actually initialized and is using GPU
      if (tf.getBackend() === 'webgl') {
        // Check if WebGL is actually hardware accelerated
        const gl = document.createElement('canvas').getContext('webgl');
        if (gl) {
          const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
          if (debugInfo) {
            const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
            log(`WebGL renderer: ${renderer}`);
            
            // Check if it's a software renderer (no real GPU)
            const isSoftware = renderer.toLowerCase().includes('swiftshader') ||
                              renderer.toLowerCase().includes('llvmpipe') ||
                              renderer.toLowerCase().includes('software');
            
            if (isSoftware) {
              log('WebGL is using software rendering (no GPU detected)');
              // Fall through to CPU
            } else {
              selectedBackend = 'webgl';
              isGPU = true;
              log('WebGL backend initialized with GPU acceleration');
            }
          } else {
            // Can't detect renderer, assume WebGL works
            selectedBackend = 'webgl';
            isGPU = true;
            log('WebGL backend initialized');
          }
        }
      }
    } catch (e) {
      log('WebGL not available: ' + (e.message || 'Not supported'));
    }
  }
  
  // -------------------------------------------------------------------------
  // FALLBACK: CPU (always available, but slower)
  // -------------------------------------------------------------------------
  if (!selectedBackend) {
    log('');
    log('========================================');
    log('  GPU NOT FOUND - USING CPU BACKEND');
    log('========================================');
    log('Training will be slower without GPU acceleration.');
    log('For better performance, use a device with a GPU.');
    log('');
    
    await tf.setBackend('cpu');
    await tf.ready();
    selectedBackend = 'cpu';
    isGPU = false;
  }
  
  // -------------------------------------------------------------------------
  // Final status
  // -------------------------------------------------------------------------
  const finalBackend = tf.getBackend();
  
  if (isGPU) {
    log(`TensorFlow.js ready with GPU acceleration (${finalBackend.toUpperCase()})`);
  } else {
    log(`TensorFlow.js ready on CPU (training will be slower)`);
  }
  
  // Store backend info in client state for UI
  clientState.backendInfo = {
    backend: finalBackend,
    isGPU: isGPU,
  };
}

/**
 * Connect to the GridMind server via Socket.IO
 */
async function connectToServer() {
  return new Promise((resolve, reject) => {
    log(`Connecting to server at ${CLIENT_CONFIG.serverUrl}...`);
    
    // Create Socket.IO connection
    clientState.socket = io(CLIENT_CONFIG.serverUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });
    
    // Handle successful connection
    clientState.socket.on('connect', () => {
      clientState.isConnected = true;
      log('Connected to server');
      updateStatus('connected');
    });
    
    // Handle welcome message with client ID
    clientState.socket.on('welcome', (data) => {
      clientState.clientId = data.clientId;
      log(`Assigned client ID: ${clientState.clientId}`);
      log(`Server stats: ${JSON.stringify(data.stats)}`);
      resolve();
    });
    
    // Handle disconnection
    clientState.socket.on('disconnect', (reason) => {
      clientState.isConnected = false;
      log(`Disconnected: ${reason}`);
      updateStatus('disconnected');
    });
    
    // Handle weight update notifications from other clients
    clientState.socket.on('weights_updated', (data) => {
      log(`Global model updated by another client (v${data.version})`);
      if (onMetricsUpdate) {
        onMetricsUpdate({ globalVersion: data.version });
      }
    });
    
    // Handle connection error
    clientState.socket.on('connect_error', (error) => {
      log(`Connection error: ${error.message}`);
      reject(error);
    });
    
    // Timeout after 10 seconds
    setTimeout(() => {
      if (!clientState.isConnected) {
        reject(new Error('Connection timeout'));
      }
    }, 10000);
  });
}

// =============================================================================
// FEDERATED LEARNING FLOW
// =============================================================================

/**
 * Execute one complete federated learning round
 * 
 * This implements the full async FL flow:
 * A) Pull → B) Train → C) Re-Fetch → D) Aggregate → E) Push
 */
async function executeFederatedRound() {
  if (clientState.isTraining) {
    log('Already training, please wait...');
    return;
  }
  
  clientState.isTraining = true;
  clientState.currentRound++;
  updateStatus('training');
  
  const roundStart = Date.now();
  log(`\n${'═'.repeat(60)}`);
  log(`FEDERATED ROUND ${clientState.currentRound}`);
  log(`${'═'.repeat(60)}`);
  
  try {
    // =========================================================================
    // STEP A: PULL - Fetch current global weights
    // =========================================================================
    log('\n[STEP A] Pulling global weights from server...');
    
    const pullResult = await pullWeights();
    
    if (pullResult.initialized) {
      // Global model exists - load it
      log(`Received global weights v${pullResult.version}`);
      const { weights, shapes } = GridMindFed.deserializeWeights(pullResult.weights);
      GridMindFed.loadWeights(clientState.model, weights, shapes);
      clientState.lastGlobalVersion = pullResult.version;
    } else {
      // No global model yet - we'll initialize it
      log('No global model yet. This client will initialize it.');
    }
    
    // =========================================================================
    // STEP B: TRAIN - Local training on private data
    // =========================================================================
    log('\n[STEP B] Training locally on GPU...');
    log(`Training for ${CLIENT_CONFIG.localEpochs} epochs...`);
    
    const trainingResult = await trainLocally();
    
    log(`Training complete. Final loss: ${trainingResult.finalLoss.toFixed(4)}`);
    log(`Final accuracy: ${(trainingResult.finalAccuracy * 100).toFixed(2)}%`);
    
    // Extract trained weights
    const extracted = GridMindFed.extractWeights(clientState.model);
    clientState.localWeights = extracted.weights;
    clientState.localShapes = extracted.shapes;
    
    // =========================================================================
    // STEP C: RE-FETCH - Get latest global weights (may have changed!)
    // =========================================================================
    log('\n[STEP C] Re-fetching latest global weights...');
    
    const refetchResult = await pullWeights();
    
    let weightsToSend;
    let shapesToSend;
    
    if (!refetchResult.initialized) {
      // Still no global model - send our weights directly
      log('Still no global model. Pushing our weights as initial global state.');
      weightsToSend = clientState.localWeights;
      shapesToSend = clientState.localShapes;
    } else {
      const newVersion = refetchResult.version;
      log(`Latest global version: ${newVersion}`);
      
      if (newVersion !== clientState.lastGlobalVersion) {
        log(`Global model changed during training (${clientState.lastGlobalVersion} → ${newVersion})`);
      }
      
      // =====================================================================
      // STEP D: AGGREGATE - Federated Averaging
      // =====================================================================
      log('\n[STEP D] Performing Federated Averaging...');
      
      const globalData = GridMindFed.deserializeWeights(refetchResult.weights);
      
      // -----------------------------------------------------------------------
      // FEDAVG COMPUTATION
      // 
      // W_new = (W_local + W_global_t1) / 2
      // 
      // This averages our locally trained weights with the latest global
      // weights, giving equal importance to both.
      // -----------------------------------------------------------------------
      weightsToSend = GridMindFed.federatedAverage(
        clientState.localWeights,  // Our trained weights
        globalData.weights,        // Latest global weights
        CLIENT_CONFIG.fedAvgLocalWeight  // 0.5 by default
      );
      shapesToSend = globalData.shapes;
      
      // Calculate how much our weights changed
      const diff = GridMindFed.calculateWeightDifference(
        clientState.localWeights,
        globalData.weights
      );
      log(`Weight difference (L2 norm): ${diff.toFixed(6)}`);
    }
    
    // =========================================================================
    // STEP E: PUSH - Send averaged weights to server
    // =========================================================================
    log('\n[STEP E] Pushing averaged weights to server...');
    
    const serialized = GridMindFed.serializeWeights(weightsToSend, shapesToSend);
    
    const pushResult = await pushWeights(serialized, trainingResult);
    
    if (pushResult.success) {
      log(`Weights accepted. New global version: ${pushResult.version}`);
    } else {
      log(`Push failed: ${pushResult.error}`);
    }
    
    // =========================================================================
    // ROUND COMPLETE
    // =========================================================================
    const roundTime = (Date.now() - roundStart) / 1000;
    log(`\n${'─'.repeat(60)}`);
    log(`Round ${clientState.currentRound} complete in ${roundTime.toFixed(2)}s`);
    log(`${'─'.repeat(60)}\n`);
    
    // Record round history
    clientState.roundHistory.push({
      round: clientState.currentRound,
      loss: trainingResult.finalLoss,
      accuracy: trainingResult.finalAccuracy,
      duration: roundTime,
      globalVersion: pushResult.version,
    });
    
    if (onMetricsUpdate) {
      onMetricsUpdate({
        round: clientState.currentRound,
        loss: trainingResult.finalLoss,
        accuracy: trainingResult.finalAccuracy,
        globalVersion: pushResult.version,
      });
    }
    
  } catch (error) {
    log(`Error in federated round: ${error.message}`);
    console.error(error);
  } finally {
    clientState.isTraining = false;
    updateStatus('ready');
  }
}

/**
 * Pull weights from server
 * @returns {Promise<Object>} Server response with weights
 */
function pullWeights() {
  return new Promise((resolve, reject) => {
    clientState.socket.emit('pull_weights', {}, (response) => {
      resolve(response);
    });
    
    // Timeout
    setTimeout(() => reject(new Error('Pull timeout')), 30000);
  });
}

/**
 * Push weights to server
 * @param {Object} serializedWeights - Serialized weight payload
 * @param {Object} trainingResult - Training metrics
 * @returns {Promise<Object>} Server response
 */
function pushWeights(serializedWeights, trainingResult) {
  return new Promise((resolve, reject) => {
    clientState.socket.emit('push_weights', {
      weights: serializedWeights,
      localEpochs: CLIENT_CONFIG.localEpochs,
      localLoss: trainingResult.finalLoss,
    }, (response) => {
      resolve(response);
    });
    
    // Timeout
    setTimeout(() => reject(new Error('Push timeout')), 30000);
  });
}

/**
 * Train the model locally on synthetic data
 * @returns {Promise<Object>} Training results
 */
async function trainLocally() {
  const history = { loss: [], accuracy: [] };
  
  // -------------------------------------------------------------------------
  // TF.JS MODEL.FIT()
  // 
  // This runs backpropagation on the GPU using WebGL/WebGPU.
  // The 'callbacks' option lets us track progress per epoch.
  // -------------------------------------------------------------------------
  await clientState.model.fit(clientState.trainData.xs, clientState.trainData.ys, {
    epochs: CLIENT_CONFIG.localEpochs,
    batchSize: CLIENT_CONFIG.batchSize,
    validationData: [clientState.valData.xs, clientState.valData.ys],
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        log(`  Epoch ${epoch + 1}/${CLIENT_CONFIG.localEpochs}: ` +
            `loss=${logs.loss.toFixed(4)}, acc=${(logs.acc * 100).toFixed(2)}%`);
        
        history.loss.push(logs.loss);
        history.accuracy.push(logs.acc);
        
        // Update metrics
        if (onMetricsUpdate) {
          onMetricsUpdate({ 
            epoch: epoch + 1, 
            loss: logs.loss, 
            accuracy: logs.acc 
          });
        }
      },
    },
  });
  
  // Store history
  clientState.trainingHistory.push(...history.loss.map((loss, i) => ({
    round: clientState.currentRound,
    epoch: i + 1,
    loss,
    accuracy: history.accuracy[i],
  })));
  
  return {
    history,
    finalLoss: history.loss[history.loss.length - 1],
    finalAccuracy: history.accuracy[history.accuracy.length - 1],
  };
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Log a message to console and UI
 * @param {string} message - Message to log
 */
function log(message) {
  const timestamp = new Date().toLocaleTimeString();
  console.log(`[${timestamp}] ${message}`);
  
  if (onLogMessage) {
    onLogMessage(`[${timestamp}] ${message}`);
  }
}

/**
 * Update client status
 * @param {string} status - New status
 */
function updateStatus(status) {
  if (onStatusUpdate) {
    onStatusUpdate(status);
  }
}

/**
 * Set UI callback functions
 * @param {Object} callbacks - Callback functions
 */
function setCallbacks(callbacks) {
  onStatusUpdate = callbacks.onStatusUpdate;
  onMetricsUpdate = callbacks.onMetricsUpdate;
  onLogMessage = callbacks.onLogMessage;
}

/**
 * Get current client state
 * @returns {Object} Current state
 */
function getState() {
  return {
    clientId: clientState.clientId,
    isConnected: clientState.isConnected,
    isTraining: clientState.isTraining,
    currentRound: clientState.currentRound,
    trainingHistory: clientState.trainingHistory,
    roundHistory: clientState.roundHistory,
    model: clientState.model,  // Expose model for UI access
    backendInfo: clientState.backendInfo,  // GPU/CPU info
  };
}

/**
 * Run multiple federated rounds
 * @param {number} numRounds - Number of rounds to run
 */
async function runMultipleRounds(numRounds) {
  log(`Starting ${numRounds} federated rounds...`);
  
  for (let i = 0; i < numRounds; i++) {
    await executeFederatedRound();
    
    // Small delay between rounds
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  log(`Completed ${numRounds} federated rounds`);
}

// =============================================================================
// EXPORTS (Global for browser use)
// =============================================================================

window.GridMindClient = {
  initialize,
  executeFederatedRound,
  runMultipleRounds,
  setCallbacks,
  getState,
  CLIENT_CONFIG,
};

console.log('[Client] GridMindClient module loaded');
