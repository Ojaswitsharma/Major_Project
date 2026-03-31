/**
 * ============================================================================
 * GridMind - Federated Learning Server
 * ============================================================================
 * 
 * This server coordinates distributed model training across browser clients.
 * It maintains the global model state and handles weight synchronization
 * via WebSockets for real-time communication.
 * 
 * SOCKET EVENTS:
 * - 'pull_weights': Client requests current global weights
 * - 'push_weights': Client submits updated weights after local training
 * - 'get_stats': Client requests server statistics
 * 
 * ARCHITECTURE:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    Express + Socket.IO                      │
 * │  ┌─────────────┐    ┌──────────────────────────────────┐   │
 * │  │   Static    │    │         Socket.IO Hub            │   │
 * │  │   Files     │    │  • pull_weights → getWeights()   │   │
 * │  │  (client/)  │    │  • push_weights → setWeights()   │   │
 * │  └─────────────┘    └──────────────────────────────────┘   │
 * │                              │                              │
 * │                              ▼                              │
 * │                    ┌──────────────────┐                    │
 * │                    │  GlobalModelStore │                    │
 * │                    │  (In-Memory)      │                    │
 * │                    └──────────────────┘                    │
 * └─────────────────────────────────────────────────────────────┘
 */

import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

// Import our modules
import * as GlobalStore from './globalModelStore.js';
import * as FedUtils from './fedUtils.js';

// =============================================================================
// CONFIGURATION
// =============================================================================

const PORT = process.env.PORT || 3000;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// =============================================================================
// EXPRESS SETUP
// =============================================================================

const app = express();
const httpServer = createServer(app);

// Enable CORS for development
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST'],
}));

// Parse JSON bodies
app.use(express.json({ limit: '50mb' })); // Large limit for model weights

// Serve static files from client directory
app.use(express.static(path.join(__dirname, '../client')));

// =============================================================================
// REST ENDPOINTS (Alternative to WebSocket for debugging)
// =============================================================================

/**
 * GET /api/stats
 * Returns current server statistics
 */
app.get('/api/stats', (req, res) => {
  const stats = GlobalStore.getStats();
  res.json(stats);
});

/**
 * GET /api/weights
 * Returns current global weights (REST alternative to socket)
 */
app.get('/api/weights', (req, res) => {
  const { weights, version, metadata } = GlobalStore.getWeights();
  
  if (weights === null) {
    return res.json({
      initialized: false,
      message: 'Global model not yet initialized. Waiting for first client.',
    });
  }
  
  const serialized = FedUtils.serializeWeights(weights);
  res.json({
    initialized: true,
    version,
    metadata,
    weights: serialized,
  });
});

/**
 * POST /api/reset
 * Resets the global model (for testing)
 */
app.post('/api/reset', (req, res) => {
  GlobalStore.reset();
  res.json({ success: true, message: 'Global model reset' });
});

// =============================================================================
// SOCKET.IO SETUP
// =============================================================================

const io = new Server(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
  },
  // Increase max payload size for large models
  maxHttpBufferSize: 50 * 1024 * 1024, // 50MB
});

// =============================================================================
// SOCKET EVENT HANDLERS
// =============================================================================

io.on('connection', (socket) => {
  const clientId = socket.id;
  console.log(`\n[Server] Client connected: ${clientId}`);
  
  // Register client in store
  GlobalStore.registerClient(clientId);
  
  // Send welcome message with current stats
  socket.emit('welcome', {
    clientId,
    stats: GlobalStore.getStats(),
    message: 'Connected to GridMind server',
  });

  // -------------------------------------------------------------------------
  // EVENT: pull_weights
  // Client requests the current global model weights
  // -------------------------------------------------------------------------
  socket.on('pull_weights', (data, callback) => {
    console.log(`[Server] Client ${clientId} requesting weights`);
    
    const { weights, version, metadata } = GlobalStore.getWeights();
    
    if (weights === null) {
      // No global model yet - client should initialize
      console.log('[Server] No global weights yet - client will initialize');
      
      if (typeof callback === 'function') {
        callback({
          initialized: false,
          message: 'No global model. Please train and push initial weights.',
        });
      }
      return;
    }
    
    // -----------------------------------------------------------------------
    // SERIALIZATION: Convert Float32Arrays to network-transmittable format
    // This is where quantization would be applied in v2
    // -----------------------------------------------------------------------
    const serializedWeights = FedUtils.serializeWeights(weights, { 
      shapes: metadata.layerShapes 
    });
    
    console.log(`[Server] Sending weights v${version} (${metadata.totalParams} params)`);
    
    if (typeof callback === 'function') {
      callback({
        initialized: true,
        version,
        metadata,
        weights: serializedWeights,
      });
    }
  });

  // -------------------------------------------------------------------------
  // EVENT: push_weights
  // Client submits updated weights after local training + FedAvg
  // -------------------------------------------------------------------------
  socket.on('push_weights', (data, callback) => {
    console.log(`[Server] Client ${clientId} pushing weights`);
    
    try {
      const { weights: serializedWeights, localEpochs, localLoss } = data;
      
      // ---------------------------------------------------------------------
      // DESERIALIZATION: Convert network format back to Float32Arrays
      // This is where dequantization would be applied in v2
      // ---------------------------------------------------------------------
      const { weights, shapes } = FedUtils.deserializeWeights(serializedWeights);
      
      // Validate weights structure
      if (!Array.isArray(weights) || weights.length === 0) {
        throw new Error('Invalid weights format');
      }
      
      // Store the new global weights along with shapes
      const result = GlobalStore.setWeights(weights, clientId, shapes);
      
      console.log(`[Server] Weights updated to v${result.version}`);
      console.log(`[Server] Client trained for ${localEpochs} epochs, final loss: ${localLoss?.toFixed(4)}`);
      
      // Broadcast update to all connected clients
      socket.broadcast.emit('weights_updated', {
        version: result.version,
        updatedBy: clientId,
        stats: GlobalStore.getStats(),
      });
      
      if (typeof callback === 'function') {
        callback({
          success: true,
          version: result.version,
          message: 'Weights accepted',
        });
      }
      
    } catch (error) {
      console.error(`[Server] Error processing weights from ${clientId}:`, error.message);
      
      if (typeof callback === 'function') {
        callback({
          success: false,
          error: error.message,
        });
      }
    }
  });

  // -------------------------------------------------------------------------
  // EVENT: get_stats
  // Client requests current server statistics
  // -------------------------------------------------------------------------
  socket.on('get_stats', (callback) => {
    const stats = GlobalStore.getStats();
    
    if (typeof callback === 'function') {
      callback(stats);
    }
  });

  // -------------------------------------------------------------------------
  // EVENT: disconnect
  // Client disconnected
  // -------------------------------------------------------------------------
  socket.on('disconnect', (reason) => {
    console.log(`[Server] Client ${clientId} disconnected: ${reason}`);
    GlobalStore.unregisterClient(clientId);
  });
});

// =============================================================================
// SERVER STARTUP
// =============================================================================

httpServer.listen(PORT, () => {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('                    GridMind Server v1.0                        ');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`  Server running at:     http://localhost:${PORT}`);
  console.log(`  Client UI at:          http://localhost:${PORT}/index.html`);
  console.log(`  API Stats at:          http://localhost:${PORT}/api/stats`);
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('  Waiting for clients to connect...');
  console.log('');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n[Server] Shutting down...');
  io.close();
  httpServer.close();
  process.exit(0);
});
