# GridMind - Browser-Based Federated Learning Platform

A prototype web-based federated model training platform that allows community users to collaboratively train a global ML model directly in their browsers using their local GPU.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GridMind Server                             │
│  ┌─────────────────┐    ┌─────────────────────────────────────────┐ │
│  │  Socket.IO Hub  │◄──►│  Global Model Store (In-Memory)         │ │
│  │                 │    │  - currentWeights: Float32Array[]       │ │
│  │  Events:        │    │  - version: number                      │ │
│  │  • pull_weights │    │  - lastUpdated: timestamp               │ │
│  │  • push_weights │    └─────────────────────────────────────────┘ │
│  └────────┬────────┘                                                │
└───────────┼─────────────────────────────────────────────────────────┘
            │ WebSocket (bidirectional)
    ┌───────┴───────┬───────────────┬───────────────┐
    ▼               ▼               ▼               ▼
┌────────┐    ┌────────┐      ┌────────┐      ┌────────┐
│Client 1│    │Client 2│      │Client 3│      │Client N│
│ (GPU)  │    │ (GPU)  │      │ (CPU)  │      │ (GPU)  │
└────────┘    └────────┘      └────────┘      └────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd server
npm install
```

### 2. Start the Server

```bash
cd server
npm start
```

The server will start at `http://localhost:3000`

### 3. Open Multiple Browser Tabs

Open `http://localhost:3000` in multiple browser tabs/windows to simulate multiple federated clients.

### 4. Run Training

1. Click **"Initialize"** in each browser tab
2. Click **"Run 1 Round"** or **"Run 5 Rounds"** to start federated training
3. Watch as clients pull weights, train locally, and push averaged updates

## Federated Learning Flow

Each client executes the following async FL flow:

```
Step A (PULL):      Client fetches global weights (W_global_t0)
                          │
                          ▼
Step B (TRAIN):     Client trains locally for N epochs → W_local
                          │
                          ▼
Step C (RE-FETCH):  Client fetches latest global weights (W_global_t1)
                    (may have changed during training!)
                          │
                          ▼
Step D (AGGREGATE): Client performs FedAvg:
                    W_new = (W_local + W_global_t1) / 2
                          │
                          ▼
Step E (PUSH):      Client sends W_new to server
```

## Project Structure

```
/home/os/major-project/
├── server/
│   ├── server.js              # Express + Socket.IO server
│   ├── globalModelStore.js    # In-memory weight storage
│   ├── fedUtils.js            # Serialization with DP/quantization hooks
│   └── package.json
├── client/
│   ├── index.html             # UI with Chart.js visualization
│   ├── client.js              # FL orchestration (main logic)
│   ├── model.js               # TF.js MNIST model definition
│   ├── dataGenerator.js       # Synthetic data generation
│   └── fedClient.js           # FedAvg implementation
└── README.md
```

## Model Architecture

Lightweight CNN for MNIST-like digit classification (~15,000 parameters):

```
Input:    [28, 28, 1]    Grayscale image
Conv2D:   8 filters      3x3, ReLU
MaxPool:  2x2
Conv2D:   16 filters     3x3, ReLU
MaxPool:  2x2
Flatten:  [400]
Dense:    32 units       ReLU
Output:   10 classes     Softmax
```

## Key Features

### GPU Acceleration
- Uses TensorFlow.js with WebGL/WebGPU backends
- Automatic fallback to CPU if GPU unavailable

### Real-time Communication
- Socket.IO for bidirectional WebSocket communication
- Instant weight updates across all connected clients

### Async Federated Learning
- Handles concurrent updates with re-fetch strategy
- Last-write-wins concurrency model (configurable)

## Future-Proofing Hooks

The codebase includes injection points for:

### 1. Model Quantization
```javascript
// In fedClient.js
TRANSFORM_CONFIG.quantization = {
  enabled: true,
  targetPrecision: 'float16', // or 'int8'
};
```

### 2. Differential Privacy
```javascript
// In fedClient.js
TRANSFORM_CONFIG.differentialPrivacy = {
  enabled: true,
  epsilon: 1.0,
  delta: 1e-5,
  clipNorm: 1.0,
};
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | Server statistics |
| `/api/weights` | GET | Current global weights |
| `/api/reset` | POST | Reset global model |

## Socket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `pull_weights` | Client → Server | Request global weights |
| `push_weights` | Client → Server | Submit updated weights |
| `weights_updated` | Server → Clients | Broadcast weight updates |
| `welcome` | Server → Client | Connection confirmation |

## Configuration

Edit `CLIENT_CONFIG` in `client/client.js`:

```javascript
const CLIENT_CONFIG = {
  serverUrl: window.location.origin,
  localEpochs: 3,           // Epochs per round
  batchSize: 32,
  trainingSamples: 500,
  validationSamples: 100,
  fedAvgLocalWeight: 0.5,   // Weight for local vs global
};
```

## Development

### Run in Development Mode

```bash
cd server
npm run dev  # Uses --watch for auto-reload
```

### Test with Multiple Clients

1. Open http://localhost:3000 in Chrome
2. Open http://localhost:3000 in Firefox
3. Open http://localhost:3000 in incognito mode

Each browser/tab represents a separate federated client with its own local data.

## Tech Stack

- **Frontend**: Vanilla JS, TensorFlow.js, Socket.IO Client, Chart.js
- **Backend**: Node.js, Express, Socket.IO
- **ML Framework**: TensorFlow.js (WebGL/WebGPU accelerated)

## License

MIT
