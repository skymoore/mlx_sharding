# Guide: Sharding GLM-4.5-Air-MLX-4bit Across Two Machines

## Overview

- **Model**: GLM-4.5-Air-MLX-4bit (46 layers, ~60GB total)
- **Machine 1** (128GB RAM): Layers 0-30 + API server (~40GB)
- **Machine 2** (64GB RAM): Layers 30-46 (~20GB)

## Architecture

```
Client → API Server (Machine 1) → Shard 1 (Machine 1, layers 0-30) 
                                → Shard 2 (Machine 2, layers 30-46) → Response
```

---

## Machine 2 Setup (64GB - Remote Shard)

### 1. Install Dependencies

```bash
cd ~/Develop/skymoore/mlx_sharding
git pull sky update
poetry install
```

### 2. Copy Model Files

You need the model on Machine 2. Either:

- **Option A**: Copy from Machine 1

  ```bash
  # On Machine 2
  rsync -avz --progress sky@machine1:/Users/sky/.lmstudio/models/lmstudio-community/GLM-4.5-Air-MLX-4bit/ \
    /Users/sky/.lmstudio/models/lmstudio-community/GLM-4.5-Air-MLX-4bit/
  ```

- **Option B**: Download directly on Machine 2 (if you have it in HuggingFace or LM Studio)

### 3. Start Shard Server (Layers 30-46)

```bash
cd ~/Develop/skymoore/mlx_sharding
poetry run python -m shard.main \
  --model /Users/sky/.lmstudio/models/lmstudio-community/GLM-4.5-Air-MLX-4bit \
  --start-layer 30 \
  --end-layer 46
```

**Expected output:**

```
Server started on port 50051
Model loaded: layers 30-46
```

**Note the IP address** of Machine 2 (e.g., `192.168.1.100`)

---

## Machine 1 Setup (128GB - Local with API)

### 1. Start Shard Server (Layers 0-30)

```bash
cd ~/Develop/skymoore/mlx_sharding
poetry run python -m shard.main \
  --model /Users/sky/.lmstudio/models/lmstudio-community/GLM-4.5-Air-MLX-4bit \
  --start-layer 0 \
  --end-layer 30
```

**Expected output:**

```
Server started on port 50051
Model loaded: layers 0-30
```

### 2. Start Chat Interface

In a **new terminal** on Machine 1, you have two options:

#### Option A: Gradio Chat UI (Recommended - Modern & Beautiful)

```bash
cd ~/Develop/skymoore/mlx_sharding
poetry run python -m shard.gradio_chat \
  --model /Users/sky/.lmstudio/models/lmstudio-community/GLM-4.5-Air-MLX-4bit \
  --start-layer 0 \
  --end-layer 30 \
  -s 192.168.1.100:50051
```

**Important:** 
- `--start-layer` and `--end-layer` define the LOCAL model layers (same as the local shard server)
- `-s` should ONLY include REMOTE shard addresses (NOT localhost)
- Gradio loads the model locally and sends hidden states to remote shards

**Expected output:**

```
Loading local model (layers 0-30)...
✓ Local model loaded: layers 0-30
✓ Connected to 1 remote shard(s)
  Remote shard 1: 192.168.1.100:50051
Running on local URL:  http://127.0.0.1:7860
```

Open http://127.0.0.1:7860 in your browser for a beautiful chat interface!

#### Option B: OpenAI-Compatible API Server (Pure API - No Web UI)

```bash
cd ~/Develop/skymoore/mlx_sharding
poetry run python -m shard.openai_api \
  --model /Users/sky/.lmstudio/models/lmstudio-community/GLM-4.5-Air-MLX-4bit \
  --start-layer 0 \
  --end-layer 30 \
  -s 192.168.1.100:50051 \
  --host 0.0.0.0 \
  --port 8080
```

**Important:**
- `--start-layer` and `--end-layer` define the LOCAL model layers
- `-s` should ONLY include REMOTE shard addresses (NOT localhost)
- API server loads the model locally and sends hidden states to remote shards
- `--host 0.0.0.0` allows external connections (for Docker/TinyChat)

**Expected output:**

```
Connected to 1 LLM shard(s)
Loading model with layers 0 to 30
OpenAI API endpoint: http://0.0.0.0:8080/v1
Health check: http://0.0.0.0:8080/health
```

**API Endpoints:**
- `POST /v1/chat/completions` - Chat completions (streaming & non-streaming)
- `POST /v1/completions` - Text completions
- `GET /v1/models` - List available models
- `GET /health` - Health check

**Note**: The web UI has been removed. Use TinyChat (Docker) or any OpenAI-compatible client to interact with the API.

---

## Testing the Setup

### Test 1: Simple Generation

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.5-air",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### Test 2: Using the generate.py Script

```bash
poetry run python generate.py \
  --model /Users/sky/.lmstudio/models/lmstudio-community/GLM-4.5-Air-MLX-4bit \
  --prompt "Write a short poem about AI" \
  --max_tokens 100 \
  --server_address "localhost:50051,192.168.1.100:50051"
```

---

## Troubleshooting

### Issue: "Connection refused" to Machine 2

**Check firewall on Machine 2:**

```bash
# Allow port 50051
sudo ufw allow 50051
# Or on macOS
sudo pfctl -d  # Disable firewall temporarily for testing
```

### Issue: "Model not found"

Verify the model path exists on both machines:

```bash
ls /Users/sky/.lmstudio/models/lmstudio-community/GLM-4.5-Air-MLX-4bit/
```

### Issue: Out of memory

- Machine 2 (64GB) should be fine with layers 30-46 (~20GB)
- Machine 1 (128GB) should be fine with layers 0-30 (~40GB) + API server
- If issues persist, adjust layer split (e.g., 0-25 and 25-46)

### Check Server Status

```bash
# On each machine, check if server is running
lsof -i :50051
```

---

## Performance Tips

1. **Use same network**: Ensure both machines are on the same local network for low latency
2. **Monitor memory**: Use `htop` or Activity Monitor to watch RAM usage
3. **Adjust batch size**: If generating multiple requests, adjust accordingly

---

## Stopping the Servers

Press `Ctrl+C` in each terminal running the servers.

---

## Summary

**Machine 2 (64GB):**

```bash
poetry run python -m shard.main --model <path> --start-layer 30 --end-layer 46
```

**Machine 1 (128GB) - Terminal 1:**

```bash
poetry run python -m shard.main --model <path> --start-layer 0 --end-layer 30
```

**Machine 1 (128GB) - Terminal 2:**

```bash
poetry run python -m shard.openai_api --model <path> -s localhost:50051,<machine2-ip>:50051
```

That's it! You now have a distributed inference setup with an OpenAI-compatible API.
