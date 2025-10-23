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

### 2. Start API Server

In a **new terminal** on Machine 1:

```bash
cd ~/Develop/skymoore/mlx_sharding
poetry run python -m shard.openai_api \
  --model /Users/sky/.lmstudio/models/lmstudio-community/GLM-4.5-Air-MLX-4bit \
  -s localhost:50051,192.168.1.100:50051
```

Replace `192.168.1.100` with Machine 2's actual IP address.

**Note**: Use `-s` or `--llm-shard-addresses` for the shard server addresses.

**Expected output:**

```
API server running on http://localhost:8080
Connected to shards: localhost:50051, 192.168.1.100:50051
```

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
