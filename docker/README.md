# Open WebUI Docker Setup for MLX Sharding

This Docker setup provides a modern chat interface (Open WebUI) that connects to your MLX sharding OpenAI-compatible API server.

## Prerequisites

1. **Docker and Docker Compose** installed on your system
2. **MLX Sharding API server** running on the host machine
3. **Shard servers** running (local and remote)

## Quick Start

### 1. Start Your MLX Sharding Infrastructure

**On Machine 2 (Remote - 64GB):**
```bash
cd ~/Develop/skymoore/mlx_sharding
poetry run python -m shard.main \
  --model /path/to/GLM-4.5-Air-MLX-4bit \
  --start-layer 31 \
  --end-layer 46 \
  --port 53332
```

**On Machine 1 (Local - 128GB):**
```bash
# Start OpenAI API server with local model
cd ~/Develop/skymoore/mlx_sharding
poetry run python -m shard.openai_api \
  --model /Users/sky/.lmstudio/models/lmstudio-community/GLM-4.5-Air-MLX-4bit \
  --start-layer 0 \
  --end-layer 30 \
  --llm-shard-addresses 10.0.33.32:53332 \
  --host 0.0.0.0 \
  --port 8080
```

### 2. Start Open WebUI Docker Container

```bash
cd ~/Develop/skymoore/mlx_sharding/docker
docker-compose up -d
```

### 3. Access Open WebUI

Open your browser and navigate to:
```
http://localhost:3000
```

**First Time Setup:**
1. Open WebUI will load and automatically detect the OpenAI API endpoint
2. No authentication required (disabled for local use)
3. Start chatting immediately!

## Configuration

### Environment Variables

You can customize the connection by editing `docker-compose.yml`:

```yaml
environment:
  - OPENAI_API_BASE_URLS=http://host.docker.internal:8080/v1  # API server URL
  - OPENAI_API_KEYS=dummy                                      # API key (not validated)
  - WEBUI_AUTH=false                                           # Disable authentication
```

### Custom Port

To run Open WebUI on a different port, edit `docker-compose.yml`:

```yaml
ports:
  - "3001:8080"  # Change 3001 to your desired port (container uses 8080 internally)
```

## Architecture

```
┌─────────────────┐
│  Open WebUI     │
│   (Docker)      │
│   Port 3000     │
└────────┬────────┘
         │ HTTP
         ↓
┌─────────────────┐
│  OpenAI API     │
│  Server         │
│  Port 8080      │
└────────┬────────┘
         │ Local Model (0-30)
         │ + gRPC
         ↓
┌─────────────────┐
│  Remote Shard   │
│  (31-46)        │
│  Port 53332     │
└─────────────────┘
```

## Troubleshooting

### Open WebUI can't connect to API server

**Check if API server is accessible:**
```bash
curl http://localhost:8080/v1/models
```

**Check Docker logs:**
```bash
docker-compose logs -f open-webui
```

### API server not responding

**Verify it's running:**
```bash
lsof -i :8080
```

**Check API server logs** in the terminal where you started it.

### Remote shard connection issues

**Test remote shard connectivity:**
```bash
nc -zv 10.0.33.32 53332
```

## Stopping the Services

```bash
# Stop Open WebUI
cd ~/Develop/skymoore/mlx_sharding/docker
docker-compose down

# To also remove the data volume:
docker-compose down -v

# Stop API server (Ctrl+C in its terminal)

# Stop remote shard (Ctrl+C on Machine 2)
```

## Advanced Usage

### Rebuild Container

If you make changes to the Dockerfile:
```bash
docker-compose build --no-cache
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f
```

### Access Container Shell

```bash
docker-compose exec open-webui sh
```

## Features

- **Modern Chat UI**: Beautiful, responsive interface inspired by ChatGPT
- **Streaming Responses**: Real-time token generation
- **Conversation History**: Maintains context across messages with persistent storage
- **Markdown Support**: Properly formatted code blocks, tables, and text
- **Code Highlighting**: Syntax highlighting for code blocks
- **Model Selection**: Switch between different models (if multiple configured)
- **Chat Management**: Create, rename, delete, and search conversations
- **Export/Import**: Save and share conversations
- **Dark/Light Mode**: Theme switching
- **No Authentication Required**: Disabled for local development

## Notes

- The `host.docker.internal` hostname allows the Docker container to access services running on the host machine
- The API key is set to "dummy" since the MLX API server doesn't validate keys by default
- Open WebUI automatically detects and connects to OpenAI-compatible endpoints
- Conversation data is persisted in a Docker volume (`open-webui`)
- Authentication is disabled for local development (`WEBUI_AUTH=false`)

## Alternative: Run Open WebUI Locally (Without Docker)

If you prefer not to use Docker:

```bash
pip install open-webui
export OPENAI_API_BASE_URLS=http://localhost:8080/v1
export OPENAI_API_KEYS=dummy
export WEBUI_AUTH=false
open-webui serve
```

Then open http://localhost:8080

## Additional Configuration

Open WebUI supports many configuration options. See the [official documentation](https://docs.openwebui.com/) for:
- Custom models
- RAG (Retrieval Augmented Generation)
- Function calling
- Image generation
- Voice input/output
- And much more!
