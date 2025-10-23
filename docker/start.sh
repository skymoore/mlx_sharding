#!/bin/bash

# Open WebUI Docker Quick Start Script

echo "======================================================================"
echo "Open WebUI Docker Setup for MLX Sharding"
echo "======================================================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Check if API server is accessible
echo "Checking if OpenAI API server is running on port 8080..."
if curl -s http://localhost:8080 > /dev/null 2>&1; then
    echo "✓ API server is accessible"
else
    echo "⚠️  Warning: API server not detected on port 8080"
    echo "   Make sure to start the API server before using TinyChat:"
    echo ""
    echo "   poetry run python -m shard.openai_api \\"
    echo "     --model /path/to/model \\"
    echo "     --start-layer 0 --end-layer 30 \\"
    echo "     --llm-shard-addresses REMOTE_IP:PORT \\"
    echo "     --host 0.0.0.0 --port 8080"
    echo ""
fi

# Pull and start the container
echo ""
echo "Pulling and starting Open WebUI container..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Open WebUI is now running!"
    echo "======================================================================"
    echo ""
    echo "Access Open WebUI at: http://localhost:3000"
    echo ""
    echo "Features:"
    echo "  • Modern ChatGPT-like interface"
    echo "  • Streaming responses"
    echo "  • Conversation history"
    echo "  • Code highlighting"
    echo "  • Dark/Light mode"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f open-webui"
    echo ""
    echo "To stop Open WebUI:"
    echo "  docker-compose down"
    echo ""
else
    echo ""
    echo "❌ Error: Failed to start Open WebUI container"
    echo "   Check the logs with: docker-compose logs"
    exit 1
fi
