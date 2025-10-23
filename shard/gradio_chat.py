"""
Gradio-based chat interface for MLX sharding.
"""
import argparse
import logging
import grpc
from pathlib import Path
from typing import List, Generator
import mlx.core as mx
import gradio as gr

from .grpc import mlx_tensor_pb2_grpc, mlx_tensor_pb2
from .utils import load_model, create_generate_step_with_grpc
from mlx_lm.tokenizer_utils import load_tokenizer

# Setup logger
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio Chat Interface for MLX Sharding")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the MLX model"
    )
    parser.add_argument(
        "-s", "--llm-shard-addresses",
        type=str,
        default="",
        help="Comma-separated list of REMOTE gRPC server addresses (e.g., '192.168.1.100:50051'). Do NOT include localhost shard."
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=0,
        help="Start layer for local model (default: 0)"
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        default=30,
        help="End layer for local model (default: 30)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the Gradio server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the Gradio server (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    return parser.parse_args()


class ChatBot:
    def __init__(self, model_path: str, shard_addresses: List[str], local_layers: tuple = None):
        self.model_path = model_path
        self.shard_addresses = shard_addresses
        
        # Load tokenizer
        tokenizer_path = Path(model_path) if Path(model_path).exists() else model_path
        self.tokenizer = load_tokenizer(tokenizer_path)
        
        # Load LOCAL model for first layers
        # This is the key: Gradio runs on same machine as first shard
        if local_layers is None:
            local_layers = (0, 30)  # Default: layers 0-30
        
        print(f"Loading local model (layers {local_layers[0]}-{local_layers[1]})...")
        self.model = load_model(model_path, start_layer=local_layers[0], end_layer=local_layers[1])
        self.cache = self.model.make_cache()
        print(f"✓ Local model loaded: layers {local_layers[0]}-{local_layers[1]}")
        
        # Connect to REMOTE gRPC shards only (not the local one)
        channel_options = [
            ('grpc.max_metadata_size', 32 * 1024 * 1024),
            ('grpc.max_send_message_length', 128 * 1024 * 1024),
            ('grpc.max_receive_message_length', 128 * 1024 * 1024),
        ]
        
        self.stubs = []
        for address in shard_addresses:
            channel = grpc.insecure_channel(address.strip(), options=channel_options)
            stub = mlx_tensor_pb2_grpc.MLXTensorServiceStub(channel)
            self.stubs.append(stub)
        
        print(f"✓ Connected to {len(self.stubs)} remote shard(s)")
        for i, addr in enumerate(shard_addresses, 1):
            print(f"  Remote shard {i}: {addr}")
    
    def reset_cache(self):
        """Reset cache on local model and all remote shards"""
        # Reset local cache
        if hasattr(self.model, "make_cache"):
            self.cache = self.model.make_cache()
            print("✓ Local cache reset")
        
        # Reset remote shard caches
        for i, stub in enumerate(self.stubs, 1):
            try:
                stub.ResetCache(mlx_tensor_pb2.ResetCacheRequest())
                print(f"✓ Remote shard {i} cache reset")
            except Exception as e:
                print(f"Warning: Failed to reset cache on remote shard {i}: {e}")
    
    def chat(
        self,
        message: str,
        history: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> Generator[dict, None, None]:
        """
        Generate a response to the user message.
        
        Args:
            message: User's input message
            history: Chat history as list of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
        
        Yields:
            Partial responses as message dicts
        """
        if not message.strip():
            yield {"role": "assistant", "content": ""}
            return
        
        # Reset cache for new conversation turn
        self.reset_cache()
        
        # Build conversation from history
        conversation = history.copy()
        conversation.append({"role": "user", "content": message})
        
        # Format with chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback if chat template not available
            prompt = f"User: {message}\nAssistant:"
        
        # Tokenize
        tokens = self.tokenizer.encode(prompt)
        prompt_tokens = mx.array([tokens])
        
        # Generate response following original design:
        # 1. Process tokens through LOCAL model with cache
        # 2. Send hidden states to REMOTE shards via gRPC
        # 3. Get logits back and sample next token
        response = ""
        try:
            from .utils import tensor_to_bytes, response_to_mlx_array
            from .grpc import mlx_tensor_pb2
            
            # Start with prompt tokens
            y = mx.array([tokens])
            
            for step in range(max_tokens):
                # Step 1: Process through LOCAL model (layers 0-30)
                logger.debug(f"=== Generation step {step} ===")
                logger.debug(f"Local input: shape={y.shape}, dtype={y.dtype}")
                
                hidden_states = self.model(y, cache=self.cache)
                logger.debug(f"Local output (hidden states): shape={hidden_states.shape}, dtype={hidden_states.dtype}")
                
                # Convert bfloat16 to float16 for gRPC transmission (matches generate.py)
                if hidden_states.dtype == mx.bfloat16:
                    hidden_states = hidden_states.astype(mx.float16)
                    logger.debug("Converted to float16 for transmission")
                
                # Step 2: Send hidden states through REMOTE shards
                # IMPORTANT: On first step, send ALL hidden states to populate remote cache
                # On subsequent steps, send only LAST token's hidden states
                if step == 0:
                    output = hidden_states  # Send all tokens on first step
                    logger.debug(f"First step: sending all hidden states: shape={output.shape}")
                else:
                    output = hidden_states[:, -1:, :]  # Send only last token
                    logger.debug(f"Subsequent step: sending last token hidden states: shape={output.shape}")
                
                for i, stub in enumerate(self.stubs, 1):
                    from .utils import tensor_to_bytes, response_to_mlx_array
                    from .grpc import mlx_tensor_pb2
                    
                    logger.debug(f"Sending to remote shard {i}: shape={output.shape}, dtype={output.dtype}")
                    
                    tensor_msg = mlx_tensor_pb2.Tensor(
                        tensor_data=tensor_to_bytes(output),
                        shape=list(output.shape),
                        dtype=str(output.dtype)
                    )
                    response_msg = stub.SendTensor(tensor_msg)
                    output = response_to_mlx_array(response_msg)
                    
                    if output is None:
                        raise ValueError(f"Remote shard {i} returned None")
                    
                    logger.debug(f"Received from remote shard {i}: shape={output.shape}, dtype={output.dtype}")
                
                # Step 3: Get logits from final output
                logits = output[:, -1, :]
                logger.debug(f"Logits shape: {logits.shape}")
                logger.debug(f"Logits stats: min={float(logits.min()):.4f}, max={float(logits.max()):.4f}, mean={float(logits.mean()):.4f}, std={float(logits.std()):.4f}")
                
                # Step 4: Sample next token
                if temperature == 0:
                    next_token = mx.argmax(logits, axis=-1)
                else:
                    if top_p > 0 and top_p < 1.0:
                        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                        from mlx_lm.sample_utils import apply_top_p
                        modified_logprobs = apply_top_p(logprobs, top_p)
                        next_token = mx.random.categorical(modified_logprobs * (1 / temperature))
                    else:
                        next_token = mx.random.categorical(logits * (1 / temperature))
                
                token_id = next_token.item()
                logger.debug(f"Sampled token: {token_id}")
                
                # Decode and yield
                decoded = self.tokenizer.decode([token_id])
                response += decoded
                yield {"role": "assistant", "content": response}
                
                # Check for EOS
                eos_ids = (
                    self.tokenizer.eos_token_id
                    if isinstance(self.tokenizer.eos_token_id, list)
                    else [self.tokenizer.eos_token_id]
                )
                if token_id in eos_ids:
                    logger.debug("EOS token reached")
                    break
                
                # Step 5: Prepare next input (just the new token)
                # Local cache will handle efficiency
                y = mx.array([[token_id]])
                
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logger.error(error_msg, exc_info=True)
            yield {"role": "assistant", "content": f"Error: {e}"}


def create_interface(chatbot: ChatBot, share: bool = False):
    """Create and launch the Gradio interface"""
    
    with gr.Blocks(
        title="MLX Sharding Chat",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🚀 MLX Sharding Chat
            
            Distributed LLM inference across multiple machines using MLX.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot_ui = gr.Chatbot(
                    label="Chat",
                    height=600,
                    show_copy_button=True,
                    type="messages",
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=4,
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear = gr.Button("Clear Chat")
                    retry = gr.Button("Retry")
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative"
                )
                
                max_tokens = gr.Slider(
                    minimum=1,
                    maximum=2048,
                    value=512,
                    step=1,
                    label="Max Tokens",
                    info="Maximum response length"
                )
                
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P",
                    info="Nucleus sampling"
                )
                
                gr.Markdown(
                    f"""
                    ### 📊 Info
                    
                    **Model:** `{Path(chatbot.model_path).name}`
                    
                    **Shards:** {len(chatbot.shard_addresses)}
                    
                    **Addresses:**
                    """
                )
                
                for i, addr in enumerate(chatbot.shard_addresses, 1):
                    gr.Markdown(f"- Shard {i}: `{addr}`")
        
        # Event handlers
        def user_message(message, history):
            return "", history + [{"role": "user", "content": message}]
        
        def bot_response(history, temperature, max_tokens, top_p):
            if not history or history[-1]["role"] != "user":
                return history
            
            message = history[-1]["content"]
            history_without_last = history[:-1]
            
            for partial_msg in chatbot.chat(
                message,
                history_without_last,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            ):
                # Update or append assistant message
                if history[-1]["role"] == "user":
                    yield history + [partial_msg]
                else:
                    history[-1] = partial_msg
                    yield history
        
        def clear_chat():
            chatbot.reset_cache()
            return []
        
        def retry_last(history):
            if not history:
                return history
            # Remove last assistant message if present
            if history[-1]["role"] == "assistant":
                history = history[:-1]
            return history
        
        # Wire up events
        msg.submit(
            user_message,
            [msg, chatbot_ui],
            [msg, chatbot_ui],
            queue=False
        ).then(
            bot_response,
            [chatbot_ui, temperature, max_tokens, top_p],
            chatbot_ui
        )
        
        submit.click(
            user_message,
            [msg, chatbot_ui],
            [msg, chatbot_ui],
            queue=False
        ).then(
            bot_response,
            [chatbot_ui, temperature, max_tokens, top_p],
            chatbot_ui
        )
        
        clear.click(clear_chat, None, chatbot_ui, queue=False)
        
        retry.click(
            retry_last,
            chatbot_ui,
            chatbot_ui,
            queue=False
        ).then(
            bot_response,
            [chatbot_ui, temperature, max_tokens, top_p],
            chatbot_ui
        )
    
    return demo


def main():
    args = parse_args()
    
    # Parse remote shard addresses
    shard_addresses = []
    if args.llm_shard_addresses:
        shard_addresses = [addr.strip() for addr in args.llm_shard_addresses.split(',') if addr.strip()]
    
    print("=" * 70)
    print("MLX Sharding - Gradio Chat Interface")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Local layers: {args.start_layer}-{args.end_layer}")
    if shard_addresses:
        print(f"Remote shards: {', '.join(shard_addresses)}")
    else:
        print("Remote shards: None (local-only mode)")
    print(f"Server: http://{args.host}:{args.port}")
    if args.share:
        print("Share: Enabled (public link will be generated)")
    print("=" * 70)
    
    # Initialize chatbot
    local_layers = (args.start_layer, args.end_layer)
    chatbot = ChatBot(args.model, shard_addresses, local_layers=local_layers)
    
    # Create and launch interface
    demo = create_interface(chatbot, share=args.share)
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
