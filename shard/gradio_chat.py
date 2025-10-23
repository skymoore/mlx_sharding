"""
Gradio-based chat interface for MLX sharding.
"""
import argparse
import grpc
from pathlib import Path
from typing import List, Generator
import mlx.core as mx
import gradio as gr

from .grpc import mlx_tensor_pb2_grpc, mlx_tensor_pb2
from .utils import load_model, create_generate_step_with_grpc
from mlx_lm.tokenizer_utils import load_tokenizer


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
        default="localhost:50051",
        help="Comma-separated list of gRPC server addresses (default: localhost:50051)"
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
    def __init__(self, model_path: str, shard_addresses: List[str]):
        self.model_path = model_path
        self.shard_addresses = shard_addresses
        
        # Load tokenizer
        tokenizer_path = Path(model_path) if Path(model_path).exists() else model_path
        self.tokenizer = load_tokenizer(tokenizer_path)
        
        # Connect to gRPC shards
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
        
        # Load model for first shard (if running locally)
        # This is optional - if all shards are remote, we don't need the model
        self.model = None
        try:
            self.model = load_model(model_path, start_layer=0, end_layer=30)
            print(f"✓ Local model loaded (layers 0-30)")
        except Exception as e:
            print(f"⚠ No local model loaded: {e}")
            print("  Assuming all processing happens on remote shards")
        
        # Create generate function
        if self.model:
            self.generate_step = create_generate_step_with_grpc(self.stubs)
        
        print(f"✓ Connected to {len(self.stubs)} shard(s)")
    
    def reset_cache(self):
        """Reset cache on all shards"""
        for stub in self.stubs:
            try:
                stub.ResetCache(mlx_tensor_pb2.ResetCacheRequest())
            except Exception as e:
                print(f"Warning: Failed to reset cache on shard: {e}")
    
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
        
        # Generate response
        response = ""
        try:
            for token, _ in zip(
                self.generate_step(
                    prompt_tokens,
                    self.model,
                    temp=temperature,
                    top_p=top_p,
                ),
                range(max_tokens)
            ):
                # Decode token
                decoded = self.tokenizer.decode([token])
                response += decoded
                yield {"role": "assistant", "content": response}
                
                # Check for EOS
                eos_ids = (
                    self.tokenizer.eos_token_id
                    if isinstance(self.tokenizer.eos_token_id, list)
                    else [self.tokenizer.eos_token_id]
                )
                if token in eos_ids:
                    break
        except Exception as e:
            yield {"role": "assistant", "content": f"Error generating response: {e}"}


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
    
    # Parse shard addresses
    shard_addresses = [addr.strip() for addr in args.llm_shard_addresses.split(',')]
    
    print("=" * 70)
    print("MLX Sharding - Gradio Chat Interface")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Shards: {', '.join(shard_addresses)}")
    print(f"Server: http://{args.host}:{args.port}")
    if args.share:
        print("Share: Enabled (public link will be generated)")
    print("=" * 70)
    
    # Initialize chatbot
    chatbot = ChatBot(args.model, shard_addresses)
    
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
