import os
import asyncio
import gradio as gr
from typing import Tuple
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from SK_SmartHome import SmartHomePlugin

class SmartHomeInterface:
    def __init__(self):
        self.kernel = None
        self.plugin = None
        self.arguments = None
        self.setup_kernel()
    
    def setup_kernel(self):
        """Initialize the Semantic Kernel and SmartHome plugin"""
        # Azure OpenAI configuration
        AZURE_OPENAI_ENDPOINT = "https://devinnovationo8144129163.openai.azure.com/"
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
        
        # Initialize kernel
        self.kernel = Kernel()
        service = AzureChatCompletion(
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )
        self.kernel.add_service(service)
        
        # Register SmartHome plugin
        self.plugin = SmartHomePlugin()
        self.kernel.add_plugin(self.plugin, plugin_name="home")
        
        # Setup arguments for auto function calling
        self.arguments = KernelArguments(
            settings=PromptExecutionSettings(
                function_choice_behavior=FunctionChoiceBehavior.Auto(
                    filters={"included_plugins": ["home"]}
                ),
            )
        )
    
    async def process_prompt(self, prompt: str) -> Tuple[str, str]:
        """Process user prompt with the SmartHome agent"""
        if not prompt.strip():
            return "Please enter a command.", self.get_status()
        
        try:
            # Invoke the kernel with the user prompt
            response = await self.kernel.invoke_prompt(prompt, arguments=self.arguments)
            agent_response = str(response)
            
            # Get current status
            current_status = self.get_status()
            
            return agent_response, current_status
            
        except Exception as e:
            return f"Error: {str(e)}", self.get_status()
    
    def get_status(self) -> str:
        """Get current smart home status"""
        if self.plugin:
            return self.plugin._state.snapshot()
        return "Plugin not initialized"

# Initialize the SmartHome interface
smart_home = SmartHomeInterface()

def gradio_interface(prompt):
    """Gradio interface function - handles the async call"""
    try:
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agent_response, status = loop.run_until_complete(smart_home.process_prompt(prompt))
        loop.close()
        
        return agent_response, status
    except Exception as e:
        return f"Error: {str(e)}", smart_home.get_status()

# Create Gradio interface
def create_ui():
    with gr.Blocks(title="Smart Home Control", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏠 Smart Home Control Panel")
        gr.Markdown("Enter commands to control your smart home devices. Examples:")
        gr.Markdown("- 'Turn on the living room lights'")
        gr.Markdown("- 'Set temperature to 22 degrees'")
        gr.Markdown("- 'Lock the doors and play jazz music'")
        gr.Markdown("- 'What's the current status?'")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Enter your command",
                    placeholder="e.g., Turn on the kitchen lights and set temperature to 24",
                    lines=2
                )
                submit_btn = gr.Button("Send Command", variant="primary")
                
                agent_response = gr.Textbox(
                    label="Agent Response",
                    interactive=False,
                    lines=3
                )
            
            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="🏠 Current Home Status",
                    interactive=False,
                    lines=6,
                    value=smart_home.get_status()
                )
        
        # Event handlers
        submit_btn.click(
            fn=gradio_interface,
            inputs=[prompt_input],
            outputs=[agent_response, status_output]
        )
        
        prompt_input.submit(
            fn=gradio_interface,
            inputs=[prompt_input],
            outputs=[agent_response, status_output]
        )
        
        # Clear input after submission
        def clear_input():
            return ""
        
        submit_btn.click(fn=clear_input, outputs=[prompt_input])
        prompt_input.submit(fn=clear_input, outputs=[prompt_input])
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True  # Create a public link
    )