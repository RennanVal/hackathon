import os
import asyncio
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from dotenv import load_dotenv
load_dotenv()

# ===== Azure OpenAI (env) =====
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


async def main():
    # Check for required environment variables
    missing = [k for k, v in {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT
    }.items() if not v]
    
    if missing:
        raise RuntimeError(f"Set the environment variables: {', '.join(missing)}")

    # Create Azure OpenAI service
    service = AzureChatCompletion(
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )

    # Create the joke agent
    agent = ChatCompletionAgent(
        service=service,
        name="JokeAgent",
        instructions=(
            "You are a friendly comedian and joke teller. Your job is to create and share jokes "
            "that are appropriate, clean, and entertaining. You can tell different types of jokes "
            "including puns, one-liners, knock-knock jokes, and short humorous stories. "
            "Always aim to bring a smile to people's faces with your humor. "
            "Keep your jokes light-hearted and suitable for all audiences."
        ),
    )

    user_input = input("\nYou: ").strip()

    try:
        # Get response from the agent
        reply = await agent.get_response(user_input)
        print(f"\nJokeAgent: {str(reply)}")
    
    except Exception as e:
        print(f"Oops! Something went wrong: {e}")


if __name__ == "__main__":
    asyncio.run(main())