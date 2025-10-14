# sk_mcp_github_remote_azure.py
import os
import asyncio

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.mcp import MCPStdioPlugin

# ===== Azure OpenAI (env) =====
AZURE_OPENAI_ENDPOINT   = "https://devinnovationo8144129163.openai.azure.com/"
AZURE_OPENAI_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"

# ===== GitHub Token =====
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

async def main():
    # docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server
    async with MCPStdioPlugin(
        name="Github",
        description="Github Plugin",
        command="docker",
        args=["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN},
    ) as github_plugin:
    #
        service = AzureChatCompletion(
                deployment_name=AZURE_OPENAI_DEPLOYMENT,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
            )
        
        agent = ChatCompletionAgent(
            service=service,
            name="GitHubAgent",
            instructions=(
                "You are a GitHub analyst. Use the GitHub MCP tools when helpful. "
                "If the repository is private, authenticate via the MCP server. "
            ),
            plugins=[github_plugin],   # <- THIS is what enables tool calling
        )

        user_question = f"Summarize the last commits of my repository RennanVal/hackathon add links."
        reply = await agent.get_response(user_question)
    
        print("\n===== AGENT RESPONSE =====\n")
        print(str(reply))


if __name__ == "__main__":
    asyncio.run(main())
