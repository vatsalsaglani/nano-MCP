import asyncio
import json
import os
import sys
import traceback
from typing import List, Dict
from dotenv import load_dotenv
from llm_mcp_client import LLMMCPClient

load_dotenv()


async def interactive_cli():
    """Interactive CLI for the MCP client"""

    print("\n=== MCP CLI Interface ===")
    print("Type a message to chat with the assistant.")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'tools' to see available tools.")
    print("Type 'history' to see conversation history.")
    print("===========================\n")

    async with LLMMCPClient() as mcp_client:
        print(f"MCP client initialized with ID: {mcp_client.client_id}")
        print("Fetching available tools...")

        if not mcp_client.tools:
            print("Warning: No tools available.")
        else:
            print(f"Available tools: {len(mcp_client.tools)}")

        try:
            while True:
                # Get user input
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break

                if user_input.lower() == "tools":
                    # Print available tools
                    print("\nAvailable Tools:")
                    for tool in mcp_client.tools:
                        print(
                            f"- {tool.get('name')}: {tool.get('description')}")
                    continue

                if user_input.lower() == "history":
                    # Print conversation history
                    print("\nConversation History:")
                    for i, msg in enumerate(mcp_client.messages):
                        if msg["role"] == "system" and i == 0:
                            print("[System Prompt]")
                        elif msg["role"] == "system":
                            print(f"[System]: {msg['content'][:100]}...")
                        else:
                            print(
                                f"[{msg['role'].title()}]: {msg['content'][:100]}..."
                            )
                    continue

                # Stream chat response
                print("\nAssistant: ", end="", flush=True)
                # async for chunk in mcp_client.stream_chat(user_input):
                async for chunk in mcp_client.interactive_stream_chat(
                    user_input):
                    if chunk:
                        print(chunk, end="", flush=True)

                print()  # Extra newline for readability

        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print(f"\n{traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(interactive_cli())
