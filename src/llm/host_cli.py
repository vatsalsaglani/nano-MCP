#!/usr/bin/env python
import asyncio
import argparse
import os
import json
import sys
from typing import Optional, Dict, Any, List

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.live import Live
from rich.table import Table
from rich.box import ROUNDED
from rich.text import Text
from rich.spinner import Spinner
from rich.layout import Layout
from rich.console import Group

from llm_mcp_client import LLMMCPClient

console = Console()


def create_parser():
    parser = argparse.ArgumentParser(
        description="CLI interface for MCP Assistant")
    parser.add_argument("--api-key", help="OpenAI/Groq API key")
    parser.add_argument("--base-url",
                        help="API base URL (e.g., https://api.openai.com/v1)")
    parser.add_argument("--model", help="Model name (e.g., gpt-4-turbo)")
    parser.add_argument("--host-model",
                        choices=["openai", "groq"],
                        help="Host model type")
    parser.add_argument("--mcp-url",
                        help="MCP API URL (e.g., http://localhost:8001)")
    return parser


def get_config_from_env():
    return {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
        "model_name": os.environ.get("MODEL_NAME"),
        "host_model": os.environ.get("HOST_MODEL"),
        "mcp_client_url": os.environ.get("MCP_CLIENT_URL"),
    }


def prompt_for_missing_config(config):

    if not config.get("openai_api_key"):
        config["openai_api_key"] = Prompt.ask("Enter API Key",
                                              #   password=True
                                              )
    if not config.get("model_name"):
        config["model_name"] = Prompt.ask("Enter Model Name",
                                          default="gpt-4-turbo")
    if not config.get("host_model"):
        options = ["openai", "groq"]
        for i, opt in enumerate(options):
            console.print(f"[cyan]{i+1}.[/cyan] [green]{opt}[/green]")
        choice = console.input(
            "[bold yellow]Select Host Model Type:[/bold yellow] ")
        try:
            index = int(choice) - 1
            if 0 <= index < len(options):
                config["host_model"] = options[index]
            else:
                config["host_model"] = "openai"  # Default if invalid
        except ValueError:
            config["host_model"] = "openai"  # Default if not a number

    if not config.get("openai_base_url"):
        config["openai_base_url"] = Prompt.ask(
            "Enter Base URL", default="https://api.openai.com/v1")

    if not config.get("mcp_client_url"):
        config["mcp_client_url"] = Prompt.ask("Enter MCP Client URL",
                                              default="http://localhost:8001")

    return config


def display_user_message(message):
    console.print(
        Panel(message, border_style="blue", title="You", title_align="left"))


def display_system_message(message):
    console.print(
        Panel(message,
              border_style="yellow",
              title="System",
              title_align="left"))


def format_tool_call(tool_call):
    tool_name = tool_call.get("name", "Unknown Tool")
    arguments = json.dumps(tool_call.get("arguments", {}), indent=2)

    table = Table(box=ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Tool Call", style="cyan")
    table.add_column("Details", style="green")
    table.add_row("Tool Name", tool_name)

    return Group(table,
                 Syntax(arguments, "json", theme="monokai", line_numbers=True))


def format_tool_result(tool_result):
    tool_name = tool_result.get("name", "Unknown Tool")
    status = tool_result.get("status", "unknown")
    header_style = "green" if status == "success" else "red"

    table = Table(box=ROUNDED,
                  show_header=True,
                  header_style=f"bold {header_style}")
    table.add_column("Tool Result", style="cyan")
    table.add_column("Details", style=header_style)
    table.add_row("Tool Name", tool_name)
    table.add_row("Status", status)

    data = tool_result.get("data", {})
    formatted_data = json.dumps(data, indent=2)

    return Group(
        table,
        Syntax(formatted_data, "json", theme="monokai", line_numbers=True))


def parse_chunk(chunk):
    if not chunk:
        return None, None

    if chunk == "[STREAM_END]":
        return "STREAM_END", None

    tag_match = chunk.split("]", 1)
    if len(tag_match) == 2 and tag_match[0].startswith("["):
        tag = tag_match[0][1:]  # Remove the opening bracket
        payload = tag_match[1]
        return tag, payload

    return None, chunk


async def main():
    parser = create_parser()
    args = parser.parse_args()

    config = get_config_from_env()

    if args.api_key:
        config["openai_api_key"] = args.api_key
    if args.base_url:
        config["openai_base_url"] = args.base_url
    if args.model:
        config["model_name"] = args.model
    if args.host_model:
        config["host_model"] = args.host_model
    if args.mcp_url:
        config["mcp_client_url"] = args.mcp_url

    config = {k: v for k, v in config.items() if v is not None}

    config = prompt_for_missing_config(config)

    console.print("\n[bold cyan]MCP Assistant CLI[/bold cyan]")
    console.print(
        "Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to end the session\n"
    )

    async with LLMMCPClient(**config) as mcp_client:
        display_system_message(
            f"Connected with {config['host_model']} using model {config['model_name']}"
        )

        while True:

            console.print("\n[bold blue]You:[/bold blue] ", end="")
            user_message = console.input("")

            if user_message.lower() in ("exit", "quit"):
                break

            display_user_message(user_message)

            llm_buffer = ""
            is_streaming_done = False
            rendered_content = []
            has_llm_content = False

            with Live(
                    Spinner("dots", text="Thinking..."),
                    refresh_per_second=10,
                    transient=False  # Keep content after Live context exits
            ) as live:
                async for chunk in mcp_client.interactive_stream_chat(
                        user_message):
                    tag, payload = parse_chunk(chunk)

                    if tag == "LLM" and payload:
                        has_llm_content = True
                        llm_buffer += payload

                        try:
                            md_content = Markdown(llm_buffer)
                            live.update(
                                Panel(md_content,
                                      title="Assistant",
                                      title_align="left",
                                      border_style="green"))
                        except Exception:

                            live.update(
                                Panel(llm_buffer,
                                      title="Assistant",
                                      title_align="left",
                                      border_style="green"))

                    elif tag == "TOOL_CALL" and payload:

                        try:
                            tool_call = json.loads(payload)
                            tool_content = format_tool_call(tool_call)
                            rendered_content.append(tool_content)

                            if has_llm_content:
                                full_content = Group(
                                    Markdown(llm_buffer) if llm_buffer else "",
                                    *rendered_content)
                            else:
                                full_content = Group(*rendered_content)

                            live.update(
                                Panel(full_content,
                                      title="Assistant",
                                      title_align="left",
                                      border_style="green"))
                        except json.JSONDecodeError:
                            display_system_message(
                                f"Failed to parse tool call: {payload}")

                    elif tag == "TOOL_RESULT" and payload:
                        try:
                            tool_result = json.loads(payload)
                            tool_content = format_tool_result(tool_result)
                            rendered_content.append(tool_content)

                            if has_llm_content:
                                full_content = Group(
                                    Markdown(llm_buffer) if llm_buffer else "",
                                    *rendered_content)
                            else:
                                full_content = Group(*rendered_content)

                            live.update(
                                Panel(full_content,
                                      title="Assistant",
                                      title_align="left",
                                      border_style="green"))
                        except json.JSONDecodeError:
                            display_system_message(
                                f"Failed to parse tool result: {payload}")

                    elif tag == "SYSTEM" and payload:

                        live.stop()
                        display_system_message(payload)
                        live.start()

                    elif tag == "ERROR" and payload:

                        live.stop()
                        display_system_message(f"Error: {payload}")
                        live.start()

                    elif tag == "STREAM_END":
                        is_streaming_done = True

                        break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Session terminated by user[/bold red]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)
