# nano-MCP: Model Context Protocol Implementation

An implementation of the Model Context Protocol (MCP) for tool-augmented Language Models, enabling LLMs like GPT-4.1 and Llama models to use external tools to complete tasks autonomously.

## Architecture

This project follows the Model Context Protocol (MCP) architectural pattern:

- **LLM Integration**: Logic for interacting with LLM APIs (OpenAI, Claude, etc.). Sends prompts, parses tool call outputs, and routes tool calls to MCP client.
- **MCP Client**: Routes tool calls to the appropriate MCP servers.
- **MCP Servers**: Dockerized FastAPI applications exposing tool APIs:
  - File Management Server (read, write, tree operations)
  - Version Control Server (git operations, command execution)

## Components

- `src/llm/llm_mcp_client.py`: The core client for LLM integration with MCP tools
- `src/llm/host_cli.py`: Command-line interface for interacting with the LLM+MCP system
- `src/llm/host.py`: FastAPI server providing a web interface at `/`

## Usage

### Environment Variables

Configure the following environment variables:

```
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4-turbo
HOST_MODEL=openai
MCP_CLIENT_URL=http://localhost:8001
```

### CLI Interface

Run the CLI interface:

```bash
cd src/llm
python host_cli.py
```

You can also provide configuration via command-line arguments:

```bash
python host_cli.py --api-key YOUR_API_KEY --model gpt-4-turbo --mcp-url http://localhost:8001
```

### Web Interface

Start the web server:

```bash
cd src/llm
python host.py
```

The web interface will be available at http://localhost:7899/

## Docker Setup

Create a `shared_data` folder in the `servers` folder.

The project uses Docker Compose to orchestrate the MCP servers:

```bash
docker-compose up -d
```

This will start:
- File Management Server on port 8000
- Version Control Server on port 8001

## Protocol Details

The Model Context Protocol (MCP) allows LLMs to:

1. Discover available tools from MCP servers
2. Call tools with proper arguments
3. Receive and process tool results
4. Continue execution in an autonomous loop

Each MCP server exposes two main endpoints:
- `GET /mcp/tools`: Lists available tools with schemas
- `POST /mcp/call`: Executes tools based on name and input

## Development

To add new tools, create a new server in the `servers/` directory implementing the MCP protocol endpoints.
