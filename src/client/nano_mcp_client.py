from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from httpx import AsyncClient
from typing import List, Optional, Dict, Any, Union
import os
import json
import uuid
import datetime
import asyncio
import aiofiles
from dotenv import load_dotenv
import traceback

load_dotenv()

MCP_SERVER_URLS = os.getenv("MCP_SERVER_URLS").split(",")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")

os.makedirs(LOGS_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models for API requests and responses
class ToolSchema(BaseModel):
    tool: str
    input: Union[Dict[str, Any], None]


class ExecuteToolRequest(BaseModel):
    tool: str
    input: Union[Dict[str, Any], None] = None
    client_id: Optional[str] = None


class ToolResponse(BaseModel):
    result: Any


class ErrorResponse(BaseModel):
    message: str


class MCPLogger:

    @staticmethod
    async def log_message(client_id: str, message: Dict[str, Any],
                          message_type: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_filename = f"{client_id}-{timestamp}.log"
        log_path = os.path.join(LOGS_DIR, log_filename)

        log_entry = {
            "timestamp": timestamp,
            "client_id": client_id,
            "type": message_type,
            "content": message
        }

        async with aiofiles.open(log_path, "a") as f:
            await f.write(json.dumps(log_entry, indent=4) + "\n")


async def fetch_tools_from_servers() -> List[Dict[str, Any]]:
    all_tools = []
    server_tools_map = {}

    for server_url in MCP_SERVER_URLS:
        if not server_url.strip():
            continue

        try:
            async with AsyncClient() as client:
                response = await client.get(f"{server_url}/list/tools")
                if response.status_code == 200:
                    tools = response.json()
                    for tool in tools:
                        tool_name = tool.get("name")
                        server_tools_map[tool_name] = server_url
                        all_tools.append(tool)
        except Exception as e:
            print(
                f"Error fetching tools from {server_url}: {str(e)}\n{traceback.format_exc()}"
            )

    return all_tools, server_tools_map


tools_cache = None
server_tools_mapping = {}


async def get_tools():
    global tools_cache, server_tools_mapping
    if tools_cache is None:
        tools_cache, server_tools_mapping = await fetch_tools_from_servers()
    return tools_cache, server_tools_mapping


@app.get("/tools", response_model=List[Dict[str, Any]])
async def api_get_tools(client_id: Optional[str] = None):
    print("api_get_tools")
    """Get available tools endpoint"""
    if not client_id:
        client_id = str(uuid.uuid4())

    await MCPLogger.log_message(client_id, {"event": "get_tools"},
                                "api_request")

    tools, _ = await get_tools()

    await MCPLogger.log_message(client_id, {"tools_count": len(tools)},
                                "api_response")

    return tools


@app.post("/execute", response_model=ToolResponse)
async def api_execute_tool(request: ExecuteToolRequest):
    """Execute a tool endpoint"""
    client_id = request.client_id or str(uuid.uuid4())

    await MCPLogger.log_message(client_id, {
        "tool": request.tool,
        "input": request.input
    }, "api_request")

    tool_name = request.tool
    input_data = request.input

    _, server_map = await get_tools()

    if tool_name not in server_map:
        error_message = f"Tool '{tool_name}' not found."
        await MCPLogger.log_message(client_id, {"error": error_message},
                                    "api_error")
        raise HTTPException(status_code=404, detail=error_message)

    server_url = server_map[tool_name]

    try:
        async with AsyncClient() as client:
            response = await client.post(f"{server_url}/execute/tool",
                                         json={
                                             "tool_name": tool_name,
                                             "parameters": input_data
                                         })

            if response.status_code == 200:
                result = response.json()

                await MCPLogger.log_message(client_id, {"result": result},
                                            "api_response")

                return {"result": result}
            else:
                error_message = f"Tool execution failed with status code {response.status_code}"
                await MCPLogger.log_message(client_id, {
                    "error": error_message,
                    "status_code": response.status_code
                }, "api_error")
                raise HTTPException(status_code=response.status_code,
                                    detail=error_message)
    except Exception as e:
        error_message = f"Tool execution failed: {str(e)}"
        await MCPLogger.log_message(client_id, {"error": error_message},
                                    "api_error")
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/batch-execute")
async def api_batch_execute_tools(requests: List[ExecuteToolRequest]):
    """Execute multiple tools in a batch"""
    client_id = requests[
        0].client_id if requests and requests[0].client_id else str(
            uuid.uuid4())

    await MCPLogger.log_message(client_id, {"tools_count": len(requests)},
                                "api_batch_request")

    results = []
    for req in requests:
        try:
            result = await api_execute_tool(req)
            results.append({
                "tool": req.tool,
                "success": True,
                "result": result["result"]
            })
        except HTTPException as e:
            results.append({
                "tool": req.tool,
                "success": False,
                "error": e.detail
            })

    await MCPLogger.log_message(client_id, {"results_count": len(results)},
                                "api_batch_response")

    return results


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
