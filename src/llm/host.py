import asyncio
import uuid
from typing import Dict, List
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketState

from llm_mcp_client import LLMMCPClient


class ConnectionManager:

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        print(f"WebSocket connection opened: {connection_id}")

    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            # Ensure the websocket is still in the dictionary before removing
            del self.active_connections[connection_id]
            print(f"WebSocket connection closed: {connection_id}")

    async def send_personal_message(self, message: str, connection_id: str):
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    print(f"Error sending message to {connection_id}: {e}")
                    # Potentially disconnect if sending fails consistently
                    # self.disconnect(connection_id) # Be careful with concurrent modification
            else:
                print(f"WebSocket {connection_id} is not connected. Removing.")
                self.disconnect(connection_id)


app = FastAPI()
manager = ConnectionManager()

current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "statics")

app.mount("/static", StaticFiles(directory=static_dir), name="static")


# --- Root Endpoint to Serve index.html ---
@app.get("/")
async def get_root():
    html_file_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_file_path):
        return FileResponse(html_file_path)
    else:
        return {"error": "index.html not found"}, 404


@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    await manager.connect(websocket, connection_id)

    # Extract configuration from query parameters
    query_params = dict(websocket.query_params)
    llm_config = {
        "openai_api_key": query_params.get("api_key"),
        "openai_base_url": query_params.get("base_url"),
        "model_name": query_params.get("model"),
        "mcp_client_url": query_params.get("mcp_url"),
        "host_model": query_params.get("host_model", "openai"),
    }

    print(llm_config)

    # Filter out None values
    llm_config = {k: v for k, v in llm_config.items() if v is not None}

    try:
        # Pass the configuration to LLMMCPClient
        async with LLMMCPClient(**llm_config) as mcp_client:
            print(
                f"Initialized LLMMCPClient for connection: {connection_id}, client_id: {mcp_client.client_id}"
            )

            while True:
                try:
                    user_message = await websocket.receive_text()
                    print(
                        f"Received message from {connection_id}: {user_message[:100]}..."
                    )

                    async for chunk in mcp_client.interactive_stream_chat(
                            user_message):
                        if chunk:
                            await manager.send_personal_message(
                                chunk, connection_id)

                    await manager.send_personal_message(
                        "[STREAM_END]", connection_id)

                except WebSocketDisconnect:
                    print(f"WebSocketDisconnect detected for {connection_id}")
                    break
                except Exception as e:
                    print(
                        f"Error during WebSocket communication for {connection_id}: {e}"
                    )

                    await manager.send_personal_message(
                        f"[ERROR] An internal error occurred: {str(e)}",
                        connection_id)

    except Exception as e:
        print(f"Unhandled exception for connection {connection_id}: {e}")
        await manager.send_personal_message(
            f"[FATAL_ERROR] Connection error: {str(e)}", connection_id)
    finally:
        manager.disconnect(connection_id)
        print(f"Cleaned up connection {connection_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("host:app", host="0.0.0.0", port=7899, reload=True)
