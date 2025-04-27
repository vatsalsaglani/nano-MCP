from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tools import *
from fastapi.middleware.cors import CORSMiddleware
from tools import *
import traceback

app = FastAPI()
cors = CORSMiddleware(
    app=app,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tool_functions = {
    "git_init": git_init,
    "git_commit": git_commit,
    "run_command": run_command
}


@app.get("/list/tools")
async def list_tools():
    return tools_list


class ExecuteToolRequest(BaseModel):
    tool_name: str
    parameters: dict


@app.post("/execute/tool")
async def execute_tool(request: ExecuteToolRequest):
    tool_name = request.tool_name
    parameters = request.parameters
    if tool_name not in tool_functions:
        raise HTTPException(status_code=400, detail="Tool not found")
    try:
        result = await tool_functions[tool_name](**parameters)
    except Exception as e:
        print(f"Error executing tool {tool_name}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f'{str(e)}\nTraceback: {traceback.format_exc()}')
    return result


if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()

    PORT = int(os.getenv("PORT"))

    uvicorn.run("server:app", host="0.0.0.0", port=PORT)
