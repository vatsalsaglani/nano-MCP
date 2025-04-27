from pathlib import Path
from pydantic import BaseModel
import aiofiles
import traceback
import logging

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --------------------------

BASE_DIR = Path("/data")
BASE_DIR.mkdir(exist_ok=True)


class ReadFileRequest(BaseModel):
    file_path: str


async def read_file(file_path: str):
    full_path = BASE_DIR / file_path
    try:
        if not full_path.is_file():
            # Specific error for file not found
            error_msg = f"File not found or is not a regular file: {full_path}"
            logger.warning(error_msg)
            return {"type": "text", "text": error_msg}

        async with aiofiles.open(full_path, mode='r') as file:
            content = await file.read()
        # Return standardized format for success
        return {"type": "text", "text": content}
    except Exception as e:
        # General error handling
        error_msg = f"Error reading file {file_path}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"type": "text", "text": error_msg}


class ShowFolderTreeRequest(BaseModel):
    path: str = ""


async def show_folder_tree(path: str = ""):
    target_path = BASE_DIR / path
    try:
        if not target_path.is_dir():
            # Specific error for directory not found
            error_msg = f"Directory not found: {target_path}"
            logger.warning(error_msg)
            return {"type": "text", "text": error_msg}

        tree_str = f"Contents of '{target_path}':\n"
        items = []
        for child in target_path.iterdir():
            item_type = "[D]" if child.is_dir() else "[F]"
            items.append(f"  {item_type} {child.name}")

        if not items:
            tree_str += "  (empty)"
        else:
            tree_str += "\n".join(items)
        # Return standardized format for success
        return {"type": "text", "text": tree_str}
    except Exception as e:
        # General error handling
        error_msg = f"Error listing directory {path}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"type": "text", "text": error_msg}


class UpdateFileRequest(BaseModel):
    file_path: str
    content: str


async def create_file(file_path: str, content: str):
    full_path = BASE_DIR / file_path
    try:
        # Ensure parent directory exists
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as dir_e:
            error_msg = f"Failed to create parent directory for {file_path}: {dir_e}\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {"type": "text", "text": error_msg}

        if full_path.exists():
            # Specific error for file exists
            error_msg = f"File already exists: {full_path}"
            logger.warning(error_msg)
            return {"type": "text", "text": error_msg}

        async with aiofiles.open(full_path, mode='w') as file:
            await file.write(content)
        # Return standardized format for success
        return {
            "type": "text",
            "text": f"File '{str(full_path)}' created successfully."
        }
    except Exception as e:
        # General error handling
        error_msg = f"Error creating file {file_path}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"type": "text", "text": error_msg}


async def update_file(file_path: str, content: str):
    full_path = BASE_DIR / file_path
    try:
        if not full_path.is_file():
            # Specific error for file not found
            error_msg = f"File not found or is not a regular file: {full_path}"
            logger.warning(error_msg)
            return {"type": "text", "text": error_msg}

        async with aiofiles.open(full_path, mode='w') as file:
            await file.write(content)
        # Return standardized format for success
        return {
            "type": "text",
            "text": f"File '{str(full_path)}' updated successfully."
        }
    except Exception as e:
        # General error handling
        error_msg = f"Error updating file {file_path}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"type": "text", "text": error_msg}


tools_list = [{
    "name": "read_file",
    "description": "Read the content of a specific file.",
    "parameters": ReadFileRequest.model_json_schema()
}, {
    "name": "show_folder_tree",
    "description":
    "List the contents (files and directories) of a specified directory.",
    "parameters": ShowFolderTreeRequest.model_json_schema()
}, {
    "name": "update_file",
    "description":
    "Update the content of an existing file. Overwrites the file.",
    "parameters": UpdateFileRequest.model_json_schema()
}, {
    "name": "create_file",
    "description": "Create a new file with the specified content.",
    "parameters": UpdateFileRequest.model_json_schema()
}]
