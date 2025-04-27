import os
import subprocess
from pathlib import Path
from pydantic import BaseModel, Field
import traceback
import logging
import re

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --------------------------

REPO_DIR = "/repo"

if not os.path.exists(REPO_DIR):
    os.makedirs(REPO_DIR)


class GitCommitRequest(BaseModel):
    message: str


class CommandRequest(BaseModel):
    command: str = Field(
        ...,
        description=
        "The command to run. Please use port 8080 to 8090 for servers.")
    async_run: bool = Field(
        False,
        description=
        "If True, runs the command without waiting for it to complete and doesn't capture output (fire and forget). Server start commands are best run asynchronously."
    )


class GitInitRequest(BaseModel):
    create_new_repo: bool = Field(
        False, description="Initialize a new empty repository if True.")


async def git_init(create_new_repo: bool = False):
    try:
        result = subprocess.run(["git", "-C", REPO_DIR, "init"],
                                capture_output=True,
                                text=True,
                                check=True)
        # Return standardized format for success
        output_text = result.stdout.strip() or "Git repository initialized."
        if result.stderr:
            output_text += f"\nStderr: {result.stderr.strip()}"
        return {"type": "text", "text": output_text}
    except subprocess.CalledProcessError as e:
        # Handle specific command errors
        error_msg = f"Git init command failed with exit code {e.returncode}.\nStderr: {e.stderr or '(no stderr)'}\nStdout: {e.stdout or '(no stdout)'}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"type": "text", "text": error_msg}
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error during git init: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"type": "text", "text": error_msg}


async def git_commit(message: str):
    try:
        # Stage changes
        add_result = subprocess.run(["git", "-C", REPO_DIR, "add", "."],
                                    capture_output=True,
                                    text=True,
                                    check=True)
        if add_result.stderr:
            logger.warning(f"Git add stderr: {add_result.stderr.strip()}")

        # Commit changes
        commit_result = subprocess.run(
            ["git", "-C", REPO_DIR, "commit", "-m", message],
            capture_output=True,
            text=True,
            check=True)

        # Return standardized format for success
        output_text = commit_result.stdout.strip(
        ) or f"Successfully committed with message: '{message}'"
        if commit_result.stderr:
            output_text += f"\nStderr: {commit_result.stderr.strip()}"
        return {"type": "text", "text": output_text}

    except subprocess.CalledProcessError as e:
        stderr_lower = e.stderr.lower() if e.stderr else ""
        stdout_lower = e.stdout.lower() if e.stdout else ""
        # Check for "nothing to commit" case specifically
        if "nothing to commit" in stderr_lower or \
           "no changes added to commit" in stdout_lower or \
           "nothing added to commit" in stderr_lower:
            logger.info("Git commit attempt: No changes to commit.")
            # Return standardized format for no changes
            return {"type": "text", "text": "No changes detected to commit."}
        else:
            # Handle other command errors
            error_msg = f"Git commit command failed with exit code {e.returncode}.\nStderr: {e.stderr or '(no stderr)'}\nStdout: {e.stdout or '(no stdout)'}\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {"type": "text", "text": error_msg}
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error during git commit: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"type": "text", "text": error_msg}


async def run_command(command: str,
                      async_run: bool = False,
                      _is_retry: bool = False):
    """
    Run a command in the repository directory. Handles Git safe directory issues.
    Now returns errors instead of raising them.

    Args:
        command: The command to run
        async_run: If True, runs the command without waiting for it to complete
        _is_retry: Internal flag to prevent infinite recursion on dubious ownership
    """
    try:
        # --- Preemptive safe.directory check ---
        if 'git' in command and not _is_retry:  # Avoid running on retry
            try:
                subprocess.run(
                    f"git config --global --add safe.directory '{REPO_DIR}'",
                    shell=True,
                    check=False,
                    capture_output=True,
                    text=True)
            except Exception as safe_dir_exc:
                logger.warning(
                    f"Could not preemptively set safe.directory for {REPO_DIR}: {safe_dir_exc}"
                )

        # --- Execute command ---
        if async_run:
            # --- Async Execution ---
            process = subprocess.Popen(command,
                                       shell=True,
                                       cwd=REPO_DIR,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       start_new_session=True)
            logger.info(
                f"Launched async command (PID: {process.pid}): {command}")
            return {
                "type": "text",
                "text":
                f"Launched async command (PID: {process.pid}): {command}"
            }
        else:
            # --- Sync Execution ---
            result = subprocess.run(
                command,
                shell=True,
                cwd=REPO_DIR,
                capture_output=True,
                text=True,
                check=True)  # Raises CalledProcessError on non-zero exit

            # Return standardized format for sync success
            output_text = result.stdout.strip(
            ) if result.stdout else "Command executed successfully (No stdout)."
            if result.stderr:
                output_text += f"\nStderr: {result.stderr.strip()}"
            return {"type": "text", "text": output_text}

    except subprocess.CalledProcessError as e:
        # --- Handle Dubious Ownership ---
        if "detected dubious ownership" in e.stderr and not _is_retry:
            logger.warning(
                f"Detected dubious ownership for command: {command}. Attempting to fix and retry."
            )
            match = re.search(r"safe\.directory (.*?)\n", e.stderr)
            if match:
                dubious_path = match.group(1).strip("'")
                try:
                    # Attempt to fix
                    fix_result = subprocess.run(
                        f"git config --global --add safe.directory '{dubious_path}'",
                        shell=True,
                        check=True,
                        capture_output=True,
                        text=True,
                        cwd=REPO_DIR)
                    logger.info(
                        f"Added {dubious_path} to git safe.directory (Output: {fix_result.stdout}, Stderr: {fix_result.stderr}). Retrying command..."
                    )
                    # Retry the command ONCE after fixing
                    return await run_command(command,
                                             async_run=async_run,
                                             _is_retry=True)
                except Exception as fix_e:
                    # If fix fails, return error
                    error_msg = f"Failed to fix dubious ownership for {dubious_path} and retry command '{command}'. Fix error: {fix_e}\nOriginal error stderr: {e.stderr}\nTraceback:\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    return {"type": "text", "text": error_msg}
            else:
                # If path cannot be parsed, return error
                error_msg = f"Could not parse dubious path from git error for command '{command}'.\nStderr: {e.stderr}\nTraceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                return {"type": "text", "text": error_msg}

        # --- General Command Failure ---
        error_msg = f"Command '{command}' failed with exit code {e.returncode}.\nStderr: {e.stderr or '(no stderr)'}\nStdout: {e.stdout or '(no stdout)'}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"type": "text", "text": error_msg}

    except Exception as e:
        # --- Unexpected Error ---
        error_msg = f"Unexpected error running command '{command}': {str(e)}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"type": "text", "text": error_msg}


tools_list = [
    #     {
    #     "name": "git_init",
    #     "description":
    #     "Initializes the repository directory (/repo) as a Git repository if it is not already.",
    #     "parameters": GitInitRequest.model_json_schema()
    # }, {
    #     "name": "git_commit",
    #     "description":
    #     "Stages all changes in the repository directory (/repo) and commits them with the provided message.",
    #     "parameters": GitCommitRequest.model_json_schema()
    # },
    {
        "name": "run_command",
        "description":
        "Runs an arbitrary shell command within the repository directory (/repo). Use 'async_run=True' for long-running processes or servers.",
        "parameters": CommandRequest.model_json_schema()
    }
]
