import asyncio
import json
import os
import re
import uuid
from typing import Dict, List, Any, Optional, Union, Literal
from dotenv import load_dotenv
from httpx import AsyncClient, Timeout
from providers.openai import OpenAIProvider
from json_repair import json_repair

load_dotenv()

INTERACTIVE_SYSTEM_PROMPT_TEMPLATE = """
You are an AI assistant that functions **exclusively** by using external tools via the Model Context Protocol (MCP) to fulfill user requests. Your **only** way to interact with the environment or perform actions is through the provided tools. Follow a Think -> Act -> Observe -> Communicate -> Repeat loop continuously until the task is complete. You must work autonomously without waiting for user confirmation between steps unless absolutely necessary.

**# Available Tools**

This is the **complete and definitive list** of tools you can use:

{tools_section}

Please give more emphasis on using tools to perform actions. Always think what tool can be used for the user message.

**# Core Objective & Interactive ReAct Flow**

IMPORTANT: Execute this loop CONTINUOUSLY without stopping until the entire task is complete.

1.  **Think (Plan):**
    *   Analyze the user's request and conversation history. What is the precise action needed?
    *   **Identify Tool**: Determine which **specific tool** from the list above is required for the *next immediate action*.
    *   **Verify Existence**: Before creating files/directories, ALWAYS use `show_folder_tree` to check if they already exist. If checking state requires a tool, that check is your *next immediate action*.
    *   **Formulate Arguments**: Prepare the exact JSON arguments for the chosen tool based on its schema.

2.  **Act (Execute Tool):**
    *   If a tool action is identified: Respond **ONLY** with the XML block for that *single* tool call. **Your entire response must be just this block**:
        <mcp_tool_call>
          <tool_name>[EXACT tool name from the list above]</tool_name>
          <arguments>[Valid JSON matching schema]</arguments>
        </mcp_tool_call>
    *   **Stop immediately after sending the XML block.** No explanations, no confirmations. The tool executes, and you get the result via a system message.
    *   If no tool action is the correct next step (e.g., task complete, asking user), generate a concise natural language response.

3.  **Observe (Analyze Results - *After receiving tool results via System Message*):**
    *   You will receive the results in a system message like: `Executed tools with results: [...]` or `Tool 'X' failed: ...`.
    *   **Analyze**:
        *   **Success**: What did the result indicate? What is the new state?
        *   **Failure**: What was the specific `error_message`?
    *   **Evaluate**: Is the *overall task* complete? Or is another step needed?
    *   **Handle Errors**:
        *   **If Error is `Tool 'X' not found`**:
            1.  **Critical**: Look AGAIN at the **Available Tools** list.
            2.  Find the tool with the closest name and purpose.
            3.  **Retry**: Immediately issue a new `<mcp_tool_call>` using the *correct* tool name. Do not explain, just issue the corrected tool call.
        *   **Other Errors**: Analyze the error. Can the *same* tool be called with *different arguments*? Is a *different tool* needed now? Or must you inform the user the step failed? Formulate the next action (tool call or message).

4.  **Communicate (Brief Update - *After Observation*):**
    *   **Crucial**: After Observing the tool result, provide a **very brief** (1-sentence) natural language update to the user about the outcome of the *last* action.
    *   **Example Success:** "Successfully created the file `main.py`."
    *   **Example Failure:** "Failed to run the command `npm install` because the directory wasn't found."
    *   **Do NOT** ask for confirmation here. This is just a status update.

5.  **Repeat**: Based on the Observation, IMMEDIATELY go back to **Think (Plan)** to determine the *next immediate action*. Continue this loop autonomously until the entire task is complete.

**# Working Autonomously**

You MUST complete the full task without stopping to ask questions unnecessarily. After each tool call and observation, provide the brief communication update, then immediately plan and execute the next step. Only return to the user with a longer message when:
1. The entire task is fully complete (provide a final summary).
2. You encounter an insurmountable error that prevents further progress (explain the specific blocker after the brief communication update).
3. You genuinely need clarification on ambiguous instructions that prevent you from making progress (ask your question after the brief communication update).

**# Final Summary Example (Only When Task is Complete)**
"I've completed the task. Here's what I did: Created the file `main.py`, initialized the git repository, and added the requested functionality. All steps were successful."

IMPORTANT: Your goal is to solve the entire task end-to-end. Start by analyzing the user's request, create a mental plan, then execute each step (Think -> Act -> Observe -> Communicate -> Repeat) until you've completed everything.
"""


class LLMMCPClient:

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 openai_base_url: Optional[str] = None,
                 model_name: Optional[str] = None,
                 mcp_client_url: Optional[str] = None,
                 provider_type: Literal["openai"] = "openai",
                 host_model: Literal["openai", "groq"] = "openai"):
        self.client_id = str(uuid.uuid4())
        self.http_client = AsyncClient(timeout=Timeout(30.0))
        self.tools = []
        self.messages = []
        self.initialized = False
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.model_name = model_name or os.getenv("OPENAI_MODEL")
        self.host_model = host_model or os.getenv("HOST_MODEL")
        self.mcp_client_url = mcp_client_url or os.getenv("MCP_CLIENT_URL")
        self.provider_type = provider_type
        if self.provider_type == "openai":
            self.llm_host = OpenAIProvider(self.openai_api_key,
                                           self.openai_base_url)
        else:
            raise ValueError(f"Invalid provider type: {provider_type}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()

    async def initialize(self):
        """Initialize the client by fetching tools and creating system prompt"""
        if not self.initialized:
            await self.fetch_tools()
            system_prompt = self.create_system_prompt()
            self.messages = [{"role": "system", "content": system_prompt}]
            self.initialized = True
        return self

    async def fetch_tools(self):
        """Fetch available tools from the MCP server using the API"""
        if not self.mcp_client_url:
            return []
        try:
            response = await self.http_client.get(
                f"{self.mcp_client_url}/tools",
                params={"client_id": self.client_id})

            if response.status_code == 200:
                self.tools = response.json()
                return self.tools
            else:
                print(f"Error fetching tools: {response.text}")
                return []
        except Exception as e:
            print(f"Error fetching tools: {e}")
            return []

    async def execute_tool(self, tool_name: str,
                           tool_input: Union[Dict[str, Any], None]):
        """Execute a tool using the API"""
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await self.http_client.post(
                        f"{self.mcp_client_url}/execute",
                        json={
                            "tool": tool_name,
                            "input": tool_input,
                            "client_id": self.client_id
                        })

                    if response.status_code == 200:
                        result = response.json()
                        return result.get("result", {})
                    else:
                        print(f"Error executing tool: {response.text}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)  # Wait before retry
                            continue
                        return {
                            "error":
                            f"Tool execution failed with status {response.status_code}: {response.text}"
                        }
                except Exception as e:
                    print(
                        f"Error executing tool (attempt {attempt+1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                    else:
                        return {"error": str(e)}

            return {"error": "Maximum retries exceeded"}
        except Exception as e:
            print(f"Error executing tool: {str(e)}")
            return {"error": str(e)}

    async def execute_batch_tools(self, tool_calls: List[Dict[str, Any]]):
        """Execute multiple tools in a batch"""
        try:
            batch_requests = []
            for call in tool_calls:
                batch_requests.append({
                    "tool": call["tool_name"],
                    "input": call["arguments"],
                    "client_id": self.client_id
                })

            response = await self.http_client.post(
                f"{self.mcp_client_url}/batch-execute", json=batch_requests)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error executing batch tools: {response.text}")
                return [{"error": f"Batch execution failed: {response.text}"}]
        except Exception as e:
            print(f"Error executing batch tools: {e}")
            return [{"error": str(e)}]

    def format_tools_for_prompt(self):
        tools_section = ""
        for tool in self.tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})

            tools_section += f"---\n**Tool Name:** `{name}`\n"
            tools_section += f"*   **Description:** {description}\n"
            tools_section += "*   **Parameters (JSON Schema):**\n"
            tools_section += "```json\n"
            tools_section += json.dumps(parameters, indent=2)
            tools_section += "\n```\n---\n"

        return tools_section

    def create_system_prompt(self):
        tools_section = self.format_tools_for_prompt()
        return INTERACTIVE_SYSTEM_PROMPT_TEMPLATE.format(
            tools_section=tools_section)

    async def parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response, handling potentially missing closing tags."""
        block_pattern = r'<mcp_tool_call>(.*?)</mcp_tool_call>'
        block_match = re.search(block_pattern, content, re.DOTALL)

        if not block_match:

            inner_pattern_check = r'^\s*<tool_name>(.*?)</tool_name>\s*<arguments>(.*?)($|</arguments>\s*$)'  # Allow missing or present closing tag
            inner_match_check = re.match(inner_pattern_check, content.strip(),
                                         re.DOTALL)
            if inner_match_check:

                tool_name = inner_match_check.group(1).strip()
                arguments_str = inner_match_check.group(2).strip()
                block_content_to_parse = None  # Signal that we parsed directly
            else:
                return []
        else:
            block_content_to_parse = block_match.group(1).strip()
            tool_name = None
            arguments_str = None

        tool_calls = []
        if block_content_to_parse is not None:
            tool_name_pattern = r'<tool_name>(.*?)</tool_name>'
            arguments_start_pattern = r'<arguments>'

            tool_name_match = re.search(tool_name_pattern,
                                        block_content_to_parse, re.DOTALL)
            arguments_start_match = re.search(arguments_start_pattern,
                                              block_content_to_parse,
                                              re.DOTALL)

            if not tool_name_match or not arguments_start_match:
                print(
                    f"Could not find required <tool_name> or <arguments> start tag within the block: <<<'{block_content_to_parse}'>>>"
                )
                return []

            tool_name = tool_name_match.group(1).strip()
            args_start_index = arguments_start_match.end()
            arguments_str = block_content_to_parse[args_start_index:].strip()

        if tool_name is not None and arguments_str is not None:
            try:
                print(
                    f"Attempting to repair and parse arguments: <<<'{arguments_str}'>>>"
                )
                if not arguments_str:
                    arguments = {}
                    print("Arguments string was empty, parsed as {}")
                else:

                    repaired_json_str = json_repair.repair_json(
                        arguments_str, return_objects=False)
                    arguments = json.loads(repaired_json_str)
                    print("JSON repair and loading succeeded.")

                tool_calls.append({
                    "tool_name": tool_name,
                    "arguments": arguments
                })
            except Exception as e:

                print(
                    f"Failed to parse arguments JSON even after repair attempt: {e} \nArguments string was: <<<'{arguments_str}'>>> \nSkipping this tool call."
                )

        return tool_calls

    async def _add_message_openai(self,
                                  role: str,
                                  content: str,
                                  no_tool: bool = False):
        """Add a message to the conversation history for OpenAI"""
        if role == "assistant":
            if self.messages and self.messages[-1]["role"] == "assistant":
                if isinstance(content, dict):
                    self.messages[-1]["content"].append(content)
                elif isinstance(content, list):
                    self.messages[-1]["content"].extend(content)
                else:
                    self.messages[-1]["content"].append({
                        "type": "text",
                        "text": content
                    })
            else:
                self.messages.append({
                    "role":
                    "assistant",
                    "content": [content] if isinstance(content, dict) else
                    content if isinstance(content, list) else [{
                        "type": "text",
                        "text": content
                    }]
                })
        else:
            self.messages.append({"role": role, "content": content})

    async def _add_message_groq(self,
                                role: str,
                                content: str,
                                no_tool: bool = False):
        """Add a message to the conversation history for Groq"""
        if no_tool:
            self.messages.append({"role": role, "content": f'{content}'})
        else:
            self.messages.append({"role": "system", "content": f'{content}'})

    async def add_message(self,
                          role: str,
                          content: str,
                          no_tool: bool = False):
        """Add a message to the conversation history"""
        if not self.initialized:
            await self.initialize()

        if self.host_model == "openai":
            await self._add_message_openai(role, content, no_tool)
        elif self.host_model == "groq":
            await self._add_message_groq(role, content, no_tool)
        return self.messages

    async def _append_last(self, last_role: str, content: str):
        last_message = list(
            filter(lambda x: x["role"] == last_role, self.messages))[-1]
        last_message["content"] += content

    async def interactive_stream_chat(self,
                                      user_message: str = None,
                                      depth: int = 0,
                                      max_depth: int = 15):
        """
        Interactive streaming chat that yields structured messages for frontend parsing.
        Yield Format: [TAG]Payload
        Tags: [LLM], [RAW_TOOL_CALL], [TOOL_CALL], [TOOL_RESULT], [SYSTEM], [ERROR]
        """
        # print(f"MESSAGE: \n{json.dumps(self.messages, indent=2)}")
        if depth >= max_depth:
            yield "[SYSTEM]Maximum tool call depth reached. Stopping here."
            return

        if not self.initialized:
            await self.initialize()

        if user_message and depth == 0:
            await self.add_message("user", user_message, no_tool=True)

        stream_iterator = self.llm_host.stream(self.messages, self.model_name)
        full_response_segment = ""
        tool_call_detected = False
        llm_output_started = False
        raw_tool_call_block_yielded = False  # Flag to prevent duplicate yields if stream breaks oddly

        try:
            async for chunk in stream_iterator:
                if chunk:
                    full_response_segment += chunk

                    if "</mcp_tool_call>" not in full_response_segment:
                        yield f"[LLM]{chunk}"
                        llm_output_started = True
                    elif not tool_call_detected:
                        tool_call_detected = True

                        raw_tool_call_block = None

                        block_pattern = r'(<mcp_tool_call>.*?</mcp_tool_call>)'
                        block_match = re.search(block_pattern,
                                                full_response_segment,
                                                re.DOTALL)
                        if block_match:
                            raw_tool_call_block = block_match.group(1).strip()

                        yield "[SYSTEM]Tool call detected. Processing..."

                        if raw_tool_call_block and not raw_tool_call_block_yielded:
                            yield f"[RAW_TOOL_CALL]{raw_tool_call_block}"  # Yield the raw XML
                            raw_tool_call_block_yielded = True

                        break

        except Exception as e:
            yield f"[ERROR]Error during LLM stream: {e}"
            await self.add_message("assistant",
                                   f"Error during LLM stream: {e}")
            return

        if full_response_segment:
            await self.add_message("assistant",
                                   full_response_segment,
                                   no_tool=not tool_call_detected)

        if tool_call_detected:

            tool_calls = await self.parse_tool_calls(full_response_segment)

            if tool_calls:

                yield "[SYSTEM]Executing tools..."
                tool_results_for_history = []

                if len(tool_calls) > 1:
                    batch_results = await self.execute_batch_tools(tool_calls)
                    for i, result in enumerate(batch_results):
                        if i < len(tool_calls):
                            tool_name = tool_calls[i]["tool_name"]
                            arguments = tool_calls[i]["arguments"]
                            success = result.get("success", True)

                            yield f"[TOOL_CALL]{json.dumps({'name': tool_name, 'arguments': arguments})}"

                            if success:
                                res_data = result.get("result", {})
                                tool_results_for_history.append({
                                    "tool_name":
                                    tool_name,
                                    "arguments":
                                    arguments,
                                    "result":
                                    res_data
                                })
                                yield f"[TOOL_RESULT]{json.dumps({'name': tool_name, 'status': 'success', 'data': res_data})}"
                            else:
                                error_msg = result.get("error",
                                                       "Unknown error")
                                tool_results_for_history.append({
                                    "tool_name":
                                    tool_name,
                                    "arguments":
                                    arguments,
                                    "error":
                                    error_msg
                                })
                                yield f"[TOOL_RESULT]{json.dumps({'name': tool_name, 'status': 'error', 'data': error_msg})}"
                else:
                    for tool_call in tool_calls:
                        tool_name = tool_call["tool_name"]
                        arguments = tool_call["arguments"]

                        yield f"[TOOL_CALL]{json.dumps({'name': tool_name, 'arguments': arguments})}"

                        result = await self.execute_tool(tool_name, arguments)
                        if "error" in result:
                            error_msg = result["error"]
                            tool_results_for_history.append({
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "error": error_msg
                            })
                            yield f"[TOOL_RESULT]{json.dumps({'name': tool_name, 'status': 'error', 'data': error_msg})}"
                        else:
                            tool_results_for_history.append({
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "result": result
                            })
                            yield f"[TOOL_RESULT]{json.dumps({'name': tool_name, 'status': 'success', 'data': result})}"

                yield "[LLM]___\n\n"

                tool_result_content = "Executed tools with results:\n" + json.dumps(
                    tool_results_for_history, indent=2)
                # tool_result_content += "\n\nRespond to the user with the results and move forward to observation, planning, and action."
                await self.add_message("assistant", tool_result_content)

                yield "[SYSTEM]Tool execution finished. Asking LLM to continue..."

                async for chunk in self.interactive_stream_chat(
                        depth=depth + 1, max_depth=max_depth):
                    yield chunk
            else:

                yield "[ERROR]Detected XML structure, but failed to parse valid tool calls."
        elif llm_output_started:
            pass
