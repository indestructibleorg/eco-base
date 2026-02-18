"""
Agent / Function Calling Service
ReAct, Plan-and-Execute, tool orchestration.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional

from src.core.router import InferenceRouter
from src.schemas.inference import (
    ChatCompletionRequest,
    ChatMessage,
    ChatRole,
    FunctionDefinition,
    ToolDefinition,
)
from src.utils.logging import get_logger

logger = get_logger("superai.specialized.agent")


class ToolRegistry:
    """Registry for callable tools available to the agent."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        self._tools[name] = {
            "definition": ToolDefinition(
                function=FunctionDefinition(
                    name=name,
                    description=description,
                    parameters=parameters,
                )
            ),
            "handler": handler,
        }

    def get_definitions(self) -> List[ToolDefinition]:
        return [t["definition"] for t in self._tools.values()]

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        return await tool["handler"](**arguments)

    @property
    def names(self) -> List[str]:
        return list(self._tools.keys())


class AgentService:
    """
    Agent service with ReAct-style reasoning and tool calling.

    Supports:
    - OpenAI-compatible function calling
    - ReAct (Reason + Act) loop
    - Plan-and-Execute for complex tasks
    - Multi-step tool orchestration
    - Conversation memory
    """

    AGENT_SYSTEM_PROMPT = (
        "You are an AI agent that can use tools to accomplish tasks. "
        "Think step by step. For each step:\n"
        "1. Analyze what information you need\n"
        "2. Choose the appropriate tool\n"
        "3. Use the tool and observe the result\n"
        "4. Decide if you need more steps or can provide the final answer\n\n"
        "Always explain your reasoning before using a tool."
    )

    def __init__(self, router: InferenceRouter, max_iterations: int = 10):
        self._router = router
        self._tool_registry = ToolRegistry()
        self._max_iterations = max_iterations

    @property
    def tools(self) -> ToolRegistry:
        return self._tool_registry

    async def run(
        self,
        task: str,
        model: Optional[str] = None,
        tools: Optional[List[ToolDefinition]] = None,
        context: Optional[List[ChatMessage]] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Execute an agent task with ReAct-style reasoning.

        Args:
            task: The task description
            model: Model to use for reasoning
            tools: Override tool definitions
            context: Additional conversation context
            max_tokens: Max tokens per step
        """
        session_id = uuid.uuid4().hex[:12]
        tool_defs = tools or self._tool_registry.get_definitions()

        messages: List[ChatMessage] = [
            ChatMessage(role=ChatRole.SYSTEM, content=self.AGENT_SYSTEM_PROMPT),
        ]
        if context:
            messages.extend(context)
        messages.append(ChatMessage(role=ChatRole.USER, content=task))

        steps: List[Dict[str, Any]] = []
        final_answer = ""

        for iteration in range(self._max_iterations):
            request = ChatCompletionRequest(
                model=model or "default",
                messages=messages,
                tools=tool_defs if tool_defs else None,
                tool_choice="auto" if tool_defs else None,
                max_tokens=max_tokens,
                temperature=0.3,
            )

            response = await self._router.chat_completion(request)
            choice = response.choices[0] if response.choices else None
            if not choice:
                break

            assistant_msg = choice.message

            # Check if the model wants to call tools
            if assistant_msg.tool_calls:
                messages.append(assistant_msg)

                for tool_call in assistant_msg.tool_calls:
                    func = tool_call.get("function", {})
                    tool_name = func.get("name", "")
                    tool_call_id = tool_call.get("id", uuid.uuid4().hex[:8])

                    try:
                        arguments = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        arguments = {}

                    step = {
                        "iteration": iteration,
                        "action": "tool_call",
                        "tool": tool_name,
                        "arguments": arguments,
                    }

                    try:
                        result = await self._tool_registry.execute(tool_name, arguments)
                        result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
                        step["result"] = result_str
                        step["status"] = "success"
                    except Exception as e:
                        result_str = f"Error: {str(e)}"
                        step["result"] = result_str
                        step["status"] = "error"

                    steps.append(step)

                    # Add tool result to conversation
                    messages.append(ChatMessage(
                        role=ChatRole.TOOL,
                        content=result_str,
                        tool_call_id=tool_call_id,
                    ))

                    logger.info(
                        "Agent tool call",
                        session=session_id,
                        iteration=iteration,
                        tool=tool_name,
                        status=step["status"],
                    )
            else:
                # No tool calls - this is the final answer
                content = assistant_msg.content if isinstance(assistant_msg.content, str) else ""
                final_answer = content
                steps.append({
                    "iteration": iteration,
                    "action": "final_answer",
                    "content": content,
                })
                break

        return {
            "session_id": session_id,
            "answer": final_answer,
            "steps": steps,
            "iterations": len(steps),
            "model": model or "default",
        }

    async def plan_and_execute(
        self,
        task: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plan-and-Execute strategy for complex multi-step tasks.
        First generates a plan, then executes each step.
        """
        # Step 1: Generate plan
        plan_prompt = (
            f"Create a step-by-step plan to accomplish this task:\n{task}\n\n"
            f"Available tools: {', '.join(self._tool_registry.names)}\n\n"
            f"Output a numbered list of steps. Each step should specify "
            f"which tool to use and what arguments to pass."
        )

        plan_request = ChatCompletionRequest(
            model=model or "default",
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content=self.AGENT_SYSTEM_PROMPT),
                ChatMessage(role=ChatRole.USER, content=plan_prompt),
            ],
            max_tokens=2048,
            temperature=0.2,
        )

        plan_response = await self._router.chat_completion(plan_request)
        plan = plan_response.choices[0].message.content if plan_response.choices else ""

        # Step 2: Execute the plan via ReAct loop
        execution_prompt = (
            f"Execute this plan step by step:\n\n{plan}\n\n"
            f"Original task: {task}"
        )

        result = await self.run(
            task=execution_prompt,
            model=model,
        )

        result["plan"] = plan
        return result