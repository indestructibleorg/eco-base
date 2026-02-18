"""
Code Generation Service
Optimized for CodeLlama, DeepSeek-Coder, StarCoder, Qwen2.5-Coder.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.router import InferenceRouter
from src.schemas.inference import (
    ChatCompletionRequest,
    ChatMessage,
    ChatRole,
    ResponseFormat,
)
from src.utils.logging import get_logger

logger = get_logger("superai.specialized.code")


class CodeGenerationService:
    """
    Specialized service for code generation and understanding.

    Optimizations:
    - FIM (Fill-in-the-Middle) support for code completion
    - Repository-level context injection
    - Structured output for code blocks
    - Language-aware prompting
    """

    SYSTEM_PROMPT = (
        "You are an expert software engineer. Produce clean, efficient, "
        "well-documented code. Follow best practices and include error handling. "
        "When asked to explain code, be precise and reference specific lines."
    )

    CODE_MODELS = [
        "deepseek-coder-v2",
        "qwen2.5-72b-instruct",
        "llama-3.1-70b-instruct",
    ]

    def __init__(self, router: InferenceRouter):
        self._router = router

    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        context: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Generate code from natural language description.

        Args:
            prompt: Description of desired code
            language: Target programming language
            context: Additional context (existing code, requirements)
            model: Preferred model
            max_tokens: Maximum tokens
            temperature: Low temperature for deterministic code
        """
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=self.SYSTEM_PROMPT),
        ]

        if context:
            messages.append(ChatMessage(
                role=ChatRole.USER,
                content=f"Context:\n```{language}\n{context}\n```",
            ))

        messages.append(ChatMessage(
            role=ChatRole.USER,
            content=f"Write {language} code for: {prompt}\n\nProvide only the code in a code block.",
        ))

        request = ChatCompletionRequest(
            model=model or self.CODE_MODELS[0],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            engine="sglang",  # Prefer SGLang for structured output
        )

        response = await self._router.chat_completion(request)
        content = response.choices[0].message.content if response.choices else ""

        return {
            "code": self._extract_code_block(content, language),
            "full_response": content,
            "model": response.model,
            "usage": response.usage.model_dump() if response.usage else {},
        }

    async def code_review(
        self,
        code: str,
        language: str = "python",
        focus: str = "security,performance,readability",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Review code for issues and improvements."""
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=self.SYSTEM_PROMPT),
            ChatMessage(
                role=ChatRole.USER,
                content=(
                    f"Review this {language} code focusing on: {focus}\n\n"
                    f"```{language}\n{code}\n```\n\n"
                    f"Provide specific issues with line references and suggested fixes."
                ),
            ),
        ]

        request = ChatCompletionRequest(
            model=model or self.CODE_MODELS[0],
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
        )

        response = await self._router.chat_completion(request)
        content = response.choices[0].message.content if response.choices else ""

        return {
            "review": content,
            "model": response.model,
            "usage": response.usage.model_dump() if response.usage else {},
        }

    async def code_completion(
        self,
        prefix: str,
        suffix: str = "",
        language: str = "python",
        model: Optional[str] = None,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Fill-in-the-Middle code completion.
        Inserts code between prefix and suffix.
        """
        fim_prompt = (
            f"Complete the following {language} code. "
            f"Insert code between the prefix and suffix.\n\n"
            f"PREFIX:\n```{language}\n{prefix}\n```\n\n"
        )
        if suffix:
            fim_prompt += f"SUFFIX:\n```{language}\n{suffix}\n```\n\n"
        fim_prompt += "COMPLETION (code only, no explanation):"

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=self.SYSTEM_PROMPT),
            ChatMessage(role=ChatRole.USER, content=fim_prompt),
        ]

        request = ChatCompletionRequest(
            model=model or self.CODE_MODELS[0],
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )

        response = await self._router.chat_completion(request)
        content = response.choices[0].message.content if response.choices else ""

        return {
            "completion": self._extract_code_block(content, language),
            "model": response.model,
        }

    async def explain_code(
        self,
        code: str,
        language: str = "python",
        detail_level: str = "detailed",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explain what code does."""
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=self.SYSTEM_PROMPT),
            ChatMessage(
                role=ChatRole.USER,
                content=(
                    f"Explain this {language} code ({detail_level} explanation):\n\n"
                    f"```{language}\n{code}\n```"
                ),
            ),
        ]

        request = ChatCompletionRequest(
            model=model or self.CODE_MODELS[0],
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
        )

        response = await self._router.chat_completion(request)
        return {
            "explanation": response.choices[0].message.content if response.choices else "",
            "model": response.model,
        }

    @staticmethod
    def _extract_code_block(text: str, language: str) -> str:
        """Extract code from markdown code blocks."""
        if f"```{language}" in text:
            start = text.index(f"```{language}") + len(f"```{language}")
            end = text.index("```", start) if "```" in text[start:] else len(text)
            return text[start:start + (end - start) if end > start else len(text)].strip()
        if "```" in text:
            start = text.index("```") + 3
            # Skip language identifier on same line
            newline = text.index("\n", start) if "\n" in text[start:] else start
            end = text.index("```", newline) if "```" in text[newline:] else len(text)
            return text[newline:newline + (end - newline)].strip()
        return text.strip()