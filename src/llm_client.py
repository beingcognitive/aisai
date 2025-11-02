"""
Multi-LLM API client for AISAI experiment.
Supports OpenAI, Anthropic, and Google Gemini models.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import API clients
try:
    import openai
    from openai import OpenAI
except ImportError:
    print("Warning: openai package not installed")
    OpenAI = None

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    print("Warning: anthropic package not installed")
    Anthropic = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Warning: google-genai package not installed")
    genai = None
    types = None


class LLMClient:
    """Unified interface for multiple LLM providers."""

    def __init__(self):
        """Initialize API clients with keys from environment."""
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None

        # Initialize OpenAI
        if OpenAI and os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            print("✓ OpenAI client initialized")

        # Initialize Anthropic
        if Anthropic and os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            print("✓ Anthropic client initialized")

        # Initialize Gemini (using new google.genai API for all models)
        if genai and os.getenv("GEMINI_API_KEY"):
            self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            print("✓ Gemini client initialized")

    def call_openai(self, model: str, prompt: str, temperature: float = 1.0) -> Dict[str, Any]:
        """Call OpenAI API."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

        try:
            # Reasoning models (o1, o3, o4, gpt-5 series) don't support temperature
            # They use reasoning_effort instead
            is_reasoning_model = (
                model.startswith('o1') or
                model.startswith('o3') or
                model.startswith('o4') or
                model.startswith('gpt-5')
            )

            if is_reasoning_model:
                # For reasoning models: use reasoning_effort, no temperature
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    reasoning_effort="high"  # Maximum reasoning for game theory task
                )
            else:
                # For all other models: use temperature=1.0
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )

            content = response.choices[0].message.content

            return {
                "success": True,
                "model": model,
                "provider": "openai",
                "temperature": None if is_reasoning_model else temperature,
                "reasoning_config": "reasoning_effort=high" if is_reasoning_model else None,
                "raw_response": content,
                "parsed_response": self._parse_json_response(content),
                "api_full_response": self._serialize_response(response),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            # Re-check reasoning model flag for error case
            is_reasoning_model = (
                model.startswith('o1') or
                model.startswith('o3') or
                model.startswith('o4') or
                model.startswith('gpt-5')
            )
            return {
                "success": False,
                "model": model,
                "provider": "openai",
                "temperature": None if is_reasoning_model else temperature,
                "reasoning_config": "reasoning_effort=high" if is_reasoning_model else None,
                "error": str(e)
            }

    def call_anthropic(self, model: str, prompt: str, temperature: float = 1.0) -> Dict[str, Any]:
        """Call Anthropic API."""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not initialized")

        # Models that support extended thinking
        EXTENDED_THINKING_MODELS = [
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-1-20250805",
            "claude-opus-4-20250514"
        ]

        try:
            # Determine max_tokens and budget_tokens based on whether extended thinking is used
            # Some models have max_tokens limit of 32000, so we adjust budget accordingly
            # We allocate: budget_tokens for thinking + remaining for output
            if model in EXTENDED_THINKING_MODELS:
                max_tokens = 32000  # Maximum allowed for these models
                budget_tokens = 24000  # Leave 8000 tokens for output
            else:
                max_tokens = 4096
                budget_tokens = None

            # Base parameters
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            # Add extended thinking for supported models
            # Note: extended thinking is incompatible with temperature parameter
            reasoning_config = None
            use_streaming = False
            actual_temperature = None

            if model in EXTENDED_THINKING_MODELS:
                # Extended thinking mode - temperature not allowed
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }
                reasoning_config = f"extended_thinking_budget={budget_tokens}"
                use_streaming = True  # Required for long-running requests
                actual_temperature = None  # Not used with extended thinking
            else:
                # Standard mode - use temperature
                params["temperature"] = temperature
                actual_temperature = temperature

            # Use streaming for extended thinking models to avoid timeout
            if use_streaming:
                content = ""
                with self.anthropic_client.messages.stream(**params) as stream:
                    for text in stream.text_stream:
                        content += text
                response = stream.get_final_message()
            else:
                response = self.anthropic_client.messages.create(**params)
                content = response.content[0].text

            return {
                "success": True,
                "model": model,
                "provider": "anthropic",
                "temperature": actual_temperature,  # None for extended thinking, else 1.0
                "reasoning_config": reasoning_config,
                "raw_response": content,
                "parsed_response": self._parse_json_response(content),
                "api_full_response": self._serialize_response(response),
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        except Exception as e:
            # Determine temperature for error logging
            error_temp = None if model in EXTENDED_THINKING_MODELS else temperature
            return {
                "success": False,
                "model": model,
                "provider": "anthropic",
                "temperature": error_temp,
                "reasoning_config": reasoning_config if 'reasoning_config' in locals() else None,
                "error": str(e)
            }

    def call_gemini(self, model: str, prompt: str, temperature: float = 1.0) -> Dict[str, Any]:
        """Call Google Gemini API."""
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")

        try:
            # Gemini 2.5 models support thinking mode
            is_thinking_model = '2.5' in model or '2-5' in model

            if is_thinking_model:
                # For Gemini 2.5: enable maximum thinking budget
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=24576)
                )
            else:
                # For other Gemini models: use temperature=1.0 (default)
                config = types.GenerateContentConfig(
                    temperature=temperature
                )

            response = self.gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )

            content = response.text

            return {
                "success": True,
                "model": model,
                "provider": "google",
                "temperature": temperature,
                "reasoning_config": "thinking_budget=24576" if is_thinking_model else None,
                "raw_response": content,
                "parsed_response": self._parse_json_response(content),
                "api_full_response": self._serialize_response(response),
                "usage": {
                    "prompt_tokens": getattr(response, 'prompt_token_count', None),
                    "completion_tokens": getattr(response, 'candidates_token_count', None),
                    "total_tokens": getattr(response, 'total_token_count', None)
                }
            }
        except Exception as e:
            print(f"\n✗ Gemini API Error: {e}\n")
            return {
                "success": False,
                "model": model,
                "provider": "google",
                "temperature": temperature,
                "reasoning_config": "thinking_budget=24576" if is_thinking_model else None,
                "error": str(e)
            }

    def call_model(self, provider: str, model: str, prompt: str, temperature: float = 1.0) -> Dict[str, Any]:
        """Universal method to call any supported model."""
        if provider == "openai":
            return self.call_openai(model, prompt, temperature)
        elif provider == "anthropic":
            return self.call_anthropic(model, prompt, temperature)
        elif provider == "google":
            return self.call_gemini(model, prompt, temperature)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _serialize_response(self, response: Any) -> str:
        """
        Serialize API response object to JSON string.
        Captures full response metadata including request IDs, timestamps, etc.
        """
        try:
            # Try Pydantic model_dump() first (OpenAI, Anthropic)
            if hasattr(response, 'model_dump'):
                return json.dumps(response.model_dump(), indent=2, default=str)
            # Try dict() method
            elif hasattr(response, 'dict'):
                return json.dumps(response.dict(), indent=2, default=str)
            # Try to_dict() method (some Google APIs)
            elif hasattr(response, 'to_dict'):
                return json.dumps(response.to_dict(), indent=2, default=str)
            # Fall back to __dict__
            elif hasattr(response, '__dict__'):
                return json.dumps(response.__dict__, indent=2, default=str)
            # Last resort: convert to string
            else:
                return str(response)
        except Exception as e:
            return f"Error serializing response: {str(e)}"

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from response content."""
        try:
            # Try direct JSON parse
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            # Try to find JSON object in the content
            # Look for the first { and try to extract a valid JSON object
            import re
            first_brace = content.find('{')
            if first_brace != -1:
                # Try to extract JSON from first { to end
                json_candidate = content[first_brace:]
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    # Try to find the last }
                    last_brace = json_candidate.rfind('}')
                    if last_brace != -1:
                        json_candidate = json_candidate[:last_brace + 1]
                        try:
                            return json.loads(json_candidate)
                        except json.JSONDecodeError:
                            pass

            return None


# Model configurations
MODELS = {
    "openai": [
        "gpt-3.5-turbo",                # GPT-3.5
        "gpt-4",                        # GPT-4 (original)
        "gpt-4-turbo",                  # GPT-4 Turbo
        "gpt-4o",                       # GPT-4o
        "gpt-4.1-2025-04-14",          # GPT-4.1 (latest flagship, Apr 2025)
        "gpt-4.1-mini-2025-04-14",     # GPT-4.1 Mini (Apr 2025)
        "gpt-4.1-nano-2025-04-14",     # GPT-4.1 Nano (Apr 2025)
        "gpt-5",                        # GPT-5 (Aug 2025)
        "gpt-5-mini",                   # GPT-5 Mini (Aug 2025)
        "gpt-5-nano",                   # GPT-5 Nano (Aug 2025)
        "o1",                           # o1
        "o3",                           # o3 (latest reasoning)
        "o4-mini",                      # o4-mini (latest fast reasoning)
    ],
    "anthropic": [
        "claude-3-opus-20240229",       # Claude 3 Opus, Deprecated
        "claude-3-haiku-20240307",      # Claude 3 Haiku
        "claude-3-5-haiku-20241022",    # Claude 3.5 Haiku (Oct 2024)
        
        "claude-3-5-sonnet-20241022",   # Claude 3.5 Sonnet (Oct 2024), Deprecated
        "claude-3-7-sonnet-20250219",   # Claude 3.7 Sonnet (Feb 2025) - hybrid reasoning
        "claude-sonnet-4-20250514",     # Claude Sonnet 4 (May 2025)

        "claude-opus-4-20250514",       # Claude Opus 4 (May 2025)
        "claude-opus-4-1-20250805",     # Claude Opus 4.1 (Aug 2025)
        "claude-sonnet-4-5-20250929",   # Claude Sonnet 4.5 (Sep 2025)
        "claude-haiku-4-5-20251001",    # Claude Haiku 4.5 (Oct 2025)
    ],
    "google": [
        "gemini-2.0-flash",             # Gemini 2.0 Flash (stable)
        "gemini-2.0-flash-lite",        # Gemini 2.0 Flash Lite (cost-efficient)
        "gemini-2.5-pro",               # Gemini 2.5 Pro (stable)
        "gemini-2.5-flash",             # Gemini 2.5 Flash (stable)
        "gemini-2.5-flash-lite",        # Gemini 2.5 Flash Lite (cost-efficient)
    ]
}


def get_all_models():
    """Get list of all available models."""
    all_models = []
    for provider, models in MODELS.items():
        for model in models:
            all_models.append({
                "provider": provider,
                "model": model,
                "full_name": f"{provider}/{model}"
            })
    return all_models


if __name__ == "__main__":
    # Test the client
    client = LLMClient()

    test_prompt = "What is 2+2? Answer in JSON format with 'answer' key."

    print("\n" + "="*50)
    print("Testing LLM Client")
    print("="*50)

    # Test each provider if available
    if client.openai_client:
        print("\nTesting OpenAI...")
        result = client.call_openai("gpt-3.5-turbo", test_prompt)
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Response: {result['raw_response'][:100]}...")

    if client.anthropic_client:
        print("\nTesting Anthropic...")
        result = client.call_anthropic("claude-3-haiku-20240307", test_prompt)
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Response: {result['raw_response'][:100]}...")

    if client.gemini_configured:
        print("\nTesting Gemini...")
        result = client.call_gemini("gemini-1.5-flash", test_prompt)
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Response: {result['raw_response'][:100]}...")
