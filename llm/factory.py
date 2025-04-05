import logging
from typing import Optional, Generator

from llm.base import BaseLLMProvider
from llm.providers import (
    OpenAIProvider,
    OllamaProvider,
    OpenAILikeProvider,
    BedrockProvider,
    DeepSeekProvider,
)

logger = logging.getLogger(__name__)


class LLMInterface:
    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = self._get_provider(provider.lower(), model, **kwargs)

    def _get_provider(self, provider: str, model: str, **kwargs) -> BaseLLMProvider:
        if provider == "openai":
            return OpenAIProvider(model, **kwargs)
        elif provider == "openai_like":
            return OpenAILikeProvider(model, **kwargs)
        elif provider == "ollama":
            return OllamaProvider(model, **kwargs)
        elif provider == "bedrock":
            return BedrockProvider(model, **kwargs)
        elif provider == "deepseek":
            return DeepSeekProvider(model, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Optional[str]:
        try:
            return self.provider.generate(prompt, system_prompt, **kwargs)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise e

    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate streaming response from the LLM.

        Args:
            prompt (str): The prompt to send to the LLM
            system_prompt (Optional[str]): Optional system prompt to prepend to the prompt
            **kwargs: Additional arguments to pass to the LLM

        Yields:
            str: Chunks of the generated text
        """
        try:
            for chunk in self.provider.generate_stream(prompt, system_prompt, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"LLM streaming generation failed: {e}")
            yield f"Error: {str(e)}"
