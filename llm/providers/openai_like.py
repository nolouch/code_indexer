import os
from typing import Optional, Generator
import openai
import logging

from llm.base import BaseLLMProvider


logger = logging.getLogger(__name__)


class OpenAILikeProvider(BaseLLMProvider):
    """
    Provider for OpenAI.
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        api_key = os.getenv("OPENAI_LIKE_API_KEY")
        base_url = os.getenv("OPENAI_LIKE_BASE_URL")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Optional[str]:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self._update_kwargs(kwargs),
        )

        if response.choices is None:
            raise Exception(f"LLM response is None: {response.error}")

        if hasattr(response.choices[0].message, "reasoning_content"):
            return (
                "<reasoning>"
                + response.choices[0].message.reasoning_content
                + "</reasoning>\n<answer>"
                + response.choices[0].message.content
                + "</answer>"
            )

        return response.choices[0].message.content.strip()

    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        try:
            response = self._retry_with_exponential_backoff(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                stream=True,  # Enable streaming
                **self._update_kwargs(kwargs),
            )

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error during OpenAI streaming: {e}")
            yield f"Error: {str(e)}"
