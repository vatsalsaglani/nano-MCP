from typing import Dict, List, AsyncGenerator
from openai import AsyncOpenAI
from providers.base import BaseProvider
import traceback


class OpenAIProvider(BaseProvider):

    def __init__(self, api_key: str, base_url: str = None):

        self.client = AsyncOpenAI(api_key=api_key)
        if base_url:
            self.client.base_url = base_url

    async def complete(self, messages: List[Dict], model_name: str,
                       **kwargs) -> str:
        response = await self.client.chat.completions.create(model=model_name,
                                                             messages=messages,
                                                             **kwargs)
        return response.choices[0].message.content

    async def stream(self, messages: List[Dict], model_name: str,
                     **kwargs) -> AsyncGenerator[str, None]:
        try:
            async for chunk in await self.client.chat.completions.create(
                    model=model_name, messages=messages, stream=True,
                    **kwargs):
                yield chunk.choices[0].delta.content
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            raise e
