from abc import ABC, abstractmethod
from typing import Dict, List, AsyncGenerator


class BaseProvider(ABC):

    @abstractmethod
    def complete(self, messages: List[Dict], model_name: str, **kwargs) -> str:
        pass

    @abstractmethod
    def stream(self, messages: List[Dict], model_name: str,
               **kwargs) -> AsyncGenerator[str, None]:
        pass
