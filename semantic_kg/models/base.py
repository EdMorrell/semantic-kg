import abc
from typing import Optional


class BaseTextGeneration(abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        n_responses: int,
        max_tokens: int,
        seed: Optional[int] = None,
    ) -> list[str | None]:
        pass
