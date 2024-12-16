import abc


class BaseTextGeneration(abc.ABC):
    @abc.abstractmethod
    def generate(
        self, prompt: str, n_responses: int, max_tokens: int
    ) -> list[str | None]:
        pass
