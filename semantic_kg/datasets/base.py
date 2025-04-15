import abc
from pathlib import Path

import pandas as pd


class BaseDatasetLoader:
    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"{data_dir} is not a directory")

    @abc.abstractmethod
    def load(self) -> pd.DataFrame:
        pass
