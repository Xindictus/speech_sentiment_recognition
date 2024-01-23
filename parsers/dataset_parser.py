# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from os.path import exists


class DatasetParser(ABC):
    def is_valid_path(self, path: str) -> bool:
        """_summary_

        Args:
            path (str): Absolute path to file/folder

        Returns:
            bool: Informs of path validity
        """
        return exists(path)

    @abstractmethod
    def parse(self) -> None:
        """_summary_
        This function will include the logic of parsing
        the Kaggle audio files and creating a dataset
        out of them
        """        
        pass
