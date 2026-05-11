from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional, Generator


@dataclass
class Sample:
    audio: np.ndarray
    sample_rate:int
    label: str

class Dataset(ABC):
    def __init__(self, name:str, dataset_path:Optional[str]=None):
        self.name=name
        self.data_path=dataset_path
    
    @abstractmethod
    def load(self)->Generator[Sample, None, None]:
        pass

