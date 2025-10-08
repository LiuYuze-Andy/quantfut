from __future__ import annotations
import pandas as pd
from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Return Series indexed by datetime with values in {-1,0,+1}.
        WHY: Decouple signal generation from execution & portfolio.
        """
        raise NotImplementedError
