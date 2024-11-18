import pandas as pd
from typing import List


class FeatureTransformation:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def drop_columns(self, columns: List[str]) -> pd.DataFrame:
        return self.df.drop(columns=columns, axis=1)
