import json
import pandas as pd
from typing import List

class DataLoader:
    """Load CSV / JSON / TXT datasets"""

    @staticmethod
    def load_csv(path: str, text_column: str | None = None) -> List[str]:
        df = pd.read_csv(path)

        if text_column and text_column in df.columns:
            return df[text_column].dropna().astype(str).tolist()

        return df.astype(str).agg(" ".join, axis=1).tolist()

    @staticmethod
    def load_txt(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [f.read()]

    @staticmethod
    def load_json(path: str, text_field: str | None = None) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            if text_field:
                return [str(item.get(text_field, "")) for item in data]
            return [json.dumps(item) for item in data]

        return [json.dumps(data)]
