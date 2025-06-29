from pydantic import BaseModel
from typing import Iterable


class PredictRequest(BaseModel):
    text: str


def validate_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
