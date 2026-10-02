"""Project settings loaded from .env (nearest one walking up from the working directory)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    alpaca_api_key_id: str = ""
    alpaca_api_secret_key: str = ""
    alpaca_data_url: str = "https://data.alpaca.markets"
    # sip = consolidated tape (free plan: only data older than 15 min, fine for past daily bars); iex = IEX only.
    alpaca_feed: str = "sip"
    fmp_api_key: str = ""
    fmp_base_url: str = "https://financialmodelingprep.com/stable"
    fmp_daily_budget: int = 240
    fmp_statement_limit: int = 5  # free plan: at most the latest 5 periods
    openai_model_fast: str = "gpt-4.1-mini"
    openai_model_full: str = "gpt-4.1"
    typesafe_api_key: str = ""
    typesafe_base_url: str = "https://openrouter.ai/api"
    jev_model: str = "typesafe/jev-1.13"
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


def load_settings() -> Settings:
    load_dotenv(find_dotenv(usecwd=True))
    env = {
        # Alpaca's own SDKs read APCA_API_KEY_ID / APCA_API_SECRET_KEY: accept those too.
        "alpaca_api_key_id": os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID"),
        "alpaca_api_secret_key": os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY"),
        "alpaca_data_url": os.getenv("ALPACA_DATA_URL"),
        "alpaca_feed": os.getenv("ALPACA_DATA_FEED"),
        "fmp_api_key": os.getenv("FMP_API_KEY"),
        "fmp_base_url": os.getenv("FMP_BASE_URL"),
        "fmp_daily_budget": os.getenv("FMP_DAILY_BUDGET"),
        "fmp_statement_limit": os.getenv("FMP_STATEMENT_LIMIT"),
        "openai_model_fast": os.getenv("OPENAI_MODEL_FAST"),
        "openai_model_full": os.getenv("OPENAI_MODEL_FULL"),
        # Jev goes through OpenRouter: an OpenRouter key works when TYPESAFE_API_KEY is not set.
        "typesafe_api_key": os.getenv("TYPESAFE_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
        "typesafe_base_url": os.getenv("TYPESAFE_BASE_URL"),
        "jev_model": os.getenv("JEV_MODEL"),
        "data_dir": os.getenv("JEVBT_DATA_DIR"),
    }
    return Settings(**{k: v for k, v in env.items() if v})
