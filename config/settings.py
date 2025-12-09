import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

class Settings:
    @staticmethod
    def _get(key, default=None):
        # 1. Streamlit secrets
        if key in st.secrets:
            return st.secrets[key]
        # 2. Fallback to environment variables
        return os.getenv(key, default)

    AZURE_OPENAI_API_KEY = _get.__func__("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = _get.__func__("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME = _get.__func__("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = _get.__func__("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    AZURE_SPEECH_KEY = _get.__func__("AZURE_SPEECH_KEY")
    AZURE_SPEECH_REGION = _get.__func__("AZURE_SPEECH_REGION")

    MAX_QUESTIONS = int(_get.__func__("MAX_QUESTIONS", 30))
    MIN_REQUIRED_TOPICS = int(_get.__func__("MIN_REQUIRED_TOPICS", 9))
    CONVERSATION_TIMEOUT_MINUTES = int(_get.__func__("CONVERSATION_TIMEOUT_MINUTES", 15))

    @classmethod
    def validate(cls):
        required = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_SPEECH_KEY",
            "AZURE_SPEECH_REGION",
        ]

        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        return True


settings = Settings()
settings.validate()