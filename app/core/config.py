from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # .env 파일에서 읽어올 변수들
    AWS_DEFAULT_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"

    REDDIT_CLIENT_ID: str
    REDDIT_CLIENT_SECRET: str
    REDDIT_PASSWORD: str
    REDDIT_USER_AGENT: str
    REDDIT_USERNAME: str

    KNOWLEDGE_BASE_ID: str

    # DB 파일 경로
    DB_PATH: str = "kfood_recipes.db"

    class Config:
        env_file = ".env" # .env 파일을 읽도록 설정

# settings 객체를 생성. 다른 파일에서 이 객체를 import해서 사용
settings = Settings()