from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # .env 파일에서 읽어올 변수들
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"

    # DB 파일 경로
    DB_PATH: str = "kfood_recipes.db"

    class Config:
        env_file = ".env" # .env 파일을 읽도록 설정

# settings 객체를 생성. 다른 파일에서 이 객체를 import해서 사용
settings = Settings()