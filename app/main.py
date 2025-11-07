from fastapi import FastAPI
from app.api import router # API 라우터 import

# FastAPI 앱 생성
app = FastAPI(
    title="K-Food Recipe Backend",
    description="Bedrock 챗봇과 DB(SQLite) 추천 기능을 제공하는 API",
    version="0.1.0"
)

# app/api/router.py에 정의된 엔드포인트들을 앱에 포함
app.include_router(router.router, prefix="/api")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "K-Food Recipe API에 오신 것을 환영합니다. /docs 로 이동하세요."}

# Uvicorn으로 이 파일을 실행: uvicorn app.main:app --reload --port 8000