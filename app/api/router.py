from fastapi import APIRouter, Body
from app.schemas.recipe import ChatRequest, ChatResponse, RecipeRecommendation
from app.services import bedrock_service, db_service

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, tags=["1. Bedrock 챗봇"])
async def handle_chat(request: ChatRequest):
    """
    (기능 1) Bedrock 챗봇 API
    유저의 쿼리와 재료를 받아 Bedrock으로 레시피를 생성
    """
    response = await bedrock_service.generate_recipe_response(
        user_query=request.user_query,
        ingredients=request.ingredients
    )
    
    return response


@router.get("/recommend", response_model=RecipeRecommendation)
async def get_hot_recommendations():
    """
    (기능 2) Hot K-Food 추천 API
    DB(SQLite)에 저장된 Top 15 중 3~4개를 랜덤으로 추천
    """
    recommendations = await db_service.get_random_recommendations(
        total=15, 
        sample_size=4 # 4개 추천
    )
    return RecipeRecommendation(recommendations=recommendations)