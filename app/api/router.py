from fastapi import APIRouter, Body
from typing import List
from app.schemas.recipe import ChatRequest, ChatResponse, HotRecipe, TopIngredient
from app.services import bedrock_service, db_service

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, tags=["Top Ingredients"])
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

@router.get("/hot-recipes", response_model=List[HotRecipe], tags=["Hot Recipes"])
async def get_hot_recipes():
    """
    (기능 2) Hot K-Food 추천 API
    DB(SQLite)에 저장된 Top 15 메뉴 중 랜덤 4개를 조회
    """
    recipes = await db_service.get_hot_recipes_from_db(limit=15)
    return recipes

@router.get("/top-ingredients", response_model=List[TopIngredient], tags=["Top Ingredients"])
async def get_top_ingredients():
    """
    (기능 3) Grocery 추천 API
    DB(SQLite)에 저장된 Top 10 재료를 조회
    """
    ingredients = await db_service.get_top_ingredients_from_db(limit=10)
    return ingredients