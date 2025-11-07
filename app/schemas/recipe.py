from pydantic import BaseModel, Field
from typing import List, Optional


# --- 1. /chat (Bedrock 챗봇)용 모델 ---

class ChatRequest(BaseModel):
    ingredients: List[str] = Field(
        max_length=3,
        description="사용자가 가진 재료 (필수, 최대 3개)"
    )

class ChatPreviewInfo(BaseModel):
    total_time: str
    ingredients: List[str]

class ChatResponse(BaseModel):
    full_recipe: str  # <recipe>...</recipe> XML 템플릿
    preview: Optional[ChatPreviewInfo] = None # 미리보기 정보


# --- 2. /hot-recipes (Reddit 랭킹)용 모델 ---
class HotRecipe(BaseModel):
    ranking: int
    recipe_name: str
    score: int
    recipe_detail_ko: Optional[str] = None
    recipe_detail_en: Optional[str] = None
    image_url: Optional[str] = None

    class Config:
        orm_mode = True


# --- 3. /top-ingredients (마트 판매 랭킹)용 모델 ---
class TopIngredient(BaseModel):
    ranking: int
    ingredient_name: str
    total_quantity: int

    class Config:
        orm_mode = True