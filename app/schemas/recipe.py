from pydantic import BaseModel
from typing import List, Optional

# --- /chat 엔드포인트용 ---
class ChatRequest(BaseModel):
    user_query: str
    ingredients: Optional[List[str]] = None # 유저가 가진 재료 (선택)

# 미리보기: 조리 시간, 재료
class ChatPreviewInfo(BaseModel):
    total_time: str
    ingredients: List[str]

class ChatResponse(BaseModel):
    full_recipe: str
    preview: Optional[ChatPreviewInfo] = None
    
# --- /recommend 엔드포인트용 ---
class RecipeRecommendation(BaseModel):
    recommendations: List[str]