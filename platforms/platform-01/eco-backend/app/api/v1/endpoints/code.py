# =============================================================================
# Code Engineering Endpoints
# =============================================================================

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import get_current_user_id, get_optional_user_id
from app.core.logging import get_logger
from app.core.exceptions import ProviderError
from app.db.base import get_db
from app.schemas.base import ResponseSchema
from app.schemas.platform import (
    CompleteCodeRequest, CompleteCodeResponse,
    ExplainCodeRequest, ExplainCodeResponse,
    ReviewCodeRequest, ReviewCodeResponse
)

router = APIRouter()
logger = get_logger("code")


@router.post("/complete", response_model=ResponseSchema[CompleteCodeResponse])
async def complete_code(
    request: Request,
    complete_request: CompleteCodeRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """代碼補全"""
    
    provider = complete_request.provider or "zeta-code"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("api_key"):
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "code_completion_requested",
        provider=provider,
        language=complete_request.language,
        code_length=len(complete_request.code),
        user_id=user_id,
    )
    
    # 模擬補全響應
    completions = [
        "    return fibonacci(n-1) + fibonacci(n-2)",
        "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    ]
    
    return ResponseSchema(
        data=CompleteCodeResponse(
            completions=completions,
            suggested_code=completions[1],
            provider=provider,
        )
    )


@router.post("/explain", response_model=ResponseSchema[ExplainCodeResponse])
async def explain_code(
    request: Request,
    explain_request: ExplainCodeRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """代碼解釋"""
    
    provider = explain_request.provider or "zeta-code"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("api_key"):
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "code_explanation_requested",
        provider=provider,
        language=explain_request.language,
        code_length=len(explain_request.code),
        user_id=user_id,
    )
    
    return ResponseSchema(
        data=ExplainCodeResponse(
            explanation="This function implements the quicksort algorithm, which is a divide-and-conquer sorting algorithm.",
            key_points=[
                "Uses recursion to sort subarrays",
                "Selects a pivot element",
                "Partitions array into elements less than, equal to, and greater than pivot",
            ],
            provider=provider,
        )
    )


@router.post("/review", response_model=ResponseSchema[ReviewCodeResponse])
async def review_code(
    request: Request,
    review_request: ReviewCodeRequest,
    user_id: str = Depends(get_optional_user_id),
    db: AsyncSession = Depends(get_db)
):
    """代碼審查"""
    
    provider = review_request.provider or "zeta-code"
    config = settings.get_provider_configs().get(provider)
    
    if not config or not config.get("api_key"):
        raise ProviderError(provider, "Configuration not found")
    
    logger.info(
        "code_review_requested",
        provider=provider,
        language=review_request.language,
        review_type=review_request.review_type,
        code_length=len(review_request.code),
        user_id=user_id,
    )
    
    # 根據審查類型返回不同的響應
    if review_request.review_type == "security":
        issues = [
            {
                "severity": "high",
                "line": 2,
                "message": "SQL Injection vulnerability detected. Use parameterized queries.",
                "rule": "sql-injection",
            }
        ]
        score = 40
    else:
        issues = [
            {
                "severity": "low",
                "line": 1,
                "message": "Consider adding type hints",
                "rule": "type-hints",
            }
        ]
        score = 85
    
    return ResponseSchema(
        data=ReviewCodeResponse(
            issues=issues,
            suggestions=["Use ORM instead of raw SQL", "Add input validation"],
            score=score,
            provider=provider,
        )
    )
