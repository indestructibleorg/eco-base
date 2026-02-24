# =============================================================================
# Pagination Utilities
# =============================================================================
# 游標分頁實現
# =============================================================================

import base64
import json
from typing import TypeVar, Generic, List, Optional, Any, Dict
from dataclasses import dataclass
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

T = TypeVar('T', bound=DeclarativeBase)


@dataclass
class CursorPaginationResult(Generic[T]):
    """游標分頁結果"""
    items: List[T]
    next_cursor: Optional[str]
    prev_cursor: Optional[str]
    has_more: bool
    total: Optional[int] = None


class CursorPaginator(Generic[T]):
    """
    游標分頁器
    
    使用游標而非偏移量進行分頁，適合大數據量和實時數據場景
    """
    
    def __init__(
        self,
        model_class: type[T],
        cursor_field: str = 'id',
        default_limit: int = 20,
        max_limit: int = 100,
    ):
        self.model_class = model_class
        self.cursor_field = cursor_field
        self.default_limit = default_limit
        self.max_limit = max_limit
    
    def _encode_cursor(self, value: Any) -> str:
        """編碼游標"""
        cursor_data = json.dumps({self.cursor_field: str(value)})
        return base64.urlsafe_b64encode(cursor_data.encode()).decode()
    
    def _decode_cursor(self, cursor: str) -> Dict[str, Any]:
        """解碼游標"""
        try:
            cursor_data = base64.urlsafe_b64decode(cursor.encode()).decode()
            return json.loads(cursor_data)
        except Exception:
            return {}
    
    async def paginate(
        self,
        session: AsyncSession,
        cursor: Optional[str] = None,
        prev_cursor: Optional[str] = None,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by = None,
    ) -> CursorPaginationResult[T]:
        """
        執行游標分頁查詢
        
        Args:
            session: 數據庫會話
            cursor: 下一頁游標
            prev_cursor: 上一頁游標
            limit: 每頁數量
            filters: 過濾條件
            order_by: 排序條件
            
        Returns:
            分頁結果
        """
        # 限制每頁數量
        page_limit = min(limit or self.default_limit, self.max_limit)
        
        # 構建基礎查詢
        query = select(self.model_class)
        
        # 應用過濾條件
        if filters:
            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    query = query.where(getattr(self.model_class, field) == value)
        
        # 應用游標條件
        cursor_value = None
        is_forward = True
        
        if cursor:
            # 向前翻頁
            decoded = self._decode_cursor(cursor)
            cursor_value = decoded.get(self.cursor_field)
            if cursor_value:
                query = query.where(getattr(self.model_class, self.cursor_field) > cursor_value)
                is_forward = True
                
        elif prev_cursor:
            # 向後翻頁
            decoded = self._decode_cursor(prev_cursor)
            cursor_value = decoded.get(self.cursor_field)
            if cursor_value:
                query = query.where(getattr(self.model_class, self.cursor_field) < cursor_value)
                is_forward = False
        
        # 應用排序
        if order_by is not None:
            query = query.order_by(order_by)
        else:
            query = query.order_by(getattr(self.model_class, self.cursor_field))
        
        # 查詢多一條用於判斷是否有更多
        query = query.limit(page_limit + 1)
        
        # 執行查詢
        result = await session.execute(query)
        items = list(result.scalars().all())
        
        # 判斷是否有更多
        has_more = len(items) > page_limit
        items = items[:page_limit]
        
        # 生成游標
        next_cursor = None
        prev_cursor_result = None
        
        if items:
            if has_more:
                last_item = items[-1]
                next_cursor = self._encode_cursor(getattr(last_item, self.cursor_field))
            
            # 生成上一頁游標
            if cursor:  # 當前在向前翻頁，需要生成 prev_cursor
                first_item = items[0]
                prev_cursor_result = self._encode_cursor(getattr(first_item, self.cursor_field))
        
        # 如果是向後翻頁，需要反轉結果
        if not is_forward:
            items = list(reversed(items))
            next_cursor, prev_cursor_result = prev_cursor_result, next_cursor
        
        return CursorPaginationResult(
            items=items,
            next_cursor=next_cursor,
            prev_cursor=prev_cursor_result,
            has_more=has_more,
        )
    
    async def get_total(self, session: AsyncSession, filters: Optional[Dict[str, Any]] = None) -> int:
        """獲取總數"""
        query = select(func.count()).select_from(self.model_class)
        
        if filters:
            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    query = query.where(getattr(self.model_class, field) == value)
        
        result = await session.execute(query)
        return result.scalar()


@dataclass
class OffsetPaginationResult(Generic[T]):
    """偏移分頁結果"""
    items: List[T]
    page: int
    page_size: int
    total: int
    pages: int
    has_next: bool
    has_prev: bool


class OffsetPaginator(Generic[T]):
    """
    偏移分頁器
    
    傳統的偏移量分頁，適合小數據量場景
    """
    
    def __init__(
        self,
        model_class: type[T],
        default_page_size: int = 20,
        max_page_size: int = 100,
    ):
        self.model_class = model_class
        self.default_page_size = default_page_size
        self.max_page_size = max_page_size
    
    async def paginate(
        self,
        session: AsyncSession,
        page: int = 1,
        page_size: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        order_by = None,
    ) -> OffsetPaginationResult[T]:
        """
        執行偏移分頁查詢
        
        Args:
            session: 數據庫會話
            page: 頁碼（從1開始）
            page_size: 每頁數量
            filters: 過濾條件
            order_by: 排序條件
            
        Returns:
            分頁結果
        """
        # 限制每頁數量
        size = min(page_size or self.default_page_size, self.max_page_size)
        
        # 確保頁碼有效
        if page < 1:
            page = 1
        
        # 計算總數
        total = await self.get_total(session, filters)
        
        # 計算總頁數
        pages = (total + size - 1) // size if total > 0 else 1
        
        # 構建查詢
        query = select(self.model_class)
        
        # 應用過濾條件
        if filters:
            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    query = query.where(getattr(self.model_class, field) == value)
        
        # 應用排序
        if order_by is not None:
            query = query.order_by(order_by)
        
        # 應用分頁
        offset = (page - 1) * size
        query = query.offset(offset).limit(size)
        
        # 執行查詢
        result = await session.execute(query)
        items = list(result.scalars().all())
        
        return OffsetPaginationResult(
            items=items,
            page=page,
            page_size=size,
            total=total,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1,
        )
    
    async def get_total(self, session: AsyncSession, filters: Optional[Dict[str, Any]] = None) -> int:
        """獲取總數"""
        query = select(func.count()).select_from(self.model_class)
        
        if filters:
            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    query = query.where(getattr(self.model_class, field) == value)
        
        result = await session.execute(query)
        return result.scalar()


def paginate_response(
    items: List[Any],
    page: int,
    page_size: int,
    total: int
) -> Dict[str, Any]:
    """
    構建分頁響應
    
    Args:
        items: 數據列表
        page: 當前頁
        page_size: 每頁數量
        total: 總數
        
    Returns:
        分頁響應字典
    """
    pages = (total + page_size - 1) // page_size if total > 0 else 1
    
    return {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": pages,
        }
    }
