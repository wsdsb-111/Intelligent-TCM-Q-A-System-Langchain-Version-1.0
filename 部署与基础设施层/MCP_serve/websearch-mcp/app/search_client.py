import os
from typing import List, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class BochaSearchRequest(BaseModel):
    query: str = Field(..., description="搜索关键词")
    limit: int = Field(5, ge=1, le=20, description="返回结果数量，默认 5")
    include: Optional[str] = Field(
        None,
        description="限定搜索域名，多个域名使用 `|` 分隔，如 `zhihu.com|baidu.com`",
    )


class BochaSearchResult(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""


class BochaSearchClient:
    _api_key = os.getenv("BOCHA_API_KEY")
    _search_url = os.getenv("BOCHA_SEARCH_URL", "https://api.bocha-ai.com/search")
    _timeout = float(os.getenv("BOCHA_SEARCH_TIMEOUT", "10"))

    @classmethod
    def search(cls, request: BochaSearchRequest) -> List[BochaSearchResult]:
        if not cls._api_key:
            raise ValueError("未配置 BOCHA_API_KEY，无法调用博查搜索 API")

        headers = {
            "Authorization": f"Bearer {cls._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": request.query,
            "limit": request.limit,
            "include": request.include,
        }

        response = requests.post(
            cls._search_url,
            json=payload,
            headers=headers,
            timeout=cls._timeout,
        )
        response.raise_for_status()

        results = response.json().get("results", [])

        return [
            BochaSearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
            )
            for item in results
        ]


