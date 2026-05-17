"""
OpenAI MCP Server - FastMCP optimized

MCP protocol layer only. Business logic is in services.py
"""

from typing import Optional, Sequence
from fastmcp import FastMCP
import os
import sys


import dotenv

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

dotenv.load_dotenv()

from openai_mcp.services import chat as _chat_service, generate as _generate_service

# 从环境变量获取API密钥
API_KEY = os.getenv("OPENAI_API_KEY", "")

# 配置认证
auth = None
if os.getenv("AUTH_ENABLED", "false").lower() == "true":
    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
    
    bearer_token = os.getenv("BEARER_TOKEN", "")
    if bearer_token:
        auth = StaticTokenVerifier(
            tokens={
                bearer_token: {
                    "client_id": "mcp-client",
                    "scopes": ["read:data", "write:data"]
                }
            }
        )
        print(f"✓ Bearer Token 认证已启用")
    else:
        print("⚠ AUTH_ENABLED=true 但未设置 BEARER_TOKEN，认证已禁用")
else:
    print("ℹ 认证已禁用 (AUTH_ENABLED=false)")

# 创建FastMCP服务器实例
mcp = FastMCP("OpenAI MCP Server", auth=auth)


@mcp.tool()
async def chat(
    message: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Chat with text models.

    Args:
        message: User message
        model: Model to use
        system_prompt: Optional system prompt
    """
    result = await _chat_service(
        message=message,
        model=model,
        system_prompt=system_prompt,
    )
    return result["content"] if result["success"] else f"Error: {result['error']}"


@mcp.tool()
async def make_images(
    message: str,
    reference_images: Optional[Sequence[str]] = None,
    model: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
) -> dict:
    """
    Generate images (unified interface).

    Supports:
    - Text-only: leave reference_images empty
    - With reference: provide reference_images

    Args:
        message: Text prompt
        reference_images: Optional list of image paths
        model: Model to use
        aspect_ratio: Image aspect ratio
        resolution: Image resolution
    """
    result = await _generate_service(
        prompt=message,
        reference_images=reference_images,
        model=model or os.getenv("IMAGE_MODEL", "gemini-3-pro-image-preview"),
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )

    if result["success"]:
        return {
            "success": True,
            "image_path": result["image_data"]["path"],
            "model": result["model"],
        }
    return {"success": False, "error": result["error"]}



def main():
    """
    启动MCP服务器。

    支持通过以下环境变量配置：
    - OPENAI_API_KEY: API密钥
    - PORT: 端口号（默认8000，仅HTTP/SSE传输生效）
    - TRANSPORT: 传输方式（默认http，可选 http、sse、stdio）
    """
    transport = os.getenv("TRANSPORT", "http")

    if transport in ("http", "sse"):
        port = int(os.getenv("PORT", "8000"))
        mcp.run(transport=transport, port=port, host="0.0.0.0")
    elif transport == "stdio":
        mcp.run(transport="stdio")
    else:
        raise ValueError(f"不支持的传输方式: {transport}，可选值: http, sse, stdio")


if __name__ == "__main__":
    main()
