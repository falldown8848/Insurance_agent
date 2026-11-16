from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.modules.chat_thread.exceptions import ApplicationError
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# 初始化日志系统
configure_logging(settings.logging.level)

logger = get_logger(__name__)


# 生命周期管理（预留）
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"【{settings.app.name}】应用启动中...")
    from app.infra.database import check_database, close_database
    try:
        await check_database()
        yield
    finally:
        logger.info("应用关闭中... ")
        await close_database()

app = FastAPI(
    title=settings.app.name,
    debug=settings.app.debug,
    lifespan=lifespan,
)

@app.get("/health", summary="健康检测接口")
async def health_check() -> dict[str, str]:
    logger.info("执行健康检查")
    return {"status": "ok"}

@app.exception_handler(ApplicationError)
def handle_exception(request: Request, exc: ApplicationError):
    logger.error(f"处理异常, 请求路径: {request.url}, 异常信息: {exc}")
    return JSONResponse(
        status_code=exc.status_code,  # http状态码
        content={
            "code": exc.code,        # 错误码， error_code
            "message": exc.message     # 错误消息，error_message
        }
    )

from app.modules.product.router import router as product_router
from app.modules.chat_thread.router import router as chat_thread_router
app.include_router(product_router)
app.include_router(chat_thread_router)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=False,
    )