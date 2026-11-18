"""批量将条款 PDF 入库：子块写入 Milvus，父块写入 PostgreSQL。

在 agent-service 目录下执行：
    uv run python scripts/ingest_kb.py
"""

import asyncio
import sys
from pathlib import Path

# 将项目根目录（agent-service）加入 sys.path，保证能导入 app 包
for parent in Path(__file__).resolve().parents:
    if (parent / "pyproject.toml").exists():
        sys.path.insert(0, str(parent))
        break

from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_ollama import OllamaEmbeddings

from app.core.config import APP_ROOT, settings
from app.core.logging import configure_logging, get_logger
from app.infra.database import AsyncSessionFactory, close_database
from app.modules.product.service import ProductService
from app.rag.pipeline import RAGPipeline


def build_vector_store() -> Milvus:
    """构建与 notebook 一致的 Milvus 向量库（稠密 + 稀疏）。"""
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b",
        dimensions=1024,
    )
    return Milvus(
        embedding_function=embeddings,
        collection_name="insurance_collection",
        builtin_function=BM25BuiltInFunction(
            analyzer_params={"type": "chinese"}
        ),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.rag.milvus_url},
        auto_id=True,
    )


async def main() -> int:
    configure_logging(settings.logging.level)
    logger = get_logger(__name__)

    # 1.查询上架产品
    async with AsyncSessionFactory() as session:
        products = await ProductService(session).list_product()

    # 2.遍历产品，逐个入库
    logger.info("开始批量入库", total=len(products))

    pipeline = RAGPipeline(build_vector_store())
    succeeded: list[str] = []
    failed: list[str] = []
    try:
        for index, product in enumerate(products, start=1):
            file_path = Path(
                APP_ROOT / "app" / "rag" / "data" / "kb" / product.clause_name
            )
            if not file_path.is_file():
                logger.warning(
                    "条款文件不存在，跳过",
                    product=product.name,
                    file=str(file_path),
                )
                continue

            logger.info(
                "开始入库",
                progress=f"{index}/{len(products)}",
                product=product.name,
                file=file_path.name,
            )
            try:
                await pipeline.ingest(product)
            except Exception:
                logger.exception("入库失败", file=file_path.name, product=product.name)
                failed.append(file_path.name)
            else:
                logger.info("入库完成", product=product.name, file=file_path.name)
                succeeded.append(file_path.name)
    finally:
        pipeline.close()
        await close_database()

    logger.info("批量入库结束", success=len(succeeded), failed=len(failed))
    if failed:
        logger.error("以下文件入库失败", files=failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
