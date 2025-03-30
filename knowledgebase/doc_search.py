from sqlalchemy import func
from setting.db import SessionLocal
from knowledgebase.doc_model import KnowledgeBlock
from llm.embedding import get_text_embedding
from typing import List


def search_knowledge_blocks(
    query: str, limit: int = 10, context_weight: float = 0.5
) -> List[KnowledgeBlock]:
    query_vector = get_text_embedding(query, "text-embedding-3-small")
    with SessionLocal() as db:
        blocks = (
            db.query(
                KnowledgeBlock.name,
                KnowledgeBlock.content,
                KnowledgeBlock.content_vec.cosine_distance(query_vector).label(
                    "content_distance"
                ),
                KnowledgeBlock.context_vec.cosine_distance(query_vector).label(
                    "context_distance"
                ),
                # Weighted combined distance (0.5 each)
                (
                    context_weight
                    * KnowledgeBlock.content_vec.cosine_distance(query_vector)
                    + (1 - context_weight)
                    * KnowledgeBlock.context_vec.cosine_distance(query_vector)
                ).label("combined_distance"),
            )
            .order_by("combined_distance")
            .limit(10)
            .all()
        )

        return blocks
