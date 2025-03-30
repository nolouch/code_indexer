from sqlalchemy import func
from setting.db import SessionLocal
from knowledgebase.doc_model import KnowledgeBlock, Concept
from llm.embedding import get_text_embedding
from typing import List


def search_knowledge_blocks(
    query: str, top_k: int = 10, context_weight: float = 0.5
) -> List[KnowledgeBlock]:
    query_vector = get_text_embedding(query, "text-embedding-3-small")
    result = {
        "concepts": [],
        "knowledge_blocks": [],
    }
    with SessionLocal() as db:
        blocks = (
            db.query(
                KnowledgeBlock.id,
                KnowledgeBlock.name,
                KnowledgeBlock.content,
                KnowledgeBlock.context,
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
            .limit(top_k)
            .all()
        )

        # search most relevant concepts
        concepts = (
            db.query(
                Concept.name,
                Concept.definition,
                Concept.definition_vec.cosine_distance(query_vector).label("distance"),
                Concept.source_ids,
            )
            .order_by("distance")
            .limit(top_k)
            .all()
        )

        for concept in concepts:
            result["concepts"].append(
                {
                    "name": concept.name,
                    "definition": concept.definition,
                    "source_ids": concept.source_ids,
                    "distance": concept.distance,
                }
            )

            for block in blocks:
                if block.id in concept.source_ids:
                    result["knowledge_blocks"].append(
                        {
                            "name": block.name,
                            "content": block.content,
                            "context": block.context,
                            "distance": block.combined_distance,
                        }
                    )

            missing_source_ids = []
            for source_id in concept.source_ids:
                for block in blocks:
                    if block.id == source_id:
                        result["knowledge_blocks"].append(block)
                    else:
                        missing_source_ids.append(source_id)

            # query missing source ids
            if len(missing_source_ids) > 0:
                missing_blocks = (
                    db.query(
                        KnowledgeBlock.id,
                        KnowledgeBlock.name,
                        KnowledgeBlock.content,
                        KnowledgeBlock.context,
                        KnowledgeBlock.content_vec.cosine_distance(query_vector).label(
                            "distance"
                        ),
                    )
                    .filter(KnowledgeBlock.id.in_(missing_source_ids))
                    .all()
                )
                for block in missing_blocks:
                    result["knowledge_blocks"].append(
                        {
                            "name": block.name,
                            "content": block.content,
                            "context": block.context,
                            "distance": block.distance,
                        }
                    )

        return result
