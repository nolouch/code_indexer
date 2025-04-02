import json
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from sqlalchemy.exc import IntegrityError
import uuid

from llm.factory import LLMInterface
from knowledgebase.doc_model import BestPractice
from setting.db import SessionLocal
from utils.json_utils import extract_json
from knowledgebase.doc_spec import gen_pr_review_best_practices_prompt


class BestPracticesKnowledgeBase:
    """
    Manages and manipulates a hierarchical tree of tags and associated best practices.
    Provides functionalities to retrieve, modify, and query tag data efficiently.
    """

    def __init__(
        self, llm_client: LLMInterface, embedding_func: Callable[[str], np.ndarray]
    ):
        self.llm_client = llm_client
        self.embedding_func = embedding_func

    def find_best_practices(
        self, query: str, top_k: int = 10
    ) -> Tuple[List[str], Optional[Dict], Optional[str]]:
        query_vector = self.embedding_func(query)
        result = {}
        with SessionLocal() as db:
            best_practices = (
                db.query(
                    BestPractice.id,
                    BestPractice.source_id,
                    BestPractice.tag,
                    BestPractice.guideline,
                    BestPractice.guideline_vec.cosine_distance(query_vector).label(
                        "distance"
                    ),
                )
                .order_by("distance")
                .limit(top_k)
                .all()
            )

            for bp in best_practices:
                if bp.source_id not in result:
                    result[bp.source_id] = []
                result[bp.source_id].append(
                    {
                        "tag": bp.tag,
                        "guideline": bp.guideline,
                        "distance": bp.distance,
                    }
                )

        return result

    def add_pr_review_best_practices(self, source_id: str, content: str) -> List[str]:
        """
        Generates a best practices for the given content.

        Args:
            llm_client (LLMInterface): The LLM client to use.
            namespace (str): The namespace to use.
            content (str): The content to generate a tag path for.

        Returns:
            List[str]: A list of tag names from root to leaf.
        """

        with SessionLocal() as session:
            bps = session.query(BestPractice).filter_by(source_id=source_id).all()
            if bps:
                raise ValueError(
                    f"Best practices already exist for source_id: {source_id}"
                )

        prompt = gen_pr_review_best_practices_prompt.format(pr_review_comments=content)

        # Call LLM to get classification
        response = self.llm_client.generate(prompt)
        # Parse the LLM response to extract tag path
        try:
            bp_response = extract_json(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse tag path JSON: {e}")

        bp_result = json.loads(bp_response)

        with SessionLocal() as session:
            try:
                for best_practice in bp_result.get("best_practices", []):
                    tag_path = best_practice.get("tag", None)
                    if tag_path is None:
                        raise ValueError("Tag path is required.")

                    print(f"insert best practice: {best_practice}")
                    session.add(
                        BestPractice(
                            id=str(uuid.uuid4()),
                            tag=tag_path,
                            source_id=source_id,
                            guideline=best_practice,
                            guideline_vec=self.embedding_func(
                                json.dumps(best_practice)
                            ),
                        )
                    )
            except IntegrityError as e:
                session.rollback()
                raise ValueError(f"Failed to insert best practice: {e}")
            except Exception as e:
                session.rollback()
                raise e

            session.commit()

        return bp_result
