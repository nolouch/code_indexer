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

    def print_tree_markdown(self, namespace: str) -> str:
        """
        Generate a markdown representation of the light tree for the given namespace.

        Args:
            namespace_name: The namespace to generate the tree for

        Returns:
            A string containing the markdown representation of the tree
        """
        light_tree = self.get_light_tree(namespace)
        if not light_tree:
            return f"No tree found for namespace: {namespace}"

        markdown_output = [f"# Tree for namespace: {namespace}\n"]

        def _print_node(node, depth=0):
            indent = "  " * depth
            node_name = node.get("tag", "Unnamed node")
            description = node.get("description", "")

            # Add description if available
            desc_text = f" - {description}" if description else ""
            markdown_output.append(f"{indent}- **{node_name}**{desc_text}")

            # Process children recursively
            children = node.get("children", [])
            for child in children:
                _print_node(child, depth + 1)

        # Process all root nodes
        for root_node in light_tree:
            _print_node(root_node)

        return "\n".join(markdown_output)

    def format_tree_as_xml(self, namespace_name: str) -> str:
        """
        Generate an XML representation of the tree.

        Args:
            namespace_name: The namespace to generate the tree for

        Returns:
            A string containing the XML representation of the tree
        """
        light_tree = self.get_light_tree(namespace_name)
        if not light_tree:
            return f'<tree namespace="{namespace_name}"></tree>'

        def _node_to_xml(node, indent="  "):
            # Get node properties
            tag_name = node.get("tag", "unnamed")
            description = node.get("description", "")

            # Escape special characters in attributes
            tag_escaped = (
                tag_name.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            if description:
                desc_escaped = (
                    description.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                )
            else:
                desc_escaped = ""

            # Start the node element
            xml_lines = [
                f'{indent}<node name="{tag_escaped}" description="{desc_escaped}">'
            ]

            # Add children recursively
            children = node.get("children", [])
            for child in children:
                child_xml = _node_to_xml(child, indent + "  ")
                xml_lines.extend(child_xml)

            # Close the node
            xml_lines.append(f"{indent}</node>")

            return xml_lines

        # Build the complete XML
        xml_lines = [f'<tree namespace="{namespace_name}">']

        # Process all root nodes
        for root_node in light_tree:
            root_xml = _node_to_xml(root_node, "  ")
            xml_lines.extend(root_xml)

        xml_lines.append("</tree>")

        return "\n".join(xml_lines)
