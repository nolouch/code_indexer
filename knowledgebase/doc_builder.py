import json
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
from math import ceil
import numpy as np
import time
from knowledgebase.doc_model import (
    Concept,
    KnowledgeBlock,
    SourceData,
    Relationship,
    SubconceptKnowledgeMapping,
)
from utils.json_utils import extract_json_array, extract_json
from utils.token import calculate_tokens
from setting.db import SessionLocal
from llm.factory import LLMInterface
from llm.embedding import get_text_embedding


class DocBuilder:
    """
    A builder class for constructing knowledge graphs from documents.
    """

    def __init__(
        self,
        llm_client: LLMInterface,
        embedding_func: Callable[[str], np.ndarray] = None,
    ):
        """
        Initialize the builder with a graph instance and specifications.
        """
        self.embedding_func = embedding_func
        self.llm_client = llm_client

    def _default_embedding_func(self, text: str) -> np.ndarray:
        return get_text_embedding(text, "text-embedding-3-small")

    def _read_file(self, file_path: str) -> str:
        """
        Read a file and return its contents.

        Parameters:
        - file_path: Path to the file

        Returns:
        - File contents as string
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_file_info(self, file_path: str) -> Tuple[str, str]:
        """
        Extract file name and extension from a file path.

        Parameters:
        - file_path: Path to the file

        Returns:
        - Tuple of (file_name, file_extension)
        """
        path = Path(file_path)
        return path.stem, path.suffix

    def split_markdown_by_heading(
        self, path: str, metadata: Dict[str, Any], heading_level=2
    ):
        # Split content by lines
        name, extension = self._extract_file_info(path)
        doc_content = self._read_file(path)
        doc_version = metadata.get("doc_version", "1.0")

        markdown_content = self._read_file(path)
        lines = markdown_content.split("\n")

        sections = {}
        current_section = []
        current_title = "default"

        for line in lines:
            # Check if line is a h2 heading
            if line.startswith("#" * heading_level + " "):
                # Save current section if it has content
                if current_section:
                    sections[current_title] = "\n".join(current_section)
                    current_section = []

                # Update current title (remove '## ' prefix)
                current_title = line[3:].strip()
            else:
                current_section.append(line)

        # Save the last section
        if current_section:
            sections[current_title] = "\n".join(current_section)

        for heading, content in sections.items():
            tokens = calculate_tokens(content)
            if tokens > 4096:
                raise ValueError(
                    f"Heading {heading} has {tokens} tokens, please split it into smaller chunks manually"
                )

        # Add document and knowledge blocks to database
        with SessionLocal() as db:
            source_data = db.query(SourceData).filter(SourceData.link == path).first()
            if not source_data:
                source_data = SourceData(
                    name=name,
                    content=markdown_content,
                    link=path,
                    version=doc_version,
                    data_type="document",
                    metadata=metadata,
                )
                db.add(source_data)
                db.flush()
                source_data_id = source_data.id
                print(f"Source data created for {path}, id: {source_data_id}")
            else:
                print(f"Source data already exists for {path}, id: {source_data.id}")
                source_data_id = source_data.id

            knowledge_blocks = (
                db.query(KnowledgeBlock)
                .filter(
                    KnowledgeBlock.source_id == source_data_id,
                    KnowledgeBlock.knowledge_type == "paragraph",
                    KnowledgeBlock.source_version == doc_version,
                )
                .all()
            )
            if knowledge_blocks:
                print(f"Knowledge blocks already exist for {path}")
                return sections

            for heading, content in sections.items():
                kb = KnowledgeBlock(
                    name=heading,
                    content="#" * heading_level + f" {heading}\n{content}",
                    knowledge_type="paragraph",
                    content_vec=self.embedding_func(content),
                    source_version=metadata.get("doc_version", "1.0"),
                    source_id=source_data_id,
                )
                db.add(kb)

            db.commit()

        return sections

    def extract_qa_blocks(
        self, file_path: Union[str, List[str]], metadata: Dict[str, Any]
    ) -> List[KnowledgeBlock]:
        # Handle single file or list of files
        if isinstance(file_path, str):
            file_paths = [file_path]
        else:
            file_paths = file_path

        doc_version = metadata.get("doc_version", "1.0")
        blocks = []
        for path in file_paths:
            # Create document entity
            name, extension = self._extract_file_info(path)
            doc_content = self._read_file(path)

            # Extract knowledge blocks using LLM
            prompt_template = self.graph_spec.get_extraction_prompt(
                "knowledge_qa_extraction"
            )
            prompt = prompt_template.format(text=doc_content)
            response = self.llm_client.generate(prompt)

            try:
                response_json_str = extract_json_array(response)
                # Parse JSON response
                extracted_blocks = json.loads(response_json_str)

                with SessionLocal() as db:
                    source_data = (
                        db.query(SourceData).filter(SourceData.link == path).first()
                    )
                    if not source_data:
                        source_data = SourceData(
                            name=name,
                            content=doc_content,
                            link=path,
                            version=doc_version,
                            data_type="document",
                            metadata=metadata,
                        )
                        db.add(source_data)
                        db.flush()
                        source_data_id = source_data.id
                        print(f"Source data created for {path}, id: {source_data_id}")
                    else:
                        print(
                            f"Source data already exists for {path}, id: {source_data.id}"
                        )
                        source_data_id = source_data.id

                    knowledge_blocks = (
                        db.query(KnowledgeBlock)
                        .filter(
                            KnowledgeBlock.source_id == source_data_id,
                            KnowledgeBlock.knowledge_type == "qa",
                            KnowledgeBlock.source_version == doc_version,
                        )
                        .all()
                    )
                    if knowledge_blocks:
                        print(f"Knowledge blocks already exist for {path}")
                        continue

                    # Create and add knowledge blocks
                    for block_data in extracted_blocks:
                        question = block_data.get("question", "")
                        answer = block_data.get("answer", "")
                        qa_content = question + "\n" + answer
                        qa_block = KnowledgeBlock(
                            name=question,
                            definition=qa_content,
                            source_version=doc_version,
                            source_id=source_data_id,
                            knowledge_type="qa",
                            content_vec=self.embedding_func(qa_content),
                        )
                        db.add(qa_block)
                        blocks.append(qa_content)
                    db.commit()

            except (json.JSONDecodeError, TypeError):
                print(f"Failed to parse knowledge blocks from {path}")

        return blocks

    def agument_block_context(self, knowledge_blocks: dict) -> dict:
        CHUNK_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>

Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

        block_context = {}
        for kb in knowledge_blocks:
            prompt = CHUNK_CONTEXT_PROMPT.format(
                doc_content=kb["source_content"], chunk_content=kb["content"]
            )

            retry_count = 0
            while retry_count < 3:
                try:
                    response = self.llm_client.generate(prompt)
                    block_context[kb["id"]] = response
                    print(f"Generated context for {kb['id']}")
                    break
                except Exception as e:
                    print(f"Failed to generate context for {kb['id']}, error: {e}")
                    time.sleep(60)
                    retry_count += 1

        with SessionLocal() as db:
            for id, context in block_context.items():
                kb = db.query(KnowledgeBlock).filter(KnowledgeBlock.id == id).first()
                kb.context = context
                kb.context_vec = self.embedding_func(context)
                db.add(kb)

            db.commit()

        return block_context

    def analyze_concepts(self, concept_file: Optional[str] = None) -> List[Concept]:
        concepts = []

        # Load predefined concepts if provided
        if concept_file:
            try:
                with open(concept_file, "r") as f:
                    predefined_concepts = json.load(f)

                with SessionLocal() as db:
                    for concept_data in predefined_concepts:
                        concept = Concept(
                            name=concept_data.get("name", ""),
                            definition=concept_data.get("definition", ""),
                            definition_vec=self.embedding_func(
                                concept_data.get("definition", "")
                            ),
                            version=concept_data.get("version", "1.0"),
                        )
                        db.add(concept)

                    db.commit()

                return predefined_concepts
            except Exception as e:
                raise ValueError(f"Error loading concepts from file: {e}")

        with SessionLocal() as db:
            knowledge_blocks = (
                db.query(KnowledgeBlock.name, KnowledgeBlock.content).filter().all()
            )
            if not knowledge_blocks:
                print("No knowledge blocks available for concept extraction")
                return []

            # split knowledges block into batches that have 10000 tokens
            total_tokens = 0
            for i, kb in enumerate(knowledge_blocks):
                tokens = calculate_tokens(kb.content)
                total_tokens += tokens

            knowledge_blocks_batches = []
            index = 0
            batch_size = ceil(total_tokens / (ceil(total_tokens / 10000)))
            for kb in knowledge_blocks:
                if total_tokens > batch_size:
                    knowledge_blocks_batches.append(knowledge_blocks[index : i + 1])
                    index = i + 1
                    total_tokens = 0

            if index < len(knowledge_blocks):
                knowledge_blocks_batches.append(knowledge_blocks[index:])

            print(f"Splitted {len(knowledge_blocks_batches)} batches")

            for batches in knowledge_blocks_batches:
                # Combine knowledge blocks for analysis
                combined_blocks = "\n\n".join(
                    [f"Block: {kb.name}\nContent: {kb.content}" for kb in batches]
                )

                if combined_blocks.strip() == "":
                    print(f"Skipping empty batch")
                    continue

                # Extract concepts using LLM
                prompt_format = self.graph_spec.get_extraction_prompt(
                    "concept_extraction"
                )
                prompt = prompt_format.format(text=combined_blocks)
                try:
                    response = self.llm_client.generate(prompt)
                except Exception as e:
                    print(
                        f"Failed to extract concepts from {combined_blocks}, error: {e}"
                    )
                    import time

                    time.sleep(60)
                    response = self.llm_client.generate(prompt)

                try:
                    response_json_str = extract_json_array(response)
                    # Parse JSON response
                    extracted_concepts = json.loads(response_json_str)

                    # Create and add concepts
                    for concept_data in extracted_concepts:
                        concept = Concept(
                            name=concept_data.get("name", ""),
                            definition=concept_data.get("definition", ""),
                            definition_vec=self.embedding_func(
                                concept_data.get("definition", "")
                            ),
                            version="1.0",
                        )
                        db.add(concept)
                    concepts.append(extracted_concepts)
                    print(f"Extracted {len(extracted_concepts)} concepts")
                except (json.JSONDecodeError, TypeError):
                    print("Failed to parse concepts from LLM response")
            db.commit()

        return concepts
