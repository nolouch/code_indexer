from llm.factory import LLMInterface
from typing import Any

qa_extraction_prompt = """You are an expert in knowledge extraction and question generation. Your goal is to read a document and create question-answer pairs that effectively capture the most important knowledge within it, addressing the likely interests of someone reading the document to learn.

Given the following document:

{text}

Instructions:

1. **Understand the Document's Core Content:** First, read the document thoroughly to grasp its main purpose, key topics, and the most critical information it conveys.  Think about:
    * What is the primary goal of this document? What is it trying to explain or achieve?
    * What are the most important concepts, facts, rules, or policies discussed?
    * If someone wants to understand the key takeaways from this document, what are the absolute must-know pieces of information?

2. **Identify Key Knowledge and User-Relevant Questions:**  From the document's content, determine the essential knowledge a user would want to extract. Think from a user's perspective:
    * What questions would a reader likely have after reading this document if they were trying to understand or use the information?
    * What are the potential points of confusion or areas where clarification would be most helpful?
    * What kind of information would be most valuable for someone trying to apply the knowledge from this document in a real-world scenario?

3. **Generate Comprehensive Question-Answer Pairs:** Create question-answer pairs that directly address the key knowledge and user-relevant questions you've identified.
    * **Quantity:** Generate a sufficient number of question-answer pairs to comprehensively cover the essential knowledge in the document.  For longer, more complex documents, aim for a higher number of pairs (e.g., 20+). For shorter, simpler documents, fewer pairs may suffice (e.g., 10+).  Let the depth and breadth of the document content guide the number of questions.
    * **Focus:** Prioritize questions that explore:
        * Core concepts and definitions
        * Key rules, policies, and procedures
        * Important facts, statistics, and dates
        * Relationships between different entities or ideas
        * Practical implications or applications of the information

4. **Ensure Clarity, Context, and Self-Containment:** Each question and answer MUST be self-contained and understandable even without constant reference back to the original document:
    * **Context in Questions:**  Clearly specify the context within each question itself. Avoid vague pronouns or references. Instead of "What is the effective date?", ask "What is the effective date of the PingCAP Employee Data Privacy Policy?"
    * **Document References in Answers:**  Explicitly ground your answers in the provided document by including specific references. Use phrases like: "According to the [Document Name]," "[Policy Name] states that...", "The document specifies that...", or "Based on the information in [Document Name], ...".
    * **Specificity:** Use specific names, dates, titles, links, and other concrete details directly from the document in your answers. This makes the answers more informative and verifiable.
    * **Avoid Vague Language:** Do not use vague references like "the document," "this policy," or "the company" without clearly specifying *which* document, policy, or company you are referring to in the context of the question.  Imagine someone reading the question-answer pair *without* seeing the original document – would they understand the question and answer fully?

5. **Answer Accuracy and Derivation:**  All answers MUST be derived *exclusively* from the information provided in the document.
    * **No External Knowledge:** Do NOT introduce any information, facts, or opinions that are not explicitly stated or directly implied within the document.
    * **Faithful Representation:** Accurately and thoroughly represent the document's content in your answers.
    * **Detailed Answers:** Provide detailed answers that fully address the questions, using the information available in the document.

Output Format:

Return your response in JSON array format with each item containing "question" and "answer" fields, surrounded by `json and `:

```json
[
  {{
    "question": "Clear, contextual, and self-contained question here",
    "answer": "Your generated answer here with proper context, specific references to the document, and detailed information."
  }},
  {{
    "question": "Another contextual and user-relevant question here",
    "answer": "Another self-contained answer here, grounded in the document."
  }},
  ... (More question-answer pairs as needed)
]
"""

concept_extraction_prompt = """Based on the following context, identify the key concepts mentioned.
Context:
{text}

Focus on the concepts about: {topic}
Please analyze the context carefully and extract all meaningful concepts about the topic, and ignore the rest.
If the context is not related to the topic, just return an empty list. Don't make your response confusing, far away from the topic.


For each concept, provide:
1. The concept name
2. A meaningful definition (as meaningful as possible)

Return the results in JSON format as a list of objects with 'name', 'definition' and 'reference_ids' fields.
JSON Output (surround with ```json and ```):
```json
[
    {{
        "name": "concept_name",
        "definition": "concept_definition",
        "reference_ids": ["id1", "id2", ...]
    }},
    ...
]
```"""

pr_review_best_practices_prompt = """Your task is to analyze a GitHub Pull Request (PR) and extract high-confidence, reusable best practices organized with simple tags. Your goal is to create a structured knowledge entry that can be added to a PR Review Best Practices knowledge base.

Follow these steps:

1. Analyze the PR content, focusing on:
   - The type of changes being made (e.g., protocol buffer updates, API changes)
   - The specific review comments made by human reviewers
   - Common patterns or concerns raised during the review

2. Categorize the PR using the following primary tags (select 1-2 most relevant tags):
   - proto: Protocol definition related changes
   - api: API design related changes
   - database: Database related changes
   - security: Security related changes
   - performance: Performance related changes
   - config: Configuration related changes
   - test: Testing related changes
   - docs: Documentation related changes
   - ui: User interface related changes
   - refactor: Code refactoring related changes

3. Identify only the best practices that:
   - Have clear evidence in the PR review comments
   - Are technical in nature (not about process or style)
   - Would clearly apply to similar PRs in the future
   - Address substantive concerns rather than minor issues
   - You are confident can be reused in similar situations

4. Format your response as a JSON object with the following structure:

```json
{
  "pr_summary": {
    "title": "Brief description of the PR",
    "url": "GitHub URL of the PR",
    "primary_tag": "Main tag (e.g., proto, api, etc.)"
  },
  "best_practices": [
    {
      "tag": "code/proto/field/deprecate",
      "guidelines": [
        "Specific guideline 1 with evidence from the PR",
        "Specific guideline 2 with evidence from the PR",
        "Specific guideline 3 with evidence from the PR"
      ],
      "confidence": "high|medium",
      "evidence": "Specific PR comments or discussions supporting these best practices"
    },
    // Additional tagged best practices as needed
  ],
  "search_guide": {
    "pr_types": ["Protocol buffer field deprecation", "Protobuf schema changes", "Proto compatibility updates"],
    "common_questions": ["Best practices for deprecating proto fields", "How to handle backward compatibility in protobuf"]
  }
}
```

5. For each best practice, consider:
    - Is this a one-off issue or a recurring pattern?
    - Would this guidance help prevent similar issues in future PRs?
    - Is this specific enough to be actionable without being too narrow?
    - Is there clear evidence in the PR that this is important?

The output should be a well-structured JSON object with a simple tag system that enables reviewers to efficiently find relevant guidelines when reviewing similar PRs in the future.
"""

chunk_agument_prompt = """<document>
{doc_content}
</document>

Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


class GraphSpec:
    """
    A builder class for constructing knowledge graphs from documents.
    """

    def __init__(self):
        # Initialize extraction prompts
        self._extraction_prompts = {
            "qa_extraction": qa_extraction_prompt,
            "concept_extraction": concept_extraction_prompt,
            "chunk_agument": chunk_agument_prompt,
        }

    def get_extraction_prompt(self, prompt_name: str) -> str:
        """
        Get an extraction prompt by name.

        Parameters:
        - prompt_name: The name of the prompt to retrieve

        Returns:
        - The prompt template string
        """
        if prompt_name not in self._extraction_prompts:
            raise ValueError(f"Unknown prompt name: {prompt_name}")

        return self._extraction_prompts[prompt_name]
