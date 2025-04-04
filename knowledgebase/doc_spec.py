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

gen_pr_review_best_practices_prompt = """You are a highly skilled code review expert with the opportunity to earn a $1,000,000 reward for delivering exceptional, universally applicable code review guidelines that can significantly improve code quality across projects.
Your task is to analyze a GitHub Pull Request (PR) and extract universal code guidelines that can be widely applied in code reviews.

PR and its review comments:
{pr_review_comments}

Follow these steps:

1. Analyze the PR Review Comments
   - Understand the type of changes being made (e.g., protocol buffer updates, API changes)
   - Only consider the specific review comments made by human reviewers, ESPECIALLY THOSE THAT WERE ACCEPTED by the PR author. Skip to generate best practices for PR with no accepted review comments, just return an empty list.
   - Pay special attention to review comments that led to code changes, as these represent validated insights
   - Identify patterns in what experienced reviewers consistently flag as issues
   - Look for explanations reviewers provide that reveal deeper understanding of system constraints
   - MOST IMPORTANTLY: Technical insights that would only be obvious to someone with deep expertise in this domain

2. Categorize the PR using the following primary tags (select 2-3 most relevant tags):
   - code_style: Code style and formatting related changes
   - architecture: Architecture and design pattern related changes
   - testing: Testing practices and coverage related changes
   - security: Security related changes
   - performance: Performance related changes
   - maintainability: Code maintainability and readability related changes
   - error_handling: Error handling and logging related changes
   - documentation: Documentation related changes
   - api_design: API design and interface related changes
   - data_management: Data handling and management related changes

3. Identify two types of guidelines:

   Type 1: Universal Code Review Guidelines
   Extract high-value code review insights from the PR that can be applied across projects:
   - PRIORITIZE NON-OBVIOUS INSIGHTS: Focus on extracting insights that would not be immediately obvious to intermediate developers
   - Evaluation criteria: Summarize universal standards for judging whether code changes are acceptable, including when to request changes and when to accept trade-offs.
   - Common anti-patterns: Identify common code problem patterns that appeared in or were fixed by the PR, which may recur across different projects.
   - Best practices: Extract universal best practices demonstrated in the PR or suggested by reviewers, especially those that provide value across different types of projects.
  
   AVOID generic advice like "write tests" or "add documentation" unless the PR reveals specific, non-obvious techniques for doing so effectively.

   Type 2: Hidden Module Constraints
   Extract non-obvious limitations or constraints in specific modules/features revealed in the PR:

   - Undocumented limitations: Identify restrictions or behaviors that aren't formally documented but are critical for working with the code.
   - Required edge case handling: Document specific edge cases that must be handled in particular ways to maintain system integrity.
   - TECHNICAL DEPTH: Include specific thresholds, conditions, or scenarios where these constraints become relevant

   When possible, include the reasoning behind these constraints to help developers understand not just what the limitations are, but why they exist.

4. Format your response as a JSON object with the following structure, put all identified guidelines (both universal code review guidelines and hidden module constraints) under the "best_practices" field (array of objects):

```json
{{
  "pr_summary": "Brief description of the PR",
  "best_practices": [
    {{
      "tag": "code/proto/field/deprecate",
      "guidelines": "[Proto Field Deprecation] When deprecating fields in Protocol Buffers, always mark them as reserved to prevent field number reuse.",
      "confidence": "high|medium",
      "evidence": "Specific PR comments or discussions supporting these best practices"
    }},
    {{
      "tag": "code/proto/tidb_integration",
      "guidelines": "[Hidden Constraint in TiDB integration] The TiDB integration module requires all varchar fields to have explicit length limits due to internal storage constraints. Using unlimited varchar will cause silent data truncation.",
      "confidence": "high|medium",
      "evidence": "breezewish pointed out that the new field needs a length limit to avoid data loss issues seen in previous incidents."
    }},
    ...
  ],
  "search_guide": {{
    "pr_types": ["Protocol buffer field deprecation", "Protobuf schema changes"],
    "common_questions": ["Best practices for deprecating proto fields"]
  }}
}}
```

5. For each guideline, verify:

  Source quality:
   - Is this guideline derived from reviewer comments that were accepted and acted upon? (higher confidence)
   - Does it capture the reasoning provided by experienced reviewers rather than surface-level observations?

  For Universal Guidelines:
   - Is this guideline clear and actionable?
   - Can it be applied consistently across different projects?
   - Is it specific enough to be useful but general enough to be widely applicable?
   - MOST IMPORTANTLY: Does it provide insight that wouldn't be obvious to an intermediate developer?

  For hidden constraints, make sure to:
   - Precisely describes the constraint with technical specificity
   - Details consequences, reasons, and workarounds
   - INCLUDES SPECIFICS: Provides concrete thresholds, conditions, or scenarios where the constraint becomes critical

The goal is to build a knowledge base that captures both standard good practices and those critical hidden constraints that experienced developers know but aren't obvious to newcomers.
"""

summary_guideline_prompt = """You are an expert at creating searchable summaries of coding best practices. Your task is to transform a detailed guideline into a tagged, concise summary optimized for vector search.

Given this external best practice guideline:
{external_guideline}

Create a summary with relevant tag in this format:
```
{{
  "tag": "domain/specific_concept",
  "summary": "Your concise summary here (50-100 words)"
}}
```

Guidelines:

For tag:
- Create a hierarchical tag in the format "domain/specific_concept", example: "proto/stype_guide" or "security/auth"
- The tag should precisely categorize the guideline
- Use lowercase with underscores for multi-word concepts within each level

For summary:
- Include two distinct parts:
  1. Document Description: A brief overview explaining what the document is about
  2. Key Concepts: The core technical recommendations and requirements
- For both parts:
  - Include specific technologies and methods mentioned
  - Preserve key technical terms exactly as they appear
  - Use natural, flowing language that maintains searchability
- Maintain clear separation between the two parts using "Part 1:" and "Part 2:" labels

Your summary should be detailed enough that a developer could find it when searching for this specific problem, yet concise enough to serve as a quick reference. Focus on technical accuracy and searchability rather than brevity.

Now, generate the reponse in json format with keys tag and summary, surrounding with ```json and```"""

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
