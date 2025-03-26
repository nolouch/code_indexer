"""
from typing import List, Dict, Any
import openai
from ..retrieval.semantic import SemanticRetriever

class LLMRefiner:
    """LLM-based search result refinement and explanation"""
    
    def __init__(self, api_key: str):
        openai.api_key = api_key
        
    def refine_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank and filter search results using LLM"""
        if not results:
            return []
            
        # Prepare context for LLM
        context = self._prepare_context(results)
        
        # Generate prompt
        prompt = f"""Given the search query: "{query}"
        
And these code search results:
{context}

Rerank these results based on their relevance to the query.
For each result, provide a relevance score between 0 and 1, and a brief explanation.
Format your response as a JSON list with fields: node, score, explanation"""
        
        try:
            # Get LLM response
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a code search assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse response
            refined = eval(response.choices[0].message.content)
            
            # Update results with LLM scores and explanations
            scored_results = []
            for item in refined:
                for result in results:
                    if result["node"] == item["node"]:
                        result.update({
                            "llm_score": item["score"],
                            "explanation": item["explanation"]
                        })
                        scored_results.append(result)
                        break
                        
            # Sort by LLM score
            scored_results.sort(key=lambda x: x["llm_score"], reverse=True)
            return scored_results
            
        except Exception as e:
            print(f"Error in LLM refinement: {e}")
            return results
            
    def explain_code(self, code: str, context: str = "") -> str:
        """Generate natural language explanation of code"""
        prompt = f"""Explain this code:
{code}

Context:
{context}

Provide a clear and concise explanation focusing on:
1. Purpose and functionality
2. Key components and their roles
3. Important relationships and dependencies"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a code explanation assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in code explanation: {e}")
            return "Could not generate explanation."
            
    def synthesize_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Synthesize a comprehensive answer from search results"""
        if not results:
            return "No relevant results found."
            
        context = self._prepare_context(results)
        prompt = f"""Given the query: "{query}"
        
And these code search results:
{context}

Synthesize a comprehensive answer that:
1. Directly addresses the query
2. References relevant code elements
3. Explains key relationships
4. Provides usage examples if applicable"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a code search assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in answer synthesis: {e}")
            return "Could not synthesize answer."
            
    def _prepare_context(self, results: List[Dict[str, Any]]) -> str:
        """Prepare search results for LLM prompt"""
        context = []
        for result in results:
            item = f"Node: {result['node']}\n"
            item += f"Type: {result['type']}\n"
            item += f"File: {result['file']}:{result['line']}\n"
            if "docstring" in result:
                item += f"Documentation: {result['docstring']}\n"
            if "score" in result:
                item += f"Search Score: {result['score']}\n"
            context.append(item)
        return "\n".join(context)
""" 