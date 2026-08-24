# Recovered Langflow component
# type: SufficiencyScorer
# class: SufficiencyScorer
# used in 1 flow(s): CRCV V4
# json path: node.data.node.template.code.value

from typing import List, Dict
from langflow import CustomComponent
from langchain_ollama import OllamaEmbeddings  # Assuming you have this; else import from your setup
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SufficiencyScorer(CustomComponent):
    display_name = "Sufficiency Scorer"
    description = "Computes embedding similarity-based sufficiency score for CRCV."

    def build_config(self):
        return {
            "refined_query": {"display_name": "Refined Query", "type": "str"},
            "threshold": {"display_name": "Similarity Threshold", "type": "float", "value": 0.7},
            "ideal_templates": {
                "display_name": "Ideal Templates (JSON list)",
                "type": "str",
                "value": '["What is the deadline for {program} {degree} in {intake}?", "Eligibility for {scholarship} in {program}?", ...]'  # Add your domain templates
            }
        }

    def build(self, refined_query: str, threshold: float = 0.7, ideal_templates: str = None) -> Dict:
        # Parse templates (JSON string to list)
        templates = eval(ideal_templates) if ideal_templates else []  # Secure this in prod

        if not templates:
            raise ValueError("Provide ideal templates for sufficiency check.")

        # Embedder (use your Ollama instance)
        embedder = OllamaEmbeddings(model="qwen3:4b")  # Match your graph's model

        # Embed query and templates
        query_emb = np.array(embedder.embed_query(refined_query)).reshape(1, -1)
        template_embs = np.array([embedder.embed_query(t) for t in templates])

        # Compute max similarity (best match to any ideal)
        similarities = cosine_similarity(query_emb, template_embs)[0]
        max_sim = np.max(similarities)
        score = max_sim  # C = max similarity

        # Decision
        is_sufficient = score >= threshold

        # Output dict for Langflow (log score for thesis metrics)
        return {"sufficiency_score": score, "decision": is_sufficient}