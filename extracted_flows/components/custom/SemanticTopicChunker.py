# Recovered Langflow component
# type: SemanticTopicChunker
# class: SemanticTopicChunker
# used in 1 flow(s): Question and Answer generation v1
# json path: node.data.node.template.code.value

from __future__ import annotations
import json
import re
from typing import Any
from lfx.custom.custom_component.component import Component
from lfx.io import MultilineInput, Output, HandleInput, MessageTextInput
from lfx.schema.dataframe import DataFrame

class SemanticTopicChunker(Component):
    display_name = "Semantic Topic Chunker"
    description = "Chunks text based on topic coherence and logical structure using an LLM."
    icon = "Divide"

    inputs = [
        HandleInput(
            name="model",
            display_name="Language Model",
            input_types=["LanguageModel"],
            required=True,
            info="Model used to determine topic boundaries (e.g., GPT-4o-mini)."
        ),
        MultilineInput(
            name="text",
            display_name="Input Text",
            required=True,
        ),
        MessageTextInput(
            name="source_name",
            display_name="Source Name/URL",
            value="manual_upload",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Semantic Chunks",
            name="chunks",
            method="segment_text",
        ),
    ]

    async def segment_text(self) -> DataFrame:
        raw_text = self.text
        model = self.model
        
        # 1. Simple sentence splitting (can be improved with NLTK/SpaCy)
        sentences = re.split(r'(?<=[.!?]) +', raw_text.replace('\n', ' '))
        
        # 2. Topic Segmentation Prompt
        # We ask the LLM to look at the list of sentences and mark indices where topics change.
        segmentation_prompt = f"""
        Analyze the following sentences from a document. 
        Your task is to identify logical boundaries where the topic changes significantly.
        
        Sentences:
        {chr(10).join([f"{i}: {s}" for i, s in enumerate(sentences)])}
        
        Return a JSON object with a list of indices where a NEW topic starts. 
        Example: {{"topic_start_indices": [0, 5, 12]}}
        Only return the JSON.
        """

        # Call LLM (using the Langchain-compatible model handle)
        response = await model.ainvoke(segmentation_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 3. Parse indices
        try:
            # Clean JSON formatting if model adds backticks
            clean_json = re.search(r'\{.*\}', content, re.DOTALL).group()
            data = json.loads(clean_json)
            start_indices = data.get("topic_start_indices", [0])
        except Exception as e:
            # Fallback if LLM fails: treat as one big chunk
            start_indices = [0]

        # 4. Group sentences into chunks
        chunks_data = []
        for i in range(len(start_indices)):
            start = start_indices[i]
            end = start_indices[i+1] if i+1 < len(start_indices) else len(sentences)
            
            chunk_text = " ".join(sentences[start:end])
            
            chunks_data.append({
                "source": self.source_name,
                "text": chunk_text,
                "chunk_index": i,
                "sentence_range": f"{start}-{end}"
            })

        return DataFrame(chunks_data)