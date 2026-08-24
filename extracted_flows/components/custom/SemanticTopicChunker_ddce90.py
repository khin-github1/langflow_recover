# Recovered Langflow component
# type: SemanticTopicChunker
# class: SemanticTopicChunker
# used in 1 flow(s): Question and Answer generation v2
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
    description = "Chunks text based on topic coherence. Automatically parses scraper JSON for source URL."
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
            name="input_data",
            display_name="Scraper JSON / Text",
            info="Paste the JSON output from the scraper here.",
            required=True,
        ),
        MessageTextInput(
            name="default_source",
            display_name="Default Source Name",
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
        raw_input = self.input_data
        model = self.model
        
        # 1. Extract Text and Source from JSON
        source_url = self.default_source
        processing_text = raw_input

        try:
            # Clean possible markdown code blocks (```json ... ```) from the input string
            clean_input = re.sub(r'^```json\s*|```$', '', raw_input.strip(), flags=re.MULTILINE)
            data_json = json.loads(clean_input)
            
            processing_text = data_json.get("text", raw_input)
            source_url = data_json.get("source", self.default_source)
        except (json.JSONDecodeError, AttributeError):
            # Fallback if input is plain text and not JSON
            pass

        # 2. Sentence splitting
        sentences = re.split(r'(?<=[.!?]) +', processing_text.replace('\n', ' '))
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 3. Topic Segmentation Prompt
        segmentation_prompt = f"""
        Analyze the following sentences. Identify logical boundaries where the topic changes.
        
        Sentences:
        {chr(10).join([f"{i}: {s}" for i, s in enumerate(sentences)])}
        
        Return a JSON object with a list of indices where a NEW topic starts. 
        Example: {{"topic_start_indices": [0, 5, 12]}}
        Only return the JSON.
        """

        # Call LLM
        response = await model.ainvoke(segmentation_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 4. Parse indices
        try:
            clean_json = re.search(r'\{.*\}', content, re.DOTALL).group()
            data = json.loads(clean_json)
            start_indices = data.get("topic_start_indices", [0])
        except Exception:
            start_indices = [0]

        # 5. Group sentences into chunks using the dynamic source_url
        chunks_data = []
        for i in range(len(start_indices)):
            start = start_indices[i]
            end = start_indices[i+1] if i+1 < len(start_indices) else len(sentences)
            
            chunk_text = " ".join(sentences[start:end])
            
            chunks_data.append({
                "source": source_url,
                "text": chunk_text,
                "chunk_index": i,
                "sentence_range": f"{start}-{end}"
            })

        return DataFrame(chunks_data)