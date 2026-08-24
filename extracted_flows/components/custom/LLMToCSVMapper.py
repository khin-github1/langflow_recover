# Recovered Langflow component
# type: LLMToCSVMapper
# class: LLMToCSVMapper
# used in 2 flow(s): Summarization_flow_chunk, Summarization_flow_simple
# json path: node.data.node.template.code.value

import json
from langflow.custom import CustomComponent
from langflow.schema import Data

class LLMToCSVMapper(CustomComponent):
    display_name = "LLM to CSV Mapper"
    description = "Parses Ollama JSON and maps it to CSV columns (URL, Extractive, Abstractive)."

    def build_config(self):
        return {
            "llm_output": {"display_name": "LLM Output", "info": "Connect the 'Text' output from Ollama here."},
            "source_url": {"display_name": "Source URL", "info": "Connect the 'source' variable or the scraper data here."},
        }

    def build(self, llm_output: str, source_url: str) -> Data:
        # 1. Clean and Parse JSON from LLM
        try:
            # Handle cases where LLM adds markdown backticks ```json ... ```
            clean_json = llm_output.replace("```json", "").replace("```", "").strip()
            data_dict = json.loads(clean_json)
        except Exception as e:
            # Fallback if JSON parsing fails
            data_dict = {
                "extractive": f"Error parsing: {str(e)}",
                "abstractive": llm_output
            }

        # 2. Map to final flat structure for CSV
        # We ensure the keys match the column names you want in your CSV
        result = {
            "url": source_url,
            "extractive_summary": data_dict.get("extractive", "N/A"),
            "abstractive_summary": data_dict.get("abstractive", "N/A")
        }

        return Data(data=result)