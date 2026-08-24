# Recovered Langflow component
# type: BatchRunComponent
# class: BatchRunComponent
# used in 3 flow(s): Question and Answer generation v0, Question and Answer generation v1, Question and Answer generation v2
# json path: node.data.node.template.code.value

from __future__ import annotations
import re
from typing import TYPE_CHECKING, Any, cast

import toml  # type: ignore[import-untyped]

from lfx.custom.custom_component.component import Component
from lfx.io import BoolInput, DataFrameInput, HandleInput, MessageTextInput, MultilineInput, Output
from lfx.log.logger import logger
from lfx.schema.dataframe import DataFrame

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

class BatchRunComponent(Component):
    display_name = "Batch Run (Split QA)"
    description = "Runs LLM and splits multiple Q&A pairs into individual rows."
    documentation: str = "https://docs.langflow.org/batch-run"
    icon = "List"

    inputs = [
        HandleInput(
            name="model",
            display_name="Language Model",
            input_types=["LanguageModel"],
            required=True,
        ),
        MultilineInput(
            name="system_message",
            display_name="Instructions",
            info="Instructions for generating the 15 Q&A pairs.",
            required=False,
        ),
        DataFrameInput(
            name="df",
            display_name="DataFrame",
            required=True,
        ),
        MessageTextInput(
            name="column_name",
            display_name="Input Column Name",
            info="The column containing the text to process.",
            required=False,
        ),
    ]

    outputs = [
        Output(
            display_name="LLM Results",
            name="batch_results",
            method="run_batch",
            info="Returns a DataFrame with columns: source, text, question, answer, batch_index",
        ),
    ]

    def _parse_qa_pairs(self, text: str) -> list[dict[str, str]]:
        """
        Parses the LLM response into a list of {'question': ..., 'answer': ...}
        This regex looks for Q: and A: patterns or numbered lists.
        """
        pairs = []
        # Split by the start of a question (e.g., "1. Q:" or "Q:")
        raw_sections = re.split(r'\d+\.\s+\*\*Q:\*\*', text)
        if len(raw_sections) <= 1:
            raw_sections = re.split(r'\*\*Q:\*\*', text)

        for section in raw_sections:
            if not section.strip():
                continue
            
            # Split the section into question and answer parts
            parts = re.split(r'\*\*A:\*\*', section)
            if len(parts) >= 2:
                q = parts[0].strip(" *:")
                # Take everything until the next potential question or end of string
                a = parts[1].split('**Q:**')[0].strip(" *:")
                pairs.append({"question": q, "answer": a})
        
        return pairs

    async def run_batch(self) -> DataFrame:
        model: Runnable = self.model
        system_msg = self.system_message or ""
        df: DataFrame = self.df
        col_name = self.column_name or "text"

        if not isinstance(df, DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")

        user_texts = df[col_name].astype(str).tolist()
        original_records = df.to_dict(orient="records")

        conversations = [
            [{"role": "system", "content": system_msg}, {"role": "user", "content": text}]
            if system_msg else [{"role": "user", "content": text}]
            for text in user_texts
        ]

        # Run the batch
        raw_responses = await model.abatch(list(conversations))
        
        final_rows = []
        for idx, (original_row, response) in enumerate(zip(original_records, raw_responses)):
            # Extract the raw text from the model
            full_text = response.content if hasattr(response, "content") else str(response)
            
            # Parse the text into individual Q&A dicts
            qa_list = self._parse_qa_pairs(full_text)
            
            # If parsing fails or finds nothing, keep one row with empty Q&A
            if not qa_list:
                qa_list = [{"question": "N/A", "answer": "Could not parse response"}]

            for qa in qa_list:
                new_row = {
                    "source": original_row.get("source", ""),
                    "text": original_row.get(col_name, ""),
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "batch_index": idx
                }
                final_rows.append(new_row)

        await logger.ainfo(f"Generated {len(final_rows)} Q&A rows from {len(original_records)} chunks.")
        return DataFrame(final_rows)