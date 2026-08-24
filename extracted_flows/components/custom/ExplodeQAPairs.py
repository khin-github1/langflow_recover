# Recovered Langflow component
# type: ExplodeQAPairs
# class: ExplodeQAPairs
# used in 2 flow(s): Question and Answer generation v0, Question and Answer generation v1
# json path: node.data.node.template.code.value

import re
from lfx.base.models.model import LCModelComponent
from lfx.io import DataFrameInput, Output
from lfx.schema.dataframe import DataFrame


class ExplodeQAPairs(LCModelComponent):

    display_name = "Explode QA Pairs"
    description = "Split model_response into one row per question-answer pair."

    inputs = [
        DataFrameInput(name="df", display_name="Input DataFrame")
    ]

    outputs = [
        Output(
            display_name="Exploded DataFrame",
            name="dataframe_output",
            method="build_output"
        )
    ]

    def build_output(self) -> DataFrame:
        df = self.df  # ✅ Correct way in your version

        rows = []

        for _, row in df.iterrows():
            source = row.get("source")
            text = row.get("text")
            batch_index = row.get("batch_index")
            response = str(row.get("model_response", ""))

            pattern = r"\d+\.\s*\*\*(.*?)\*\*\s*\*\*Answer:\*\*\s*(.*?)(?=\d+\.\s*\*\*|$)"
            matches = re.findall(pattern, response, re.DOTALL)

            for q, a in matches:
                rows.append({
                    "source": source,
                    "text": text,
                    "batch_index": batch_index,
                    "question": q.strip(),
                    "answer": a.strip()
                })

        return DataFrame(rows)