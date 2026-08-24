# Recovered Langflow component
# type: save_dataframe_as_csv
# class: SaveDataFrameAsCSV
# used in 1 flow(s): CRCV Evaluation Runner 0.0.01
# json path: node.data.node.template.code.value

from __future__ import annotations

import csv
import os
import tempfile
from typing import Any, Dict, List, Optional

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, HandleInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


class SaveDataFrameAsCSV(Component):
    display_name = "Save CSV (Download)"
    description = "Writes an input DataFrame to /mnt/data and outputs a Data(file_path) for download."
    icon = "download"
    name = "save_dataframe_as_csv"

    inputs = [
        HandleInput(
            name="df",
            display_name="Input DataFrame",
            input_types=["DataFrame", "Data"],
            required=True,
        ),
        MessageTextInput(
            name="filename",
            display_name="Filename",
            value="output.csv",
            required=True,
        ),
        BoolInput(
            name="include_all_columns",
            display_name="Include all columns",
            value=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="CSV File (Data)", name="file", method="save"),
    ]

    def save(self) -> Data:
        df = self.df
        filename = (self.filename or "output.csv").strip() or "output.csv"
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        out_dir = "/mnt/data" if os.path.isdir("/mnt/data") else None
        if out_dir:
            path = os.path.join(out_dir, filename)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            path = tmp.name
            tmp.close()

        # Convert DataFrame -> list[dict]
        rows: List[Dict[str, Any]] = []
        if isinstance(df, DataFrame):
            for _, row in df.iterrows():
                rows.append(row.to_dict())
        elif isinstance(df, Data) and isinstance(df.data, dict):
            # If someone passes one-row Data
            rows = [dict(df.data)]
        else:
            rows = []

        if not rows:
            # still create a file with just headers none
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write("")
            return Data(data={"file_path": path, "file_name": filename, "mime_type": "text/csv"})

        # Determine columns
        keys = set()
        for r in rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

        return Data(
            data={
                "file_path": path,
                "file_name": filename,
                "mime_type": "text/csv",
            }
        )
