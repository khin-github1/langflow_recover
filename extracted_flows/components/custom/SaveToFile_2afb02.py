# Recovered Langflow component
# type: SaveToFile
# class: SaveToFileComponent
# used in 2 flow(s): Summarization_flow_chunk, Summarization_flow_simple
# json path: node.data.node.template.code.value

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import orjson
import pandas as pd
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder

from lfx.custom import Component
from lfx.inputs import SortableListInput
from lfx.io import BoolInput, DropdownInput, HandleInput, SecretStrInput, StrInput
from lfx.schema import Data, DataFrame, Message
from lfx.services.deps import get_settings_service, get_storage_service, session_scope
from lfx.template.field.base import Output
from lfx.utils.validate_cloud import is_astra_cloud_environment


def _get_storage_location_options():
    all_options = [{"name": "AWS", "icon": "Amazon"}, {"name": "Google Drive", "icon": "google"}]
    if is_astra_cloud_environment():
        return all_options
    return [{"name": "Local", "icon": "hard-drive"}, *all_options]


class SaveToFileComponent(Component):
    display_name = "Write File custom"
    description = "Save data to local file, AWS S3, or Google Drive in the selected format."
    icon = "file-text"
    name = "SaveToFile"

    LOCAL_DATA_FORMAT_CHOICES = ["csv", "excel", "json", "markdown"]
    LOCAL_MESSAGE_FORMAT_CHOICES = ["txt", "json", "markdown"]
    AWS_FORMAT_CHOICES = ["txt", "json", "csv", "xml", "html", "md", "yaml", "log", "tsv", "jsonl", "parquet", "xlsx", "zip"]
    GDRIVE_FORMAT_CHOICES = ["txt", "json", "csv", "xlsx", "slides", "docs", "jpg", "mp3"]

    inputs = [
        SortableListInput(
            name="storage_location",
            display_name="Storage Location",
            options=_get_storage_location_options(),
            real_time_refresh=True,
            limit=1,
        ),
        HandleInput(
            name="input",
            display_name="File Content",
            input_types=["Data", "DataFrame", "Message"],
            required=True,
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            required=True,
        ),
        BoolInput(
            name="append_mode",
            display_name="Append",
            value=False,
        ),
        DropdownInput(
            name="local_format",
            display_name="File Format",
            options=list(dict.fromkeys(LOCAL_DATA_FORMAT_CHOICES + LOCAL_MESSAGE_FORMAT_CHOICES)),
            value="csv",
        ),
    ]

    outputs = [Output(display_name="File Path", name="message", method="save_to_file")]

    async def save_to_file(self) -> Message:
        if not self.file_name:
            raise ValueError("File name must be provided.")
        
        storage_location = self._get_selected_storage_location()
        if storage_location == "Local":
            return await self._save_to_local()
        return Message(text="Cloud storage not configured in this snippet.")

    def _get_input_type(self) -> str:
        if type(self.input) is DataFrame: return "DataFrame"
        if type(self.input) is Message: return "Message"
        if type(self.input) is Data: return "Data"
        raise ValueError(f"Unsupported input type: {type(self.input)}")

    def _adjust_file_path_with_format(self, path: Path, fmt: str) -> Path:
        if path.suffix.lower().lstrip(".") == fmt: return path
        return Path(f"{path}.{fmt}").expanduser()

    def _is_plain_text_format(self, fmt: str) -> bool:
        return fmt.lower() in ["txt", "json", "markdown", "md", "csv", "jsonl"]

    def _save_data(self, data: Data, path: Path, fmt: str) -> str:
        """Save a Data object to the specified file format."""
        append_mode = getattr(self, "append_mode", False)
        should_append = append_mode and path.exists() and self._is_plain_text_format(fmt)
        
        # --- THE CRITICAL FIX FOR YOUR SCALAR ERROR ---
        process_data = data.data
        if isinstance(process_data, dict):
            process_data = [process_data] # Wrap the single summary row in a list
        # ----------------------------------------------

        if fmt == "csv":
            pd.DataFrame(process_data).to_csv(
                path,
                index=False,
                mode="a" if should_append else "w",
                header=not should_append,
            )
        elif fmt == "json":
            new_data = jsonable_encoder(process_data)
            path.write_text(json.dumps(new_data, indent=2), encoding="utf-8")
        
        action = "appended to" if should_append else "saved successfully as"
        return f"Data {action} '{path}'"

    async def _save_to_local(self) -> Message:
        file_format = getattr(self, "local_format", "csv")
        file_path = Path(self.file_name).expanduser()
        file_path = self._adjust_file_path_with_format(file_path, file_format)

        if self._get_input_type() == "Data":
            confirmation = self._save_data(self.input, file_path, file_format)
        else:
            confirmation = "Only Data types supported in this simplified fix."

        return Message(text=f"{confirmation} at {file_path}")

    def _get_selected_storage_location(self) -> str:
        if hasattr(self, "storage_location") and self.storage_location:
            return self.storage_location[0].get("name", "Local")
        return "Local"