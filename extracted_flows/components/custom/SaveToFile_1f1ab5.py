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
    display_name = "Write File"
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
            limit=1,
            real_time_refresh=True,
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
        storage_location = self._get_selected_storage_location()
        if storage_location == "Local":
            return await self._save_to_local()
        # Add AWS/Gdrive routing here if needed
        return Message(text="Cloud storage not implemented in this snippet.")

    def _get_input_type(self) -> str:
        if isinstance(self.input, DataFrame): return "DataFrame"
        if isinstance(self.input, Message): return "Message"
        if isinstance(self.input, Data): return "Data"
        return "Unknown"

    async def _save_to_local(self) -> Message:
        file_format = self.local_format
        file_name_with_ext = f"{self.file_name}.{file_format}"
        file_path = Path(file_name_with_ext)

        # 1. Physical Save to Docker Container
        if self._get_input_type() == "Data":
            data_to_save = self.input.data
            # If data is a dict inside a list (common for CSV), ensure it's handled
            df = pd.DataFrame(data_to_save if isinstance(data_to_save, list) else [data_to_save])
            df.to_csv(file_path, index=False, mode='a' if self.append_mode else 'w', header=not (self.append_mode and file_path.exists()))
        
        # 2. THE FIX: Correct Storage Sync for UI Visibility
        try:
            storage_service = get_storage_service()
            with open(file_path, 'rb') as f:
                # FIX: use 'data=' instead of 'content='
                storage_service.save_file(
                    flow_id=self.graph.flow_id, 
                    file_name=file_name_with_ext, 
                    data=f.read() 
                )
            return Message(text=f"Success! File '{file_name_with_ext}' is ready in 'My Files'.")
        except Exception as e:
            return Message(text=f"Saved to container, but UI sync failed: {str(e)}")

    def _get_selected_storage_location(self) -> str:
        if hasattr(self, "storage_location") and self.storage_location:
            return self.storage_location[0].get("name", "Local")
        return "Local"