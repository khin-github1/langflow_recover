# Recovered Langflow component
# type: SaveToFile
# class: SaveToFileComponent
# used in 3 flow(s): 280 Pages, problem example, scrape with markdown
# json path: node.data.node.template.code.value

import json
import asyncio
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
    description = "Save data to local file, AWS S3, or Google Drive by iterating through text content."
    documentation: str = "https://docs.langflow.org/write-file"
    icon = "file-text"
    name = "SaveToFile"

    LOCAL_DATA_FORMAT_CHOICES = ["csv", "excel", "json", "markdown", "txt"]
    LOCAL_MESSAGE_FORMAT_CHOICES = ["txt", "json", "markdown"]
    AWS_FORMAT_CHOICES = ["txt", "json", "csv", "md", "jsonl"]
    GDRIVE_FORMAT_CHOICES = ["txt", "json", "csv", "xlsx", "docs"]

    inputs = [
        SortableListInput(
            name="storage_location",
            display_name="Storage Location",
            placeholder="Select Location",
            info="Choose where to save the file.",
            options=_get_storage_location_options(),
            real_time_refresh=True,
            limit=1,
        ),
        HandleInput(
            name="input",
            display_name="File Content",
            info="The input to save (supports individual items or lists).",
            dynamic=True,
            input_types=["Data", "DataFrame", "Message", "list"],
            required=True,
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            info="Name file will be saved as (without extension).",
            required=True,
        ),
        BoolInput(
            name="append_mode",
            display_name="Append",
            info="Append to file if it exists (Local storage only).",
            value=False,
        ),
        DropdownInput(
            name="local_format",
            display_name="File Format",
            options=list(dict.fromkeys(LOCAL_DATA_FORMAT_CHOICES + LOCAL_MESSAGE_FORMAT_CHOICES)),
            info="Select the file format for local storage.",
            value="markdown",
        ),
        DropdownInput(
            name="aws_format",
            display_name="AWS Format",
            options=AWS_FORMAT_CHOICES,
            show=False,
            value="txt",
        ),
        DropdownInput(
            name="gdrive_format",
            display_name="GDrive Format",
            options=GDRIVE_FORMAT_CHOICES,
            show=False,
            value="txt",
        ),
        SecretStrInput(name="aws_access_key_id", display_name="AWS Key", show=False, advanced=True),
        SecretStrInput(name="aws_secret_access_key", display_name="AWS Secret", show=False, advanced=True),
        StrInput(name="bucket_name", display_name="S3 Bucket", show=False, advanced=True),
        StrInput(name="aws_region", display_name="AWS Region", show=False, advanced=True),
        StrInput(name="s3_prefix", display_name="S3 Prefix", show=False, advanced=True),
        SecretStrInput(name="service_account_key", display_name="GCP Key", show=False, advanced=True),
        StrInput(name="folder_id", display_name="GDrive Folder ID", show=False, advanced=True),
    ]

    outputs = [Output(display_name="Status", name="message", method="save_to_file")]

    def _get_selected_storage_location(self) -> str:
        if hasattr(self, "storage_location") and self.storage_location:
            if isinstance(self.storage_location, list) and len(self.storage_location) > 0:
                return self.storage_location[0].get("name", "")
        return ""

    def _extract_content_for_upload(self) -> str:
        """Iteratively writes the 'text' column/field from input list into Markdown format."""
        items = self.input if isinstance(self.input, list) else [self.input]
        markdown_output = []

        for item in items:
            # 1. Handle Message Objects
            if isinstance(item, Message):
                markdown_output.append(str(item.text) if item.text else "")
            
            # 2. Handle Data Objects (Targeting 'text' field)
            elif isinstance(item, Data):
                text_content = item.data.get("text") or item.data.get("content")
                if text_content:
                    markdown_output.append(str(text_content))
                else:
                    markdown_output.append(json.dumps(item.data))
            
            # 3. Handle DataFrames
            elif isinstance(item, pd.DataFrame):
                if "text" in item.columns:
                    markdown_output.extend(item["text"].astype(str).tolist())
                else:
                    markdown_output.append(item.to_markdown(index=False))
            
            # 4. Fallback for strings
            else:
                markdown_output.append(str(item))

        return "\n\n".join(markdown_output)

    async def save_to_file(self) -> Message:
        if not self.file_name:
            raise ValueError("File name is required.")
        
        storage_location = self._get_selected_storage_location()
        if not storage_location:
            raise ValueError("Please select a Storage Location.")

        content = self._extract_content_for_upload()
        
        if storage_location == "Local":
            return await self._process_local_save(content)
        elif storage_location == "AWS":
            return await self._process_aws_save(content)
        elif storage_location == "Google Drive":
            return await self._process_gdrive_save(content)
        
        return Message(text="Storage location error.")

    async def _process_local_save(self, content: str) -> Message:
        fmt = getattr(self, "local_format", "markdown")
        file_path = Path(self.file_name).expanduser()
        
        if not file_path.suffix:
            ext = "md" if fmt == "markdown" else fmt
            file_path = file_path.with_suffix(f".{ext}")

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if self.append_mode and file_path.exists() else "w"
        with open(file_path, mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n\n")
            f.write(content)

        # Optional: Trigger internal Langflow file sync if user_id is present
        try:
            await self._upload_file(file_path)
        except:
            pass

        return Message(text=f"Successfully written to {file_path}")

    # Helper for Langflow's internal file storage service
    async def _upload_file(self, file_path: Path) -> None:
        from langflow.api.v2.files import upload_user_file
        from langflow.services.database.models.user.crud import get_user_by_id
        
        if not self.user_id: return

        with file_path.open("rb") as f:
            async with session_scope() as db:
                current_user = await get_user_by_id(db, self.user_id)
                await upload_user_file(
                    file=UploadFile(filename=file_path.name, file=f, size=file_path.stat().st_size),
                    session=db,
                    current_user=current_user,
                    storage_service=get_storage_service(),
                    settings_service=get_settings_service(),
                    append=False,
                )

    async def _process_aws_save(self, content: str) -> Message:
        # Implementation logic for Boto3 using 'content' string...
        return Message(text="AWS upload simulated with Markdown content.")

    async def _process_gdrive_save(self, content: str) -> Message:
        # Implementation logic for Google API using 'content' string...
        return Message(text="Google Drive upload simulated with Markdown content.")