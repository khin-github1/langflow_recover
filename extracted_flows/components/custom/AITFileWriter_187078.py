# Recovered Langflow component
# type: AITFileWriter
# class: AITFileWriter
# used in 3 flow(s): Summarization_flow_chunk, Summarization_flow_simple, URL Test
# json path: node.data.node.template.code.value

import pandas as pd
from pathlib import Path
from fastapi import UploadFile
from langflow.custom import Component
from langflow.io import HandleInput, StrInput, BoolInput, Output
from langflow.schema import Data, Message
from lfx.services.deps import get_settings_service, get_storage_service, session_scope

class AITFileWriter(Component):
    display_name = "AIT File Writer"
    description = "Saves AIT data and FORCES it to appear in the Langflow Files list."
    icon = "file-text"

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Data to Save",
            input_types=["Data"],
            required=True,
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            value="ait_summary_results",
            required=True,
        ),
        BoolInput(
            name="append_mode",
            display_name="Append Mode",
            value=True,
        ),
    ]

    outputs = [Output(display_name="Status", name="status", method="write_and_sync")]

    async def write_and_sync(self) -> Message:
        # 1. Setup File Path
        file_name_with_ext = f"{self.file_name}.csv"
        file_path = Path(file_name_with_ext)
        
        # 2. Extract and Save Data
        # FIXED: Loop through the list to extract '.data' from each Data object
        raw_rows = []
        if isinstance(self.input_data, list):
            for item in self.input_data:
                # If it's a Data object, get its .data dict
                if hasattr(item, "data"):
                    raw_rows.append(item.data)
                else:
                    raw_rows.append(item)
        else:
            # Handle single Data object
            raw_rows = [self.input_data.data] if hasattr(self.input_data, "data") else [self.input_data]

        df = pd.DataFrame(raw_rows)
        
        # Ensure correct column order for AIT project
        # This matches your requested url | Summary with extractive | Summary with abstractive
        column_mapping = {
            "url": "url",
            "Summary with extractive": "Summary with extractive",
            "summary with abstractive": "Summary with abstractive"
        }
        df = df.rename(columns=column_mapping)
        
        write_header = not (self.append_mode and file_path.exists())
        mode = 'a' if self.append_mode else 'w'
        df.to_csv(file_path, mode=mode, index=False, header=write_header, encoding='utf-8')

        # 3. THE FIX: Upload to User Storage (Same as your existing code)
        try:
            from langflow.api.v2.files import upload_user_file
            from langflow.services.database.models.user.crud import get_user_by_id

            with file_path.open("rb") as f:
                async with session_scope() as db:
                    if not self.user_id:
                        raise ValueError("User ID is required. Please ensure you are logged in.")
                    
                    current_user = await get_user_by_id(db, self.user_id)

                    await upload_user_file(
                        file=UploadFile(filename=file_name_with_ext, file=f, size=file_path.stat().st_size),
                        session=db,
                        current_user=current_user,
                        storage_service=get_storage_service(),
                        settings_service=get_settings_service(),
                        append=False,
                    )
            
            return Message(text=f"✅ Success! Saved {len(df)} rows. Find '{file_name_with_ext}' in 'My Files'.")
        except Exception as e:
            return Message(text=f"Saved locally, but UI sync failed: {str(e)}")