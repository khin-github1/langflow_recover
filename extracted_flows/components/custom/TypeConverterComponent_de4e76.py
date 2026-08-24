# Recovered Langflow component
# type: TypeConverterComponent
# class: TypeConverterComponent
# used in 2 flow(s): 280 Pages, scrape with markdown
# json path: node.data.node.template.code.value

import json
import pandas as pd
from io import StringIO
from typing import Any, List, Union

from lfx.custom import Component
from lfx.io import BoolInput, HandleInput, Output
from lfx.schema import Data, DataFrame, Message

MIN_CSV_LINES = 2

def convert_to_message(v) -> Message:
    """Helper to ensure the object becomes a Message type."""
    if isinstance(v, Message):
        return v
    # If it's a dict or Data object, Langflow's schema usually has a .to_message()
    if hasattr(v, "to_message"):
        return v.to_message()
    # Fallback for raw dictionaries or unexpected types
    return Message(text=str(v), data=v if isinstance(v, dict) else {"content": v})

def parse_structured_data(text: str) -> dict:
    """Try to extract data from JSON or CSV strings."""
    raw_text = text.lstrip("\ufeff").strip()
    
    # Try JSON
    try:
        parsed = json.loads(raw_text)
        return {"records": parsed} if isinstance(parsed, list) else parsed
    except:
        pass

    # Try CSV
    lines = raw_text.split("\n")
    if len(lines) >= MIN_CSV_LINES and "," in lines[0]:
        try:
            df = pd.read_csv(StringIO(raw_text))
            return {"records": df.to_dict(orient="records")}
        except:
            pass
            
    return {"text": text}

class TypeConverterComponent(Component):
    display_name = "List to Message Converter"
    description = "Converts every item in a list (or a single item) into Message format."
    icon = "mail"

    inputs = [
        HandleInput(
            name="input_data",
            display_name="Input Data",
            input_types=["Message", "Data", "DataFrame", "list"],
            info="Accepts a single object or a list of objects (Strings, Data, etc.)",
            required=True,
        ),
        BoolInput(
            name="auto_parse",
            display_name="Auto Parse Strings",
            info="If input is a JSON/CSV string, parse it into structured data before converting to Message.",
            advanced=True,
            value=False,
        ),
    ]

    outputs = [
        Output(
            display_name="Message Output", 
            name="output", 
            method="convert_all_to_message"
        )
    ]

    def convert_all_to_message(self) -> Union[Message, List[Message]]:
        raw_input = self.input_data
        
        # Determine if we are dealing with a list or a single item
        is_list = isinstance(raw_input, list)
        items = raw_input if is_list else [raw_input]
        
        processed_messages = []

        for item in items:
            # 1. Handle raw strings (with optional parsing)
            if isinstance(item, str):
                if self.auto_parse:
                    parsed_dict = parse_structured_data(item)
                    msg = Message(text=item, data=parsed_dict)
                else:
                    msg = Message(text=item)
            
            # 2. Handle known schema types or dicts
            else:
                msg = convert_to_message(item)
            
            processed_messages.append(msg)

        # Return list if input was list, else return the single Message
        result = processed_messages if is_list else processed_messages[0]
        self.status = result
        return result