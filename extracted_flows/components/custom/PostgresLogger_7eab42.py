# Recovered Langflow component
# type: PostgresLogger
# class: PostgresLogger
# used in 3 flow(s): CRCV-v1.0, CRCV-v1.1, CRCV-v1.1.01
# json path: node.data.node.template.code.value

from datetime import datetime

import psycopg2

from langflow.custom import Component
from langflow.io import (
    MessageTextInput,
    StrInput,
    IntInput,
    Output,
)
from langflow.schema.message import Message


class PostgresLogger(Component):
    """
    Log LLM answers into the public.crcv_logs table in Postgres.

    - Input: LLM answer text (from a Message)
    - Side effect: INSERT row into crcv_logs
    - Output: same text wrapped back into a Message (so you can inline it in the chain)
    """

    display_name = "Postgres Logger"
    description = "Logs LLM replies to Postgres (crcv_logs)."
    icon = "database"
    name = "PostgresLogger"

    # ---- Inputs ----
    inputs = [
        # Take the text of a Message (MessageTextInput gives you .text as a plain str)
        MessageTextInput(
            name="answer_text",
            display_name="Answer Text",
            info="LLM answer to log into crcv_logs",
        ),
        StrInput(
            name="db_url",
            display_name="DB URL",
            value="postgresql://user:password@pgvector:5432/default",
            info="Postgres connection string",
        ),
        StrInput(
            name="session_id",
            display_name="Session ID",
            value="default_session",
            advanced=True,
        ),
        IntInput(
            name="turn_index",
            display_name="Turn Index",
            value=0,
            advanced=True,
        ),
        StrInput(
            name="llm_name",
            display_name="LLM Name",
            value="unknown_model",
            advanced=True,
        ),
    ]

    # ---- Outputs ----
    outputs = [
        Output(
            name="logged_message",
            display_name="Logged Message",
            method="log_and_forward",
            info="Same text as input, after attempting to log.",
        )
    ]

    # ---- Logic ----
    def log_and_forward(self) -> Message:
        text = self.answer_text or ""
        db_url = self.db_url or ""
        session_id = self.session_id or "default_session"
        llm_name = self.llm_name or "unknown_model"
        turn_index = int(self.turn_index or 0)

        # Default status – will show in the node UI
        self.status = f"Attempting to log {len(text)} chars"

        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO crcv_logs
                    (session_id, turn_index, llm_name, answer_text, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, turn_index, llm_name, text, datetime.utcnow()),
            )
            conn.commit()
            cur.close()
            conn.close()

            self.status = f"Logged to crcv_logs (len={len(text)})"
        except Exception as e:
            # Don’t kill the flow, just record the error
            self.status = f"Postgres logging error: {e}"

        # Pass the answer on as a Message so you can still feed it to ChatOutput
        return Message(text=text, sender="assistant")
