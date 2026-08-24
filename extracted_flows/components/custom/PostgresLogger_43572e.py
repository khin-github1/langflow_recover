# Recovered Langflow component
# type: PostgresLogger
# class: PostgresLogger
# used in 2 flow(s): CRCV-v1.1, CRCV-v1.1.01
# json path: node.data.node.template.code.value

from datetime import datetime

import psycopg2

from langflow.custom import Component
from langflow.io import MessageTextInput, StrInput, Output
from langflow.schema.message import Message


class PostgresLogger(Component):
    """
    Log LLM answers into public.crcv_logs.

    - Input: answer_text (Message)
    - Side effect: INSERT row into crcv_logs
    - Output: same text as a Message
    """

    display_name = "Postgres Logger"
    description = "Logs LLM replies to Postgres (crcv_logs)."
    icon = "database"
    name = "PostgresLogger"

    # ---- Inputs ----
    inputs = [
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

    def log_and_forward(self) -> Message:
        text = self.answer_text or ""
        db_url = self.db_url or ""
        session_id = self.session_id or "default_session"
        llm_name = self.llm_name or "unknown_model"

        self.status = f"Logging len={len(text)}"

        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()

            # 1) Compute next turn_index for this session_id
            cur.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 "
                "FROM crcv_logs WHERE session_id = %s",
                (session_id,),
            )
            next_turn = cur.fetchone()[0]

            # 2) Insert the log row
            cur.execute(
                """
                INSERT INTO crcv_logs
                    (session_id, turn_index, llm_name, answer_text, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, next_turn, llm_name, text, datetime.utcnow()),
            )

            conn.commit()
            cur.close()
            conn.close()

            self.status = (
                f"Logged session={session_id} turn={next_turn} model={llm_name}"
            )
        except Exception as e:
            # Don't crash the flow, just show error in node status
            self.status = f"Postgres logging error: {e}"

        # Pass answer through to downstream nodes
        return Message(text=text, sender="assistant")
