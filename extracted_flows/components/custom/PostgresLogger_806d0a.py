# Recovered Langflow component
# type: PostgresLogger
# class: PostgresLogger
# used in 2 flow(s): CRCV-v1.1, CRCV-v1.1.01
# json path: node.data.node.template.code.value

from datetime import datetime

import psycopg2

from langflow.custom import Component
from langflow.io import MessageInput, StrInput, Output
from langflow.schema.message import Message


class PostgresLogger(Component):
    """
    Log LLM answers into public.crcv_logs.

    - Input: full Message from the generator or Chat Output
    - Side effect: INSERT row into crcv_logs
    - Output: the same Message (so you can pass it on to Chat Output)
    """

    display_name = "Postgres Logger"
    description = "Logs LLM replies to Postgres (crcv_logs)."
    icon = "database"
    name = "PostgresLogger"

    # ---- Inputs ----
    inputs = [
        # Full Message, not just text
        MessageInput(
            name="answer_message",
            display_name="Answer Message",
            info="Full Message coming from the generator or Chat Output.",
        ),
        StrInput(
            name="db_url",
            display_name="DB URL",
            value="postgresql://user:password@pgvector:5432/default",
            info="Postgres connection string",
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
            info="Same message as input, after logging.",
        )
    ]

    def log_and_forward(self) -> Message:
        msg = self.answer_message

        # Defensive: if something weird is wired in
        if isinstance(msg, Message):
            text = msg.text or ""
            # Use the session_id carried by LangFlow
            session_id = msg.session_id or "default_session"
        else:
            text = str(msg or "")
            session_id = "default_session"

        db_url = self.db_url or ""
        llm_name = self.llm_name or "unknown_model"

        self.status = f"Logging len={len(text)} session={session_id}"

        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()

            # Compute next turn_index for this session_id
            cur.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 "
                "FROM crcv_logs WHERE session_id = %s",
                (session_id,),
            )
            next_turn = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO crcv_logs
                    (session_id, turn_index, llm_name,
                     answer_text, latency_ms, total_tokens, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    next_turn,
                    llm_name,
                    text,
                    None,   # latency_ms placeholder
                    None,   # total_tokens placeholder
                    datetime.utcnow(),
                ),
            )

            conn.commit()
            cur.close()
            conn.close()

            self.status = (
                f"Logged s={session_id} turn={next_turn} model={llm_name}"
            )
        except Exception as e:
            self.status = f"Postgres logging error: {e}"

        # Pass the same message onward
        if isinstance(msg, Message):
            return msg
        return Message(text=text, sender="assistant", session_id=session_id)
