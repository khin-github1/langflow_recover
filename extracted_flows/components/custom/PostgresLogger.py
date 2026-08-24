# Recovered Langflow component
# type: PostgresLogger
# class: PostgresLogger
# used in 7 flow(s): CRCV-v0.1 Evaluation, CRCV-v1.1, CRCV-v1.1.01, CRCV-v1.102, CRCV-v1.103 (Normal) Demo, CRCV-v1.103 (Normal) Khin (1), CRCV-v1.103 (Normal) Oak Backups
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

    inputs = [
        MessageInput(
            name="answer_message",
            display_name="Answer Message",
            info="Full Message coming from the generator or Chat Output.",
        ),
        StrInput(
            name="db_url",
            display_name="DB URL",
            value="postgresql://user:password@pgvector:5432/default",
        ),
        StrInput(
            name="llm_name",
            display_name="LLM Name",
            value="unknown_model",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            name="logged_message",
            display_name="Logged Message",
            method="log_and_forward",
        )
    ]

    def log_and_forward(self) -> Message:
        msg = self.answer_message

        # --- Extract text + session id from full Message ---
        if isinstance(msg, Message):
            text = msg.text or ""
            # Prefer LangFlow's session_id; fall back to flow_id
            session_id = (
                (msg.session_id or "").strip()
                or (getattr(msg, "flow_id", "") or "").strip()
                or "unknown_session"
            )
        else:
            text = str(msg or "")
            session_id = "unknown_session"

        db_url = self.db_url or ""
        llm_name = (self.llm_name or "").strip() or "unknown_model"

        self.status = f"Logging s={session_id} len={len(text)}"

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
                    None,  # latency_ms placeholder
                    None,  # total_tokens placeholder
                    datetime.utcnow(),
                ),
            )

            conn.commit()
            cur.close()
            conn.close()

            self.status = (
                f"Logged session={session_id} turn={next_turn} model={llm_name}"
            )
        except Exception as e:
            self.status = f"Postgres logging error: {e}"

        # Pass the same message on
        if isinstance(msg, Message):
            return msg
        return Message(text=text, sender="assistant", session_id=session_id)
