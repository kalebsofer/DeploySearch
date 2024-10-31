import psycopg2
from config.settings import get_settings
import uuid
import streamlit as st
from dataclasses import dataclass, field
from datetime import datetime, timezone

settings = get_settings()


@dataclass
class SearchLog:
    user_id: str
    session_id: str
    query: str
    selected_document: str
    similarity_score: float
    feedback_received: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LogManager:
    def __init__(self):
        self.settings = get_settings()

    def get_db_connection(self):
        return psycopg2.connect(
            dbname=self.settings.POSTGRES_DB,
            user=self.settings.POSTGRES_USER,
            password=self.settings.POSTGRES_PASSWORD,
            host="postgres",
            port="5432",
        )

    @staticmethod
    def get_or_create_user_id():
        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        return st.session_state.user_id

    @staticmethod
    def get_or_create_session_id():
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id

    def log_search(self, query: str, selected_document: str, similarity_score: float):
        log = SearchLog(
            user_id=self.get_or_create_user_id(),
            session_id=self.get_or_create_session_id(),
            query=query,
            selected_document=selected_document,
            similarity_score=similarity_score,
            created_at=datetime.now(timezone.utc),
        )

        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO search_logs 
                    (user_id, session_id, query, selected_document, similarity_score, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s AT TIME ZONE 'UTC')
                    """,
                    (
                        log.user_id,
                        log.session_id,
                        log.query,
                        log.selected_document,
                        log.similarity_score,
                        log.created_at,
                    ),
                )
            conn.commit()

    def update_feedback(self, query: str, selected_document: str):
        user_id = self.get_or_create_user_id()
        session_id = self.get_or_create_session_id()

        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE search_logs 
                    SET feedback_received = TRUE 
                    WHERE user_id = %s
                    AND session_id = %s
                    AND query = %s 
                    AND selected_document = %s 
                    AND created_at = (
                        SELECT created_at 
                        FROM search_logs 
                        WHERE user_id = %s 
                        AND session_id = %s 
                        AND query = %s 
                        AND selected_document = %s 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    )
                    """,
                    (
                        user_id,
                        session_id,
                        query,
                        selected_document,
                        user_id,
                        session_id,
                        query,
                        selected_document,
                    ),
                )
            conn.commit()


log_manager = LogManager()
