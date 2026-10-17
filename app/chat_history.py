import sqlite3
from pathlib import Path
from datetime import datetime

# Store the database in the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "chat_history.db"


def get_connection():
    """Create a connection to the SQLite database."""
    connection = sqlite3.connect(str(DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create the required database tables."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_username
            ON conversations(username)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id)
        """)


def create_conversation(username, title):
    """Create a conversation and return its ID."""
    title = title.strip()[:100] or "New conversation"
    created_at = datetime.now().isoformat(timespec="seconds")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO conversations (username, title, created_at)
            VALUES (?, ?, ?)
            """,
            (username, title, created_at),
        )
        return cursor.lastrowid


def save_message(conversation_id, role, content):
    """Save a user question or assistant response."""
    if role not in ("user", "assistant"):
        raise ValueError("Role must be 'user' or 'assistant'.")

    created_at = datetime.now().isoformat(timespec="seconds")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO messages
                (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, created_at),
        )


def get_conversations(username):
    """Return the user's conversations, newest first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, username, title, created_at
            FROM conversations
            WHERE username = ?
            ORDER BY id DESC
            """,
            (username,),
        ).fetchall()

        return [dict(row) for row in rows]


def get_messages(conversation_id):
    """Return all messages in a conversation, oldest first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

        return [dict(row) for row in rows]


def delete_conversation(username, conversation_id):
    """Delete a conversation belonging to the specified user."""
    with get_connection() as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        cursor = connection.execute(
            """
            DELETE FROM conversations
            WHERE id = ? AND username = ?
            """,
            (conversation_id, username),
        )

        return cursor.rowcount > 0
