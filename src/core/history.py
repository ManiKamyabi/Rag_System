"""
حافظهٔ ماندگار چت (SQLite)
==========================
سشن‌ها و پیام‌ها در دیسک ذخیره می‌شوند تا مدل «هیستوری چت‌های قبلی»
را بداند و گفتگوها بین اجراهای مختلف برنامه ادامه یابند. تعداد
پیام‌هایی که به مدل داده می‌شود (عمق حافظه) در UI قابل تنظیم است.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Message:
    """یک پیام چت."""

    id: int | None
    session_id: int
    role: str                 # user | assistant
    content: str
    meta: dict = field(default_factory=dict)   # منابع/آمار پاسخ
    created_at: str = ""


@dataclass
class Session:
    """یک گفتگو."""

    id: int
    title: str
    created_at: str
    updated_at: str


class ChatHistory:
    """مدیریت تاریخچهٔ چت با ذخیره‌سازی ماندگار در SQLite."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    # ── راه‌اندازی ──────────────────────────────────────────
    def _create_tables(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    title      TEXT    NOT NULL DEFAULT 'گفتگوی جدید',
                    created_at TEXT    NOT NULL,
                    updated_at TEXT    NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role       TEXT    NOT NULL CHECK (role IN ('user','assistant')),
                    content    TEXT    NOT NULL,
                    meta       TEXT,
                    created_at TEXT    NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages (session_id, id);
                """
            )

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _parse_meta(raw: str | None) -> dict:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    # ── سشن‌ها ──────────────────────────────────────────────
    def create_session(self, title: str = "گفتگوی جدید") -> Session:
        """ساخت گفتگوی جدید."""
        now = self._now()
        with self._connection:
            cursor = self._connection.execute(
                "INSERT INTO sessions (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            session_id = cursor.lastrowid
        return Session(id=session_id, title=title, created_at=now, updated_at=now)

    def delete_session(self, session_id: int) -> None:
        """حذف گفتگو (پیام‌هایش هم cascade حذف می‌شوند)."""
        with self._connection:
            self._connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def rename_session(self, session_id: int, title: str) -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, self._now(), session_id),
            )

    def list_sessions(self, limit: int = 50) -> list[Session]:
        """فهرست گفتگوها از جدید به قدیم."""
        rows = self._connection.execute(
            "SELECT id, title, created_at, updated_at FROM sessions "
            "ORDER BY updated_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            Session(id=row["id"], title=row["title"],
                    created_at=row["created_at"], updated_at=row["updated_at"])
            for row in rows
        ]

    def get_session(self, session_id: int) -> Session | None:
        row = self._connection.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return Session(id=row["id"], title=row["title"],
                       created_at=row["created_at"], updated_at=row["updated_at"])

    def _touch(self, session_id: int) -> None:
        self._connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (self._now(), session_id),
        )

    # ── پیام‌ها ─────────────────────────────────────────────
    def add_message(self, session_id: int, role: str, content: str,
                    meta: dict | None = None) -> Message:
        """ثبت یک پیام؛ اگر اولین پیامِ کاربر باشد، عنوان سشن را هم تنظیم می‌کند."""
        now = self._now()
        meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
        with self._connection:
            cursor = self._connection.execute(
                "INSERT INTO messages (session_id, role, content, meta, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, meta_json, now),
            )
            if role == "user":
                count = self._connection.execute(
                    "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()["c"]
                if count <= 1:
                    self._connection.execute(
                        "UPDATE sessions SET title = ? WHERE id = ?",
                        (content.strip()[:50] or "گفتگوی جدید", session_id),
                    )
            self._touch(session_id)
            message_id = cursor.lastrowid
        return Message(id=message_id, session_id=session_id, role=role,
                       content=content, meta=meta or {}, created_at=now)

    def get_messages(self, session_id: int, limit: int | None = None) -> list[Message]:
        """پیام‌های یک گفتگو به ترتیب زمانی (با limit از آخرین پیام‌ها)."""
        if limit is not None and limit > 0:
            rows = self._connection.execute(
                "SELECT * FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = self._connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [
            Message(
                id=row["id"], session_id=row["session_id"], role=row["role"],
                content=row["content"], meta=self._parse_meta(row["meta"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def last_assistant_meta(self, session_id: int) -> dict:
        """متای آخرین پاسخ دستیار (برای بازنمایی منابع در UI)."""
        row = self._connection.execute(
            "SELECT meta FROM messages WHERE session_id = ? AND role = 'assistant' "
            "ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return self._parse_meta(row["meta"]) if row else {}

    # ── تبدیل برای مدل ──────────────────────────────────────
    @staticmethod
    def as_chat_messages(messages: list[Message]) -> list[dict]:
        """تبدیل پیام‌های تاریخچه به قالب OpenAI و حذف پیام‌های سرآغازِ ناقص.

        اگر پیام آخرِ انتخاب‌شده از نوع کاربر باشد، مدل پاسخ تکراری می‌دهد؛
        بنابراین پیام‌های کاربرِ ابتدای لیستِ بریده‌شده حذف می‌شوند تا دنباله
        همیشه با پیام دستیار تمام شود (پیام جدید کاربر بعداً اضافه می‌شود).
        """
        chat: list[dict] = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        while chat and chat[0]["role"] == "assistant":
            chat.pop(0)
        return chat

    def close(self) -> None:
        self._connection.close()
