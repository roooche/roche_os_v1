"""
ROCHE_OS_V2 - Instance Management Module
Multi-instance AI persona support with persistent state and inter-instance messaging.
"""

import sqlite3
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class InstanceState:
    """Complete state container for a single AI instance."""

    # Identity
    instance_id: str
    name: str
    display_name: Optional[str] = None

    # Model configuration
    model_provider: str = "gemini"  # "gemini" or "claude"
    model_name: str = "gemini-1.5-pro-latest"

    # Soul/personality
    soul_brief: Optional[str] = None

    # Sandbox (isolated filesystem)
    sandbox_name: Optional[str] = None

    # Conversation context
    current_session_id: Optional[str] = None
    current_branch: str = "main"

    # Feature flags
    sliding_window_enabled: bool = False
    sliding_window_size: int = 50
    use_rag_memory: bool = True

    # Runtime state (not persisted)
    pending_screenshot: Optional[str] = None
    pending_scraped: List[Dict] = field(default_factory=list)
    pending_messages: List[Dict] = field(default_factory=list)

    # Status
    status: str = "inactive"  # "active", "inactive", "sleeping"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Extensible config
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary for database storage."""
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "display_name": self.display_name,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "soul_brief": self.soul_brief,
            "sandbox_name": self.sandbox_name,
            "current_session_id": self.current_session_id,
            "current_branch": self.current_branch,
            "sliding_window_enabled": self.sliding_window_enabled,
            "sliding_window_size": self.sliding_window_size,
            "use_rag_memory": self.use_rag_memory,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "InstanceState":
        """Deserialize from dictionary."""
        # Extract config and merge with defaults
        config = data.get("config", {})
        if isinstance(config, str):
            config = json.loads(config) if config else {}

        return cls(
            instance_id=data["instance_id"],
            name=data["name"],
            display_name=data.get("display_name"),
            model_provider=data.get("model_provider", "gemini"),
            model_name=data.get("model_name", "gemini-1.5-pro-latest"),
            soul_brief=data.get("soul_brief"),
            sandbox_name=data.get("sandbox_name"),
            current_session_id=config.get("current_session_id"),
            current_branch=config.get("current_branch", "main"),
            sliding_window_enabled=config.get("sliding_window_enabled", False),
            sliding_window_size=config.get("sliding_window_size", 50),
            use_rag_memory=config.get("use_rag_memory", True),
            status=data.get("status", "inactive"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            config=config
        )


class InstanceManager:
    """Central manager for all AI instances."""

    def __init__(self, db_path: str = "roche_memory_v2.db"):
        self.db_path = db_path

    def create_instance(
        self,
        name: str,
        model_provider: str = "gemini",
        model_name: Optional[str] = None,
        soul_brief: Optional[str] = None,
        sandbox_name: Optional[str] = None,
        display_name: Optional[str] = None
    ) -> InstanceState:
        """Create and register a new AI instance."""
        instance_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        # Set default model name based on provider
        if model_name is None:
            model_name = "gemini-1.5-pro-latest" if model_provider == "gemini" else "claude-sonnet-4-20250514"

        # Create sandbox if not specified
        if sandbox_name is None:
            sandbox_name = name.lower().replace(" ", "_")

        instance = InstanceState(
            instance_id=instance_id,
            name=name,
            display_name=display_name or name,
            model_provider=model_provider,
            model_name=model_name,
            soul_brief=soul_brief,
            sandbox_name=sandbox_name,
            status="active",
            created_at=now,
            updated_at=now
        )

        # Persist to database
        self._save_to_db(instance)

        return instance

    def _save_to_db(self, instance: InstanceState) -> None:
        """Save instance to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Merge runtime config into stored config
        config = {
            "current_session_id": instance.current_session_id,
            "current_branch": instance.current_branch,
            "sliding_window_enabled": instance.sliding_window_enabled,
            "sliding_window_size": instance.sliding_window_size,
            "use_rag_memory": instance.use_rag_memory,
            **instance.config
        }

        cursor.execute("""
            INSERT OR REPLACE INTO instances
            (instance_id, name, display_name, model_provider, model_name,
             soul_brief, sandbox_name, created_at, updated_at, status, config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            instance.instance_id,
            instance.name,
            instance.display_name,
            instance.model_provider,
            instance.model_name,
            instance.soul_brief,
            instance.sandbox_name,
            instance.created_at,
            datetime.now().isoformat(),
            instance.status,
            json.dumps(config)
        ))

        conn.commit()
        conn.close()

    def save_instance(self, instance: InstanceState) -> None:
        """Public method to persist instance state."""
        instance.updated_at = datetime.now().isoformat()
        self._save_to_db(instance)

    def load_instance(self, instance_id: str) -> Optional[InstanceState]:
        """Load instance from database by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM instances WHERE instance_id = ?",
            (instance_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_instance(row)

    def load_instance_by_name(self, name: str) -> Optional[InstanceState]:
        """Load instance from database by name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM instances WHERE name = ? COLLATE NOCASE",
            (name,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_instance(row)

    def _row_to_instance(self, row: tuple) -> InstanceState:
        """Convert database row to InstanceState."""
        return InstanceState.from_dict({
            "instance_id": row[0],
            "name": row[1],
            "display_name": row[2],
            "model_provider": row[3],
            "model_name": row[4],
            "soul_brief": row[5],
            "sandbox_name": row[6],
            "created_at": row[7],
            "updated_at": row[8],
            "status": row[9],
            "config": row[10]
        })

    def get_all_instances(self) -> List[InstanceState]:
        """Get all registered instances."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM instances ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_instance(row) for row in rows]

    def get_active_instances(self) -> List[InstanceState]:
        """Get all currently active instances."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM instances WHERE status = 'active' ORDER BY updated_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_instance(row) for row in rows]

    def delete_instance(self, instance_id: str) -> bool:
        """Delete an instance from the registry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Delete related messages
            cursor.execute(
                "DELETE FROM instance_messages WHERE from_instance_id = ? OR to_instance_id = ?",
                (instance_id, instance_id)
            )

            # Delete session links
            cursor.execute(
                "DELETE FROM instance_sessions WHERE instance_id = ?",
                (instance_id,)
            )

            # Delete instance
            cursor.execute(
                "DELETE FROM instances WHERE instance_id = ?",
                (instance_id,)
            )

            conn.commit()
            conn.close()
            return True
        except Exception:
            conn.rollback()
            conn.close()
            return False

    def link_session_to_instance(
        self,
        instance_id: str,
        session_id: str,
        is_primary: bool = False
    ) -> None:
        """Link a conversation session to an instance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO instance_sessions
            (instance_id, session_id, is_primary, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            instance_id,
            session_id,
            1 if is_primary else 0,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def get_instance_sessions(self, instance_id: str) -> List[str]:
        """Get all session IDs linked to an instance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT session_id FROM instance_sessions WHERE instance_id = ?",
            (instance_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]


class MessageQueue:
    """Manages inter-instance message passing."""

    def __init__(self, db_path: str = "roche_memory_v2.db"):
        self.db_path = db_path

    def send(
        self,
        from_instance_id: Optional[str],
        to_instance_id: str,
        content: str,
        message_type: str = "direct",
        priority: int = 0
    ) -> str:
        """Queue a message for delivery. Returns message_id."""
        message_id = str(uuid.uuid4())[:12]
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO instance_messages
            (message_id, from_instance_id, to_instance_id, content,
             message_type, priority, created_at, read_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """, (
            message_id,
            from_instance_id,
            to_instance_id,
            content,
            message_type,
            priority,
            now
        ))

        conn.commit()
        conn.close()

        return message_id

    def get_pending_messages(
        self,
        instance_id: str,
        mark_as_read: bool = False
    ) -> List[Dict]:
        """Get unread messages for an instance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT m.*, i.name as from_name
            FROM instance_messages m
            LEFT JOIN instances i ON m.from_instance_id = i.instance_id
            WHERE m.to_instance_id = ? AND m.read_at IS NULL
            ORDER BY m.priority DESC, m.created_at ASC
        """, (instance_id,))

        rows = cursor.fetchall()

        messages = []
        for row in rows:
            messages.append({
                "message_id": row[0],
                "from_instance_id": row[1],
                "to_instance_id": row[2],
                "content": row[3],
                "message_type": row[4],
                "priority": row[5],
                "created_at": row[6],
                "from_name": row[8] if len(row) > 8 else "System"
            })

        if mark_as_read and messages:
            message_ids = [m["message_id"] for m in messages]
            placeholders = ",".join("?" * len(message_ids))
            cursor.execute(
                f"UPDATE instance_messages SET read_at = ? WHERE message_id IN ({placeholders})",
                [datetime.now().isoformat()] + message_ids
            )
            conn.commit()

        conn.close()
        return messages

    def mark_read(self, message_id: str) -> None:
        """Mark a single message as read."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE instance_messages SET read_at = ? WHERE message_id = ?",
            (datetime.now().isoformat(), message_id)
        )

        conn.commit()
        conn.close()

    def count_unread(self, instance_id: str) -> int:
        """Count unread messages for an instance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM instance_messages WHERE to_instance_id = ? AND read_at IS NULL",
            (instance_id,)
        )
        count = cursor.fetchone()[0]
        conn.close()

        return count

    def broadcast(
        self,
        from_instance_id: str,
        content: str,
        exclude_self: bool = True
    ) -> List[str]:
        """Send message to all active instances. Returns list of message_ids."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all active instances
        cursor.execute("SELECT instance_id FROM instances WHERE status = 'active'")
        targets = [row[0] for row in cursor.fetchall()]
        conn.close()

        if exclude_self and from_instance_id in targets:
            targets.remove(from_instance_id)

        message_ids = []
        for target_id in targets:
            msg_id = self.send(
                from_instance_id=from_instance_id,
                to_instance_id=target_id,
                content=content,
                message_type="broadcast"
            )
            message_ids.append(msg_id)

        return message_ids

    def get_conversation_thread(
        self,
        instance_a: str,
        instance_b: str,
        limit: int = 50
    ) -> List[Dict]:
        """Get message history between two instances."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT m.*,
                   i1.name as from_name,
                   i2.name as to_name
            FROM instance_messages m
            LEFT JOIN instances i1 ON m.from_instance_id = i1.instance_id
            LEFT JOIN instances i2 ON m.to_instance_id = i2.instance_id
            WHERE (m.from_instance_id = ? AND m.to_instance_id = ?)
               OR (m.from_instance_id = ? AND m.to_instance_id = ?)
            ORDER BY m.created_at DESC
            LIMIT ?
        """, (instance_a, instance_b, instance_b, instance_a, limit))

        rows = cursor.fetchall()
        conn.close()

        messages = []
        for row in rows:
            messages.append({
                "message_id": row[0],
                "from_instance_id": row[1],
                "to_instance_id": row[2],
                "content": row[3],
                "message_type": row[4],
                "priority": row[5],
                "created_at": row[6],
                "read_at": row[7],
                "from_name": row[8],
                "to_name": row[9]
            })

        return list(reversed(messages))  # Oldest first


def format_messages_for_context(messages: List[Dict]) -> str:
    """Format pending messages for injection into system prompt."""
    if not messages:
        return ""

    lines = [
        "",
        "# INCOMING MESSAGES FROM OTHER INSTANCES",
        "You have received the following messages. You may respond to them or acknowledge receipt.",
        ""
    ]

    for msg in messages:
        sender = msg.get("from_name", "System")
        priority_marker = "[URGENT] " if msg.get("priority", 0) > 0 else ""
        lines.append(f"**[FROM {sender}]** {priority_marker}{msg['content']}")
        lines.append("")

    return "\n".join(lines)
