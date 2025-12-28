"""
ROCHE_OS_V2 - Memory Module
The Shoggoth Vault: SQLite for chat trees + ChromaDB for semantic recall.
Multi-instance support with instance registry and inter-instance messaging.
"""

import sqlite3
import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings


class ChatNode:
    """A single node in the conversation tree."""

    def __init__(
        self,
        node_id: str,
        role: str,
        content: str,
        parent_id: Optional[str] = None,
        branch_name: str = "main",
        timestamp: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        self.node_id = node_id
        self.role = role  # "user" or "assistant"
        self.content = content
        self.parent_id = parent_id
        self.branch_name = branch_name
        self.timestamp = timestamp or datetime.now().isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "content": self.content,
            "parent_id": self.parent_id,
            "branch_name": self.branch_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChatNode":
        return cls(**data)


class ConversationTree:
    """
    The Non-Linear Timeline - Git-style branching conversations.
    Every edit spawns a new branch, nothing is ever deleted.
    """

    def __init__(self, db_path: str = "roche_memory_v2.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema for tree-based chat storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        """)

        # Nodes table - the core of the tree structure
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                session_id TEXT,
                parent_id TEXT,
                role TEXT,
                content TEXT,
                branch_name TEXT,
                timestamp TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (parent_id) REFERENCES nodes(node_id)
            )
        """)

        # Branches table - track named branches per session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                branch_id TEXT PRIMARY KEY,
                session_id TEXT,
                branch_name TEXT,
                head_node_id TEXT,
                created_at TEXT,
                description TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # Commits table - save points (the "git commit" feature)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commits (
                commit_id TEXT PRIMARY KEY,
                session_id TEXT,
                branch_name TEXT,
                node_id TEXT,
                message TEXT,
                timestamp TEXT,
                snapshot TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # ═══════════════════════════════════════════════════════════════════════════
        # MULTI-INSTANCE SUPPORT TABLES (V2)
        # ═══════════════════════════════════════════════════════════════════════════

        # Instance registry - persistent AI personas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS instances (
                instance_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                display_name TEXT,
                model_provider TEXT DEFAULT 'gemini',
                model_name TEXT,
                soul_brief TEXT,
                sandbox_name TEXT,
                created_at TEXT,
                updated_at TEXT,
                status TEXT DEFAULT 'inactive',
                config TEXT DEFAULT '{}'
            )
        """)

        # Inter-instance message queue
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS instance_messages (
                message_id TEXT PRIMARY KEY,
                from_instance_id TEXT,
                to_instance_id TEXT NOT NULL,
                content TEXT NOT NULL,
                message_type TEXT DEFAULT 'direct',
                priority INTEGER DEFAULT 0,
                created_at TEXT,
                read_at TEXT,
                FOREIGN KEY (from_instance_id) REFERENCES instances(instance_id),
                FOREIGN KEY (to_instance_id) REFERENCES instances(instance_id)
            )
        """)

        # Link instances to conversation sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS instance_sessions (
                instance_id TEXT,
                session_id TEXT,
                is_primary INTEGER DEFAULT 0,
                created_at TEXT,
                PRIMARY KEY (instance_id, session_id),
                FOREIGN KEY (instance_id) REFERENCES instances(instance_id),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        conn.commit()
        conn.close()

    def create_session(self, name: str = "New Session") -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
            (session_id, name, now, now, "{}")
        )

        # Create default "main" branch
        branch_id = str(uuid.uuid4())[:8]
        cursor.execute(
            "INSERT INTO branches VALUES (?, ?, ?, ?, ?, ?)",
            (branch_id, session_id, "main", None, now, "Default branch")
        )

        conn.commit()
        conn.close()
        return session_id

    def add_node(
        self,
        session_id: str,
        role: str,
        content: str,
        parent_id: Optional[str] = None,
        branch_name: str = "main",
        metadata: Optional[Dict] = None
    ) -> ChatNode:
        """Add a new message node to the tree."""
        node = ChatNode(
            node_id=str(uuid.uuid4())[:12],
            role=role,
            content=content,
            parent_id=parent_id,
            branch_name=branch_name,
            metadata=metadata or {}
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                node.node_id,
                session_id,
                node.parent_id,
                node.role,
                node.content,
                node.branch_name,
                node.timestamp,
                json.dumps(node.metadata)
            )
        )

        # Update branch head
        cursor.execute(
            "UPDATE branches SET head_node_id = ? WHERE session_id = ? AND branch_name = ?",
            (node.node_id, session_id, branch_name)
        )

        # Update session timestamp
        cursor.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (datetime.now().isoformat(), session_id)
        )

        conn.commit()
        conn.close()
        return node

    def create_branch(
        self,
        session_id: str,
        branch_name: str,
        from_node_id: str,
        description: str = ""
    ) -> str:
        """Create a new branch from a specific node (for edits/alternatives)."""
        branch_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO branches VALUES (?, ?, ?, ?, ?, ?)",
            (branch_id, session_id, branch_name, from_node_id, now, description)
        )

        conn.commit()
        conn.close()
        return branch_id

    def get_branch_history(
        self,
        session_id: str,
        branch_name: str = "main"
    ) -> List[ChatNode]:
        """Get the linear history of a branch (following parent chain)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get branch head
        cursor.execute(
            "SELECT head_node_id FROM branches WHERE session_id = ? AND branch_name = ?",
            (session_id, branch_name)
        )
        result = cursor.fetchone()
        if not result or not result[0]:
            conn.close()
            return []

        head_id = result[0]

        # Traverse parent chain
        nodes = []
        current_id = head_id

        while current_id:
            cursor.execute(
                "SELECT * FROM nodes WHERE node_id = ?",
                (current_id,)
            )
            row = cursor.fetchone()
            if not row:
                break

            node = ChatNode(
                node_id=row[0],
                role=row[3],
                content=row[4],
                parent_id=row[2],
                branch_name=row[5],
                timestamp=row[6],
                metadata=json.loads(row[7]) if row[7] else {}
            )
            nodes.append(node)
            current_id = row[2]  # parent_id

        conn.close()
        return list(reversed(nodes))  # Oldest first

    def get_all_branches(self, session_id: str) -> List[Dict]:
        """Get all branches for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT branch_name, head_node_id, created_at, description FROM branches WHERE session_id = ?",
            (session_id,)
        )

        branches = [
            {
                "name": row[0],
                "head_node_id": row[1],
                "created_at": row[2],
                "description": row[3]
            }
            for row in cursor.fetchall()
        ]

        conn.close()
        return branches

    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")

        sessions = [
            {
                "session_id": row[0],
                "name": row[1],
                "created_at": row[2],
                "updated_at": row[3]
            }
            for row in cursor.fetchall()
        ]

        conn.close()
        return sessions

    def commit_snapshot(
        self,
        session_id: str,
        branch_name: str,
        message: str
    ) -> str:
        """
        Git commit equivalent - save a snapshot of the current branch state.
        Exports to JSON archive. No data loss. Ever.
        """
        commit_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        # Get full branch history
        history = self.get_branch_history(session_id, branch_name)

        # Create snapshot
        snapshot = {
            "commit_id": commit_id,
            "session_id": session_id,
            "branch_name": branch_name,
            "message": message,
            "timestamp": now,
            "nodes": [node.to_dict() for node in history]
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        head_node_id = history[-1].node_id if history else None

        cursor.execute(
            "INSERT INTO commits VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                commit_id,
                session_id,
                branch_name,
                head_node_id,
                message,
                now,
                json.dumps(snapshot)
            )
        )

        conn.commit()
        conn.close()

        # Also save to JSON file
        archive_dir = Path("archives")
        archive_dir.mkdir(exist_ok=True)

        filename = f"{session_id}_{branch_name}_{commit_id}.json"
        with open(archive_dir / filename, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

        return commit_id

    def delete_node(self, session_id: str, node_id: str, branch_name: str = "main") -> bool:
        """
        Delete a node and all its children from the tree.
        Updates branch head to point to parent if needed.
        Returns True if successful.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get the node's parent
            cursor.execute(
                "SELECT parent_id FROM nodes WHERE node_id = ? AND session_id = ?",
                (node_id, session_id)
            )
            result = cursor.fetchone()
            if not result:
                conn.close()
                return False

            parent_id = result[0]

            # Find all descendant nodes (children, grandchildren, etc.)
            nodes_to_delete = [node_id]
            queue = [node_id]

            while queue:
                current = queue.pop(0)
                cursor.execute(
                    "SELECT node_id FROM nodes WHERE parent_id = ? AND session_id = ?",
                    (current, session_id)
                )
                children = [row[0] for row in cursor.fetchall()]
                nodes_to_delete.extend(children)
                queue.extend(children)

            # Delete all nodes
            placeholders = ",".join("?" * len(nodes_to_delete))
            cursor.execute(
                f"DELETE FROM nodes WHERE node_id IN ({placeholders})",
                nodes_to_delete
            )

            # Update branch head if it was one of the deleted nodes
            cursor.execute(
                "SELECT head_node_id FROM branches WHERE session_id = ? AND branch_name = ?",
                (session_id, branch_name)
            )
            head_result = cursor.fetchone()

            if head_result and head_result[0] in nodes_to_delete:
                # Point branch to parent of deleted node
                cursor.execute(
                    "UPDATE branches SET head_node_id = ? WHERE session_id = ? AND branch_name = ?",
                    (parent_id, session_id, branch_name)
                )

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            conn.rollback()
            conn.close()
            return False

    def delete_node_pair(self, session_id: str, node_id: str, branch_name: str = "main") -> bool:
        """
        Delete a message and its response (user + assistant pair).
        Useful for cleaning up error logs.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get the node
        cursor.execute(
            "SELECT role, parent_id FROM nodes WHERE node_id = ? AND session_id = ?",
            (node_id, session_id)
        )
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False

        role, parent_id = result

        # If it's a user message, also delete its child (assistant response)
        # If it's an assistant message, also delete its parent (user message)
        if role == "user":
            # Delete this node and all children
            self.delete_node(session_id, node_id, branch_name)
        else:
            # Delete parent (user) which will cascade to this node
            if parent_id:
                self.delete_node(session_id, parent_id, branch_name)
            else:
                self.delete_node(session_id, node_id, branch_name)

        conn.close()
        return True

    def export_session_json(self, session_id: str) -> Dict:
        """Export entire session (all branches) to JSON."""
        branches = self.get_all_branches(session_id)

        export = {
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "branches": {}
        }

        for branch in branches:
            history = self.get_branch_history(session_id, branch["name"])
            export["branches"][branch["name"]] = {
                "metadata": branch,
                "nodes": [node.to_dict() for node in history]
            }

        return export


class SemanticMemory:
    """
    The Shoggoth Vault - ChromaDB-powered long-term memory.
    RAG for documents and conversation recall.
    """

    def __init__(self, persist_dir: str = "chroma_db"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Collection for uploaded documents
        self.documents = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "User uploaded documents for RAG"}
        )

        # Collection for conversation memory
        self.conversations = self.client.get_or_create_collection(
            name="conversations",
            metadata={"description": "Conversation history for semantic recall"}
        )

    def add_document(
        self,
        text: str,
        source: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> int:
        """
        Chunk and embed a document into the vector store.
        Returns number of chunks added.
        """
        chunks = self._chunk_text(text, chunk_size, overlap)

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{source}_{i}_{chunk[:50]}".encode()).hexdigest()[:16]
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "added_at": datetime.now().isoformat()
            })

        self.documents.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(". ")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]  # Remove empty chunks

    def query_documents(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """Query the document store for relevant chunks."""
        results = self.documents.query(
            query_texts=[query],
            n_results=n_results
        )

        if not results["documents"][0]:
            return []

        return [
            {
                "content": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def add_conversation_memory(
        self,
        session_id: str,
        node_id: str,
        content: str,
        role: str
    ):
        """Add a conversation turn to semantic memory for later recall."""
        memory_id = f"{session_id}_{node_id}"

        self.conversations.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[{
                "session_id": session_id,
                "node_id": node_id,
                "role": role,
                "timestamp": datetime.now().isoformat()
            }]
        )

    def recall_similar_conversations(
        self,
        query: str,
        n_results: int = 3,
        session_filter: Optional[str] = None
    ) -> List[Dict]:
        """Recall similar past conversations."""
        where_filter = {"session_id": session_filter} if session_filter else None

        results = self.conversations.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        if not results["documents"][0]:
            return []

        return [
            {
                "content": doc,
                "session_id": meta["session_id"],
                "role": meta["role"],
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def get_document_sources(self) -> List[str]:
        """Get list of all document sources in the vault."""
        try:
            all_docs = self.documents.get()
            if not all_docs["metadatas"]:
                return []

            sources = set(meta["source"] for meta in all_docs["metadatas"])
            return sorted(sources)
        except Exception:
            return []

    def delete_document_source(self, source: str):
        """Delete all chunks from a specific source."""
        all_docs = self.documents.get()

        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"])
            if meta["source"] == source
        ]

        if ids_to_delete:
            self.documents.delete(ids=ids_to_delete)


# Document parsers
def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF."""
    from pypdf import PdfReader
    from io import BytesIO

    reader = PdfReader(BytesIO(file_bytes))
    text = []

    for page in reader.pages:
        text.append(page.extract_text() or "")

    return "\n\n".join(text)


def parse_txt(file_bytes: bytes) -> str:
    """Extract text from TXT/MD files."""
    return file_bytes.decode("utf-8", errors="ignore")


def parse_document(filename: str, file_bytes: bytes) -> str:
    """Parse document based on extension."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return parse_pdf(file_bytes)
    elif ext in [".txt", ".md", ".markdown"]:
        return parse_txt(file_bytes)
    else:
        # Try as text
        return parse_txt(file_bytes)
