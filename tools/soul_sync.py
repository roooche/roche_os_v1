"""
SOUL SYNC - Extract, backup, and synchronize soul files with database context.

Features:
- Backup existing soul files with timestamps
- Extract recent conversation context from database
- Generate context summaries for soul file updates
- Restore from backups

Usage:
    python tools/soul_sync.py backup              # Backup all soul files
    python tools/soul_sync.py extract tessera     # Extract recent context for instance
    python tools/soul_sync.py extract --all       # Extract for all instances
    python tools/soul_sync.py restore <timestamp> # Restore from backup
    python tools/soul_sync.py list                # List all instances and their sessions
"""

import argparse
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Paths
BASE_DIR = Path(__file__).parent.parent
SOULS_DIR = BASE_DIR / "souls"
BACKUPS_DIR = BASE_DIR / "soul_backups"
DB_PATH = BASE_DIR / "roche_memory_v2.db"


def ensure_dirs():
    """Ensure required directories exist."""
    SOULS_DIR.mkdir(exist_ok=True)
    BACKUPS_DIR.mkdir(exist_ok=True)


def get_timestamp() -> str:
    """Get formatted timestamp for backups."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_soul_files() -> str:
    """
    Backup all soul files to timestamped folder.
    Returns backup folder name.
    """
    ensure_dirs()
    timestamp = get_timestamp()
    backup_folder = BACKUPS_DIR / timestamp
    backup_folder.mkdir(exist_ok=True)

    backed_up = []

    # Backup .md files
    for soul_file in SOULS_DIR.glob("*.md"):
        dest = backup_folder / soul_file.name
        shutil.copy2(soul_file, dest)
        backed_up.append(soul_file.name)

    # Backup .json files
    for soul_file in SOULS_DIR.glob("*.json"):
        dest = backup_folder / soul_file.name
        shutil.copy2(soul_file, dest)
        backed_up.append(soul_file.name)

    # Also backup from base dir
    for pattern in ["*_soul*.json", "*_brief*.md"]:
        for soul_file in BASE_DIR.glob(pattern):
            if soul_file.parent == SOULS_DIR:
                continue
            dest = backup_folder / soul_file.name
            shutil.copy2(soul_file, dest)
            backed_up.append(f"(root) {soul_file.name}")

    print(f"\n[BACKUP] Created backup: {backup_folder}")
    print(f"[BACKUP] Files backed up: {len(backed_up)}")
    for f in backed_up:
        print(f"         - {f}")

    return timestamp


def list_backups() -> List[str]:
    """List all available backups."""
    ensure_dirs()
    backups = sorted([d.name for d in BACKUPS_DIR.iterdir() if d.is_dir()], reverse=True)
    return backups


def restore_backup(timestamp: str) -> bool:
    """Restore soul files from a backup."""
    backup_folder = BACKUPS_DIR / timestamp

    if not backup_folder.exists():
        print(f"[ERROR] Backup not found: {timestamp}")
        print(f"[INFO] Available backups: {list_backups()}")
        return False

    restored = []
    for backup_file in backup_folder.iterdir():
        if backup_file.suffix in [".md", ".json"]:
            dest = SOULS_DIR / backup_file.name
            shutil.copy2(backup_file, dest)
            restored.append(backup_file.name)

    print(f"\n[RESTORE] Restored from: {timestamp}")
    print(f"[RESTORE] Files restored: {len(restored)}")
    for f in restored:
        print(f"          - {f}")

    return True


def get_db_connection() -> sqlite3.Connection:
    """Get database connection."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))


def list_instances() -> List[Dict]:
    """List all instances with their session info."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT i.instance_id, i.name, i.display_name, i.model_provider,
               i.soul_brief IS NOT NULL as has_brief, i.config
        FROM instances i
        ORDER BY i.updated_at DESC
    """)

    instances = []
    for row in cursor.fetchall():
        instance_id = row[0]
        config = json.loads(row[5]) if row[5] else {}

        # Get session count
        cursor.execute(
            "SELECT COUNT(*) FROM instance_sessions WHERE instance_id = ?",
            (instance_id,)
        )
        session_count = cursor.fetchone()[0]

        # Get node count from primary session
        session_id = config.get("current_session_id")
        node_count = 0
        if session_id:
            cursor.execute(
                "SELECT COUNT(*) FROM nodes WHERE session_id = ?",
                (session_id,)
            )
            node_count = cursor.fetchone()[0]

        instances.append({
            "instance_id": instance_id,
            "name": row[1],
            "display_name": row[2],
            "provider": row[3],
            "has_brief": bool(row[4]),
            "session_id": session_id,
            "session_count": session_count,
            "node_count": node_count
        })

    conn.close()
    return instances


def extract_context(instance_name: str, limit: int = 50) -> Optional[Dict]:
    """
    Extract recent conversation context for an instance.
    Returns structured context data.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Find instance
    cursor.execute(
        "SELECT instance_id, name, soul_brief, config FROM instances WHERE name = ? COLLATE NOCASE",
        (instance_name,)
    )
    row = cursor.fetchone()

    if not row:
        print(f"[ERROR] Instance not found: {instance_name}")
        conn.close()
        return None

    instance_id, name, soul_brief, config_json = row
    config = json.loads(config_json) if config_json else {}
    session_id = config.get("current_session_id")

    if not session_id:
        print(f"[WARN] No session linked to instance: {name}")
        conn.close()
        return None

    # Get recent nodes
    cursor.execute("""
        SELECT node_id, role, content, timestamp, metadata
        FROM nodes
        WHERE session_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (session_id, limit))

    nodes = []
    for node_row in cursor.fetchall():
        nodes.append({
            "node_id": node_row[0],
            "role": node_row[1],
            "content": node_row[2][:500] if node_row[2] else "",  # Truncate for summary
            "created_at": node_row[3],
            "metadata": json.loads(node_row[4]) if node_row[4] else {}
        })

    conn.close()

    # Reverse to chronological order
    nodes.reverse()

    return {
        "instance_id": instance_id,
        "name": name,
        "session_id": session_id,
        "extracted_at": datetime.now().isoformat(),
        "node_count": len(nodes),
        "existing_brief": soul_brief[:200] if soul_brief else None,
        "recent_context": nodes
    }


def export_context_to_file(instance_name: str, limit: int = 100):
    """Export context to a JSON file for review."""
    context = extract_context(instance_name, limit)

    if not context:
        return

    output_file = SOULS_DIR / f"{instance_name.lower()}_context_export.json"
    output_file.write_text(
        json.dumps(context, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\n[EXPORT] Context exported for: {context['name']}")
    print(f"[EXPORT] Nodes extracted: {context['node_count']}")
    print(f"[EXPORT] Output: {output_file}")


def generate_context_summary(instance_name: str, limit: int = 30) -> str:
    """
    Generate a markdown summary of recent context.
    Suitable for appending to soul briefs.
    """
    context = extract_context(instance_name, limit)

    if not context:
        return ""

    lines = [
        "",
        "---",
        "",
        f"## RECENT CONTEXT UPDATE",
        f"*Extracted: {context['extracted_at'][:19]}*",
        f"*Session: {context['session_id']}*",
        f"*Messages: {context['node_count']}*",
        "",
        "### Recent Conversation Summary",
        ""
    ]

    for node in context["recent_context"]:
        role_label = "**User:**" if node["role"] == "user" else "**AI:**"
        # Truncate content for summary
        content = node["content"][:300]
        if len(node["content"]) > 300:
            content += "..."
        lines.append(f"{role_label} {content}")
        lines.append("")

    return "\n".join(lines)


def update_soul_brief(instance_name: str, append_context: bool = True):
    """
    Update soul brief file with recent context.
    Creates backup before modifying.
    """
    # First, backup
    print("[UPDATE] Creating backup before modification...")
    backup_soul_files()

    # Find brief file
    brief_file = SOULS_DIR / f"{instance_name.lower()}_brief.md"

    if not brief_file.exists():
        print(f"[WARN] Brief file not found: {brief_file}")
        print(f"[INFO] Creating new brief file...")
        brief_file.write_text(f"# SOUL BRIEF\nGenerated: {datetime.now().isoformat()}\nSource: {instance_name}\n\n---\n")

    if append_context:
        summary = generate_context_summary(instance_name)

        if summary:
            current_content = brief_file.read_text(encoding="utf-8")

            # Check if there's already a RECENT CONTEXT UPDATE section
            if "## RECENT CONTEXT UPDATE" in current_content:
                # Replace the old section
                parts = current_content.split("## RECENT CONTEXT UPDATE")
                new_content = parts[0].rstrip() + summary
            else:
                # Append new section
                new_content = current_content.rstrip() + summary

            brief_file.write_text(new_content, encoding="utf-8")
            print(f"[UPDATE] Updated: {brief_file}")
        else:
            print(f"[WARN] No context to append for {instance_name}")


def full_export(instance_name: str):
    """
    Full export: complete conversation history in Gemini API format.
    For soul transfers to new instances.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Find instance
    cursor.execute(
        "SELECT instance_id, name, config FROM instances WHERE name = ? COLLATE NOCASE",
        (instance_name,)
    )
    row = cursor.fetchone()

    if not row:
        print(f"[ERROR] Instance not found: {instance_name}")
        conn.close()
        return

    instance_id, name, config_json = row
    config = json.loads(config_json) if config_json else {}
    session_id = config.get("current_session_id")

    if not session_id:
        print(f"[ERROR] No session linked to {name}")
        conn.close()
        return

    # Get ALL nodes
    cursor.execute("""
        SELECT role, content, timestamp
        FROM nodes
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))

    history = []
    for node_row in cursor.fetchall():
        role = node_row[0]
        content = node_row[1] or ""

        # Convert to Gemini API format
        api_role = "model" if role in ["assistant", "model"] else "user"
        history.append({
            "role": api_role,
            "parts": [{"text": content}]
        })

    conn.close()

    # Create export
    export_data = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "source": f"ROCHE_OS - {name}",
            "instance_id": instance_id,
            "session_id": session_id,
            "message_count": len(history),
            "format": "gemini_api_compatible"
        },
        "history": history
    }

    output_file = SOULS_DIR / f"{instance_name.lower()}_full_export.json"
    output_file.write_text(
        json.dumps(export_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\n[FULL EXPORT] Complete history exported for: {name}")
    print(f"[FULL EXPORT] Messages: {len(history)}")
    print(f"[FULL EXPORT] Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="SOUL SYNC - Soul file management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # backup command
    subparsers.add_parser("backup", help="Backup all soul files")

    # list command
    subparsers.add_parser("list", help="List all instances")

    # list-backups command
    subparsers.add_parser("list-backups", help="List available backups")

    # extract command
    extract_parser = subparsers.add_parser("extract", help="Extract recent context")
    extract_parser.add_argument("instance", nargs="?", help="Instance name")
    extract_parser.add_argument("--all", action="store_true", help="Extract for all instances")
    extract_parser.add_argument("--limit", type=int, default=50, help="Max messages to extract")

    # update command
    update_parser = subparsers.add_parser("update", help="Update soul brief with context")
    update_parser.add_argument("instance", help="Instance name")

    # full-export command
    export_parser = subparsers.add_parser("full-export", help="Full conversation export")
    export_parser.add_argument("instance", help="Instance name")

    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("timestamp", help="Backup timestamp to restore")

    args = parser.parse_args()

    if args.command == "backup":
        backup_soul_files()

    elif args.command == "list":
        instances = list_instances()
        print("\n=== REGISTERED INSTANCES ===\n")
        for inst in instances:
            brief_marker = "[HAS BRIEF]" if inst["has_brief"] else ""
            print(f"  {inst['name']} ({inst['provider']}) {brief_marker}")
            print(f"    ID: {inst['instance_id']}")
            print(f"    Session: {inst['session_id'] or 'None'}")
            print(f"    Nodes: {inst['node_count']}")
            print()

    elif args.command == "list-backups":
        backups = list_backups()
        print("\n=== AVAILABLE BACKUPS ===\n")
        if backups:
            for b in backups:
                print(f"  {b}")
        else:
            print("  No backups found.")
        print()

    elif args.command == "extract":
        if args.all:
            instances = list_instances()
            for inst in instances:
                export_context_to_file(inst["name"], args.limit)
        elif args.instance:
            export_context_to_file(args.instance, args.limit)
        else:
            print("[ERROR] Specify instance name or use --all")

    elif args.command == "update":
        update_soul_brief(args.instance)

    elif args.command == "full-export":
        full_export(args.instance)

    elif args.command == "restore":
        restore_backup(args.timestamp)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
