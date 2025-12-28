"""
SANDBOX MODULE - Isolated Filesystem for AI Instances

Each instance (Tessera, Gemini, Vigil, etc.) gets their own sandbox directory.
They can read/write/create files ONLY within their sandbox.
No access to parent directories, main code, or other instances' sandboxes.

Security: Path traversal attacks are blocked. Symlinks are resolved and validated.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime


class SandboxError(Exception):
    """Raised when sandbox security is violated."""
    pass


class InstanceSandbox:
    """
    Provides sandboxed filesystem access for a specific AI instance.
    All paths are jail-rooted to the instance's sandbox directory.
    """

    # Base sandbox directory (relative to project root)
    SANDBOXES_ROOT = Path(__file__).parent.parent / "sandboxes"

    # Forbidden patterns in filenames
    FORBIDDEN_PATTERNS = ['..', '~', '$', '|', ';', '&', '`', '\x00']

    # Max file size (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    def __init__(self, instance_name: str):
        """
        Initialize sandbox for a specific instance.

        Args:
            instance_name: Name of the instance (e.g., 'tessera', 'gemini')
        """
        # Sanitize instance name
        self.instance_name = self._sanitize_name(instance_name)
        self.sandbox_root = (self.SANDBOXES_ROOT / self.instance_name).resolve()

        # Create sandbox if it doesn't exist
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize instance name to prevent path injection."""
        # Only allow alphanumeric, underscore, hyphen
        sanitized = ''.join(c for c in name.lower() if c.isalnum() or c in '_-')
        if not sanitized:
            raise SandboxError("Invalid instance name")
        return sanitized

    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve a path, ensuring it stays within sandbox.

        Args:
            path: Relative path within sandbox

        Returns:
            Resolved absolute Path object

        Raises:
            SandboxError: If path escapes sandbox
        """
        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in path:
                raise SandboxError(f"Forbidden pattern in path: {pattern}")

        # Resolve the full path
        if Path(path).is_absolute():
            raise SandboxError("Absolute paths not allowed")

        full_path = (self.sandbox_root / path).resolve()

        # Ensure resolved path is within sandbox
        try:
            full_path.relative_to(self.sandbox_root)
        except ValueError:
            raise SandboxError(f"Path escape attempt blocked: {path}")

        return full_path

    def read_file(self, path: str) -> str:
        """
        Read a file from the sandbox.

        Args:
            path: Relative path to file within sandbox

        Returns:
            File contents as string
        """
        full_path = self._validate_path(path)

        if not full_path.exists():
            raise SandboxError(f"File not found: {path}")

        if not full_path.is_file():
            raise SandboxError(f"Not a file: {path}")

        if full_path.stat().st_size > self.MAX_FILE_SIZE:
            raise SandboxError(f"File too large (max {self.MAX_FILE_SIZE // 1024 // 1024}MB)")

        return full_path.read_text(encoding='utf-8')

    def write_file(self, path: str, content: str, append: bool = False) -> Dict:
        """
        Write content to a file in the sandbox.

        Args:
            path: Relative path to file within sandbox
            content: Content to write
            append: If True, append instead of overwrite

        Returns:
            Dict with file info
        """
        if len(content) > self.MAX_FILE_SIZE:
            raise SandboxError(f"Content too large (max {self.MAX_FILE_SIZE // 1024 // 1024}MB)")

        full_path = self._validate_path(path)

        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)

        mode = 'a' if append else 'w'
        full_path.write_text(content, encoding='utf-8') if not append else \
            full_path.open(mode, encoding='utf-8').write(content)

        return {
            "path": path,
            "size": len(content),
            "action": "appended" if append else "written",
            "timestamp": datetime.now().isoformat()
        }

    def list_files(self, path: str = ".", recursive: bool = False) -> List[Dict]:
        """
        List files in a sandbox directory.

        Args:
            path: Relative path to directory (default: sandbox root)
            recursive: If True, list recursively

        Returns:
            List of file info dicts
        """
        full_path = self._validate_path(path)

        if not full_path.exists():
            raise SandboxError(f"Directory not found: {path}")

        if not full_path.is_dir():
            raise SandboxError(f"Not a directory: {path}")

        files = []
        pattern = '**/*' if recursive else '*'

        for item in full_path.glob(pattern):
            rel_path = item.relative_to(self.sandbox_root)
            files.append({
                "path": str(rel_path),
                "name": item.name,
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None,
                "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
            })

        return sorted(files, key=lambda x: (not x['is_dir'], x['path']))

    def delete_file(self, path: str) -> Dict:
        """
        Delete a file from the sandbox.

        Args:
            path: Relative path to file

        Returns:
            Dict with deletion info
        """
        full_path = self._validate_path(path)

        if not full_path.exists():
            raise SandboxError(f"File not found: {path}")

        if full_path.is_dir():
            raise SandboxError("Use delete_directory for directories")

        full_path.unlink()

        return {
            "path": path,
            "action": "deleted",
            "timestamp": datetime.now().isoformat()
        }

    def create_directory(self, path: str) -> Dict:
        """
        Create a directory in the sandbox.

        Args:
            path: Relative path for new directory

        Returns:
            Dict with creation info
        """
        full_path = self._validate_path(path)
        full_path.mkdir(parents=True, exist_ok=True)

        return {
            "path": path,
            "action": "created",
            "timestamp": datetime.now().isoformat()
        }

    def delete_directory(self, path: str, recursive: bool = False) -> Dict:
        """
        Delete a directory from the sandbox.

        Args:
            path: Relative path to directory
            recursive: If True, delete contents recursively

        Returns:
            Dict with deletion info
        """
        full_path = self._validate_path(path)

        if not full_path.exists():
            raise SandboxError(f"Directory not found: {path}")

        if not full_path.is_dir():
            raise SandboxError("Not a directory")

        # Prevent deleting sandbox root
        if full_path == self.sandbox_root:
            raise SandboxError("Cannot delete sandbox root")

        if recursive:
            shutil.rmtree(full_path)
        else:
            try:
                full_path.rmdir()
            except OSError:
                raise SandboxError("Directory not empty. Use recursive=True to delete contents.")

        return {
            "path": path,
            "action": "deleted",
            "recursive": recursive,
            "timestamp": datetime.now().isoformat()
        }

    def get_info(self) -> Dict:
        """Get sandbox info and stats."""
        total_size = 0
        file_count = 0
        dir_count = 0

        for item in self.sandbox_root.rglob('*'):
            if item.is_file():
                file_count += 1
                total_size += item.stat().st_size
            elif item.is_dir():
                dir_count += 1

        return {
            "instance": self.instance_name,
            "root": str(self.sandbox_root),
            "files": file_count,
            "directories": dir_count,
            "total_size_bytes": total_size,
            "total_size_human": f"{total_size / 1024:.1f} KB" if total_size < 1024*1024 else f"{total_size / 1024 / 1024:.1f} MB"
        }


def get_available_sandboxes() -> List[str]:
    """List all available instance sandboxes."""
    sandboxes_root = InstanceSandbox.SANDBOXES_ROOT
    if not sandboxes_root.exists():
        return []
    return [d.name for d in sandboxes_root.iterdir() if d.is_dir()]


def create_sandbox(instance_name: str) -> InstanceSandbox:
    """Create and return a new sandbox for an instance."""
    return InstanceSandbox(instance_name)
