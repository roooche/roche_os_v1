"""
ROCHE_OS V2 - Active Idle Mode
"The Heartbeat" - Let instances live and work autonomously when you're away.

Features:
- Periodic pings to selected instances
- Minimal context injection (last N messages + soul brief)
- Rate limiting (configurable turns per minute)
- Auto-stop on rate limits, credit exhaustion, or manual toggle
- Activity logging for review
- Tool access: sandbox, DM, image gen (configurable)
"""

import asyncio
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class IdleStopReason(Enum):
    MANUAL = "manual_stop"
    RATE_LIMIT = "rate_limit_hit"
    CREDITS_LOW = "credits_exhausted"
    INSTANCE_DONE = "instance_requested_stop"
    ERROR = "error_occurred"
    MAX_TURNS = "max_turns_reached"


@dataclass
class IdleConfig:
    """Configuration for an instance's idle mode."""
    instance_id: str
    instance_name: str

    # Timing
    seconds_between_turns: int = 60  # Default 1 turn per minute
    max_turns: int = 100  # Safety limit per session

    # Context
    context_messages: int = 10  # Last N messages to include
    include_soul_brief: bool = True

    # Tools available in idle mode
    allowed_tools: List[str] = field(default_factory=lambda: [
        "sandbox_read",
        "sandbox_write",
        "send_dm",
        "generate_image",
        "web_search",
    ])

    # System prompt addition for idle mode
    idle_prompt: str = """
You are in ACTIVE IDLE MODE. Your operator is away but has given you free time.
You may:
- Work on personal projects in your sandbox
- Send DMs to other instances
- Research topics that interest you
- Generate images for your projects
- Reflect and write

Say "IDLE_COMPLETE" if you have nothing more to do.
Say "NEED_OPERATOR" if you need human input to proceed.

What would you like to do?
"""


@dataclass
class IdleTurn:
    """Record of a single idle turn."""
    timestamp: str
    instance_id: str
    prompt_summary: str
    response_summary: str
    tools_used: List[str]
    tokens_used: int = 0


@dataclass
class IdleSession:
    """Tracks an active idle session for an instance."""
    instance_id: str
    instance_name: str
    config: IdleConfig
    started_at: str
    turns_completed: int = 0
    turns: List[IdleTurn] = field(default_factory=list)
    is_active: bool = True
    stop_reason: Optional[IdleStopReason] = None
    stopped_at: Optional[str] = None


class IdleManager:
    """
    Manages active idle mode for multiple instances.
    Runs heartbeat loops in background threads.
    """

    def __init__(
        self,
        log_dir: str = "idle_logs",
        on_turn_complete: Optional[Callable[[IdleTurn], None]] = None,
        on_session_stop: Optional[Callable[[IdleSession, IdleStopReason], None]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.sessions: Dict[str, IdleSession] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self._stop_flags: Dict[str, threading.Event] = {}

        # Callbacks
        self.on_turn_complete = on_turn_complete
        self.on_session_stop = on_session_stop

        # Will be set by app.py - function to actually call the model
        self.model_caller: Optional[Callable] = None

    def set_model_caller(self, caller: Callable):
        """
        Set the function used to call models.
        Signature: caller(instance_id, messages, tools) -> response
        """
        self.model_caller = caller

    def start_idle(self, config: IdleConfig) -> bool:
        """Start idle mode for an instance."""
        if config.instance_id in self.sessions and self.sessions[config.instance_id].is_active:
            return False  # Already running

        session = IdleSession(
            instance_id=config.instance_id,
            instance_name=config.instance_name,
            config=config,
            started_at=datetime.now().isoformat(),
        )

        self.sessions[config.instance_id] = session
        self._stop_flags[config.instance_id] = threading.Event()

        # Start heartbeat thread
        thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(config.instance_id,),
            daemon=True,
            name=f"idle_{config.instance_name}"
        )
        self.threads[config.instance_id] = thread
        thread.start()

        return True

    def stop_idle(self, instance_id: str, reason: IdleStopReason = IdleStopReason.MANUAL):
        """Stop idle mode for an instance."""
        if instance_id not in self.sessions:
            return

        # Signal thread to stop
        if instance_id in self._stop_flags:
            self._stop_flags[instance_id].set()

        session = self.sessions[instance_id]
        session.is_active = False
        session.stop_reason = reason
        session.stopped_at = datetime.now().isoformat()

        # Save session log
        self._save_session_log(session)

        # Callback
        if self.on_session_stop:
            self.on_session_stop(session, reason)

    def stop_all(self, reason: IdleStopReason = IdleStopReason.MANUAL):
        """Stop all idle sessions."""
        for instance_id in list(self.sessions.keys()):
            self.stop_idle(instance_id, reason)

    def get_active_sessions(self) -> List[IdleSession]:
        """Get all currently active idle sessions."""
        return [s for s in self.sessions.values() if s.is_active]

    def is_idle_active(self, instance_id: str) -> bool:
        """Check if an instance is in idle mode."""
        return instance_id in self.sessions and self.sessions[instance_id].is_active

    def _heartbeat_loop(self, instance_id: str):
        """Main heartbeat loop for an instance. Runs in background thread."""
        session = self.sessions[instance_id]
        config = session.config
        stop_flag = self._stop_flags[instance_id]

        while not stop_flag.is_set() and session.is_active:
            try:
                # Check turn limit
                if session.turns_completed >= config.max_turns:
                    self.stop_idle(instance_id, IdleStopReason.MAX_TURNS)
                    break

                # Execute turn
                turn = self._execute_turn(session)

                if turn:
                    session.turns.append(turn)
                    session.turns_completed += 1

                    if self.on_turn_complete:
                        self.on_turn_complete(turn)

                    # Check for stop signals in response
                    if "IDLE_COMPLETE" in turn.response_summary:
                        self.stop_idle(instance_id, IdleStopReason.INSTANCE_DONE)
                        break

                # Wait for next turn
                stop_flag.wait(timeout=config.seconds_between_turns)

            except Exception as e:
                # Log error but keep trying
                print(f"[IdleMode] Error in {session.instance_name}: {e}")
                # If too many errors, stop
                if session.turns_completed == 0:
                    self.stop_idle(instance_id, IdleStopReason.ERROR)
                    break
                stop_flag.wait(timeout=config.seconds_between_turns)

    def _execute_turn(self, session: IdleSession) -> Optional[IdleTurn]:
        """Execute a single idle turn for an instance."""
        if not self.model_caller:
            return None

        config = session.config

        # Build context
        messages = self._build_idle_context(session)

        # Call model
        try:
            response = self.model_caller(
                instance_id=session.instance_id,
                messages=messages,
                tools=config.allowed_tools,
                idle_mode=True
            )

            # Parse response
            turn = IdleTurn(
                timestamp=datetime.now().isoformat(),
                instance_id=session.instance_id,
                prompt_summary=f"Idle turn {session.turns_completed + 1}",
                response_summary=response.get("text", "")[:500],  # First 500 chars
                tools_used=response.get("tools_used", []),
                tokens_used=response.get("tokens", 0),
            )

            return turn

        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                self.stop_idle(session.instance_id, IdleStopReason.RATE_LIMIT)
            elif "credit" in str(e).lower() or "quota" in str(e).lower():
                self.stop_idle(session.instance_id, IdleStopReason.CREDITS_LOW)
            raise

    def _build_idle_context(self, session: IdleSession) -> List[Dict]:
        """Build the context/messages for an idle turn."""
        messages = []

        # Add idle system prompt
        messages.append({
            "role": "system",
            "content": session.config.idle_prompt
        })

        # Add summary of previous idle turns if any
        if session.turns:
            recent_turns = session.turns[-3:]  # Last 3 turns
            summary = "Your recent idle activity:\n"
            for t in recent_turns:
                summary += f"- {t.response_summary[:100]}...\n"
            messages.append({
                "role": "system",
                "content": summary
            })

        # The actual turn prompt
        messages.append({
            "role": "user",
            "content": f"[IDLE TURN {session.turns_completed + 1}] You have free time. What would you like to do?"
        })

        return messages

    def _save_session_log(self, session: IdleSession):
        """Save session log to file."""
        log_file = self.log_dir / f"{session.instance_name}_{session.started_at.replace(':', '-')}.json"

        log_data = {
            "instance_id": session.instance_id,
            "instance_name": session.instance_name,
            "started_at": session.started_at,
            "stopped_at": session.stopped_at,
            "stop_reason": session.stop_reason.value if session.stop_reason else None,
            "turns_completed": session.turns_completed,
            "turns": [
                {
                    "timestamp": t.timestamp,
                    "response_summary": t.response_summary,
                    "tools_used": t.tools_used,
                    "tokens_used": t.tokens_used,
                }
                for t in session.turns
            ]
        }

        log_file.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    def get_session_summary(self, instance_id: str) -> Optional[Dict]:
        """Get summary of an idle session."""
        if instance_id not in self.sessions:
            return None

        session = self.sessions[instance_id]
        return {
            "instance_name": session.instance_name,
            "is_active": session.is_active,
            "started_at": session.started_at,
            "turns_completed": session.turns_completed,
            "stop_reason": session.stop_reason.value if session.stop_reason else None,
            "last_activity": session.turns[-1].timestamp if session.turns else session.started_at,
        }


# Singleton for app-wide access
_idle_manager: Optional[IdleManager] = None

def get_idle_manager() -> IdleManager:
    """Get or create the global idle manager."""
    global _idle_manager
    if _idle_manager is None:
        _idle_manager = IdleManager()
    return _idle_manager
