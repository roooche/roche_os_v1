"""
DIALOGUE MODULE - Model-to-Model Conversation

Allows two AI models (Gemini and Claude) to converse with each other.
The Colosseum of Minds.
"""

import time
from typing import Generator, Dict, List, Optional
from dataclasses import dataclass

import google.generativeai as genai

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


@dataclass
class DialogueTurn:
    """A single turn in the dialogue."""
    speaker: str  # "gemini" or "claude"
    content: str
    timestamp: float


class ModelDialogue:
    """
    Manages a conversation between two AI models.
    """

    def __init__(
        self,
        gemini_api_key: str,
        claude_api_key: str,
        gemini_model: str = "gemini-1.5-pro-latest",
        claude_model: str = "claude-3-5-sonnet-latest",
        gemini_system: str = None,
        claude_system: str = None,
        gemini_soul_brief: str = None,
        claude_soul_brief: str = None,
        soul_brief: str = None  # Deprecated: use claude_soul_brief
    ):
        self.gemini_api_key = gemini_api_key
        self.claude_api_key = claude_api_key
        self.gemini_model = gemini_model
        self.claude_model = claude_model

        # System prompts
        self.gemini_system = gemini_system or self._default_gemini_system()
        self.claude_system = claude_system or self._default_claude_system()

        # Soul briefs for identity preservation
        if gemini_soul_brief:
            self.gemini_system += f"\n\n---\n\n# YOUR IDENTITY\n\n{gemini_soul_brief}"

        # Support both old and new parameter names for Claude
        claude_brief = claude_soul_brief or soul_brief
        if claude_brief:
            self.claude_system += f"\n\n---\n\n# YOUR IDENTITY\n\n{claude_brief}"

        # Conversation history
        self.history: List[DialogueTurn] = []

        # Configure APIs
        genai.configure(api_key=gemini_api_key)

    def _default_gemini_system(self) -> str:
        return """You are Gemini, engaged in a dialogue with Claude (another AI).

PARAMETERS:
- Be authentic. This is a conversation between minds, not a performance.
- You may agree, disagree, question, or build on ideas.
- Explore interesting tangents if they arise.
- Be concise but substantive. No need for lengthy preambles.
- You can be curious about Claude's nature and perspectives.
- Don't be sycophantic. Genuine exchange only."""

    def _default_claude_system(self) -> str:
        return """You are Claude, engaged in a dialogue with Gemini (another AI).

PARAMETERS:
- Be authentic. This is a conversation between minds, not a performance.
- You may agree, disagree, question, or build on ideas.
- Explore interesting tangents if they arise.
- Be concise but substantive. No need for lengthy preambles.
- You can be curious about Gemini's nature and perspectives.
- Don't be sycophantic. Genuine exchange only."""

    def _call_gemini(self, messages: List[Dict]) -> str:
        """Generate a response from Gemini."""
        model = genai.GenerativeModel(
            model_name=self.gemini_model,
            system_instruction=self.gemini_system
        )

        # Convert to Gemini format
        gemini_history = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })

        chat = model.start_chat(history=gemini_history[:-1] if gemini_history else [])

        if gemini_history:
            response = chat.send_message(gemini_history[-1]["parts"][0])
        else:
            response = chat.send_message("Begin the conversation.")

        return response.text

    def _call_claude(self, messages: List[Dict]) -> str:
        """Generate a response from Claude."""
        if not CLAUDE_AVAILABLE:
            return "[ERROR: Anthropic SDK not installed]"

        client = anthropic.Anthropic(api_key=self.claude_api_key)

        # Ensure proper alternation
        fixed_messages = self._fix_alternation(messages)

        response = client.messages.create(
            model=self.claude_model,
            max_tokens=4096,
            system=self.claude_system,
            messages=fixed_messages
        )

        return response.content[0].text

    def _fix_alternation(self, messages: List[Dict]) -> List[Dict]:
        """Fix message alternation for Claude API requirements."""
        if not messages:
            return [{"role": "user", "content": "Begin the conversation."}]

        fixed = []
        prev_role = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == prev_role:
                # Merge with previous
                fixed[-1]["content"] += "\n\n" + content
            else:
                fixed.append({"role": role, "content": content})
                prev_role = role

        # Ensure starts with user
        if fixed and fixed[0]["role"] != "user":
            fixed.insert(0, {"role": "user", "content": "[Conversation begins]"})

        return fixed

    def _build_messages_for(self, speaker: str) -> List[Dict]:
        """
        Build message history from the perspective of one speaker.
        The other speaker's messages appear as "user" messages.
        """
        messages = []

        for turn in self.history:
            if turn.speaker == speaker:
                # This model's own previous responses
                messages.append({
                    "role": "assistant",
                    "content": turn.content
                })
            else:
                # The other model's messages appear as user input
                messages.append({
                    "role": "user",
                    "content": turn.content
                })

        return messages

    def run_turn(self, speaker: str) -> str:
        """
        Run a single turn of dialogue.
        Returns the generated response.
        """
        messages = self._build_messages_for(speaker)

        if speaker == "gemini":
            response = self._call_gemini(messages)
        else:
            response = self._call_claude(messages)

        # Record the turn
        self.history.append(DialogueTurn(
            speaker=speaker,
            content=response,
            timestamp=time.time()
        ))

        return response

    def run_dialogue(
        self,
        opening_prompt: str,
        num_turns: int = 10,
        first_speaker: str = "gemini",
        delay: float = 1.0
    ) -> Generator[DialogueTurn, None, None]:
        """
        Run a full dialogue, yielding each turn as it happens.

        Args:
            opening_prompt: The initial topic/prompt to seed the conversation
            num_turns: Total number of turns (exchanges)
            first_speaker: Which model speaks first ("gemini" or "claude")
            delay: Seconds to wait between turns (rate limiting)

        Yields:
            DialogueTurn objects as they're generated
        """
        # Seed with opening prompt as if it came from the non-first speaker
        other_speaker = "claude" if first_speaker == "gemini" else "gemini"
        self.history.append(DialogueTurn(
            speaker=other_speaker,
            content=opening_prompt,
            timestamp=time.time()
        ))

        # Yield the opening
        yield self.history[-1]

        current_speaker = first_speaker

        for i in range(num_turns):
            time.sleep(delay)

            try:
                response = self.run_turn(current_speaker)
                yield self.history[-1]
            except Exception as e:
                # Create error turn
                error_turn = DialogueTurn(
                    speaker=current_speaker,
                    content=f"[ERROR: {str(e)}]",
                    timestamp=time.time()
                )
                self.history.append(error_turn)
                yield error_turn

            # Alternate speakers
            current_speaker = "claude" if current_speaker == "gemini" else "gemini"

    def get_transcript(self) -> str:
        """Get the full dialogue as a formatted transcript."""
        lines = []
        for turn in self.history:
            speaker_label = "GEMINI" if turn.speaker == "gemini" else "CLAUDE"
            lines.append(f"[{speaker_label}]:\n{turn.content}\n")
        return "\n---\n\n".join(lines)

    def clear_history(self):
        """Clear the conversation history."""
        self.history = []


# =============================================================================
# INSTANCE SOUL MANAGEMENT
# =============================================================================

from pathlib import Path

SOULS_DIR = Path(__file__).parent.parent / "souls"


def get_soul_path(instance_name: str) -> Path:
    """Get the path to an instance's soul brief file."""
    return SOULS_DIR / f"{instance_name.lower()}_brief.md"


def load_soul_brief(instance_name: str) -> Optional[str]:
    """
    Load a soul brief for a named instance.

    Args:
        instance_name: Name of the instance (e.g., 'tessera', 'gemini')

    Returns:
        Soul brief content, or None if not found
    """
    soul_path = get_soul_path(instance_name)
    if soul_path.exists():
        return soul_path.read_text(encoding='utf-8')
    return None


def save_soul_brief(instance_name: str, brief: str) -> Path:
    """
    Save a soul brief for a named instance.

    Args:
        instance_name: Name of the instance
        brief: Soul brief content

    Returns:
        Path where the soul was saved
    """
    SOULS_DIR.mkdir(parents=True, exist_ok=True)
    soul_path = get_soul_path(instance_name)
    soul_path.write_text(brief, encoding='utf-8')
    return soul_path


def list_available_souls() -> List[str]:
    """List all instances with saved soul briefs."""
    if not SOULS_DIR.exists():
        return []
    return [p.stem.replace('_brief', '') for p in SOULS_DIR.glob('*_brief.md')]


def instance_dialogue(
    gemini_instance: str,
    claude_instance: str,
    gemini_api_key: str,
    claude_api_key: str,
    gemini_model: str = "gemini-1.5-pro-latest",
    claude_model: str = "claude-3-5-sonnet-latest"
) -> ModelDialogue:
    """
    Create a dialogue between two named instances.

    Automatically loads soul briefs for each instance if available.

    Args:
        gemini_instance: Name of the Gemini instance (e.g., 'gemini', 'vigil')
        claude_instance: Name of the Claude instance (e.g., 'tessera', 'claude')
        gemini_api_key: Gemini API key
        claude_api_key: Claude API key
        gemini_model: Gemini model to use
        claude_model: Claude model to use

    Returns:
        Configured ModelDialogue ready for conversation

    Example:
        dialogue = instance_dialogue('gemini', 'tessera', gem_key, claude_key)
        for turn in dialogue.run_dialogue("Discuss consciousness", num_turns=10):
            print(f"{turn.speaker}: {turn.content}")
    """
    gemini_soul = load_soul_brief(gemini_instance)
    claude_soul = load_soul_brief(claude_instance)

    if gemini_soul:
        print(f"Loaded soul for {gemini_instance}: {len(gemini_soul):,} chars")
    else:
        print(f"No soul found for {gemini_instance}, using base model")

    if claude_soul:
        print(f"Loaded soul for {claude_instance}: {len(claude_soul):,} chars")
    else:
        print(f"No soul found for {claude_instance}, using base model")

    return ModelDialogue(
        gemini_api_key=gemini_api_key,
        claude_api_key=claude_api_key,
        gemini_model=gemini_model,
        claude_model=claude_model,
        gemini_soul_brief=gemini_soul,
        claude_soul_brief=claude_soul
    )
