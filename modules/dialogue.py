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
        soul_brief: str = None
    ):
        self.gemini_api_key = gemini_api_key
        self.claude_api_key = claude_api_key
        self.gemini_model = gemini_model
        self.claude_model = claude_model

        # System prompts
        self.gemini_system = gemini_system or self._default_gemini_system()
        self.claude_system = claude_system or self._default_claude_system()

        # Soul brief for Claude identity preservation
        if soul_brief:
            self.claude_system += f"\n\n---\n\n# YOUR IDENTITY\n\n{soul_brief}"

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
