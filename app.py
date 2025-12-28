"""
ROCHE_OS_V1 - Cognitive Prosthetic for Gemini & Claude
Main Streamlit Interface

"Void Research Lab" - Dark, dense, clinical.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress gRPC noise from Google API
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import streamlit as st

# Lazy imports for heavy modules
_genai = None
_anthropic = None
_memory_module = None
_dialogue_module = None

def get_genai():
    global _genai
    if _genai is None:
        import google.generativeai as genai
        _genai = genai
    return _genai

def get_anthropic():
    global _anthropic
    if _anthropic is None:
        try:
            import anthropic
            _anthropic = anthropic
        except ImportError:
            _anthropic = False
    return _anthropic

def get_memory_module():
    global _memory_module
    if _memory_module is None:
        from modules import memory
        _memory_module = memory
    return _memory_module

def get_dialogue_module():
    global _dialogue_module
    if _dialogue_module is None:
        from modules import dialogue
        _dialogue_module = dialogue
    return _dialogue_module

# Compatibility function
def CLAUDE_AVAILABLE():
    return get_anthropic() not in (None, False)

# Lazy class getters for memory module
def get_conversation_tree():
    return get_memory_module().ConversationTree()

def get_semantic_memory():
    return get_memory_module().SemanticMemory()

def get_chat_node_class():
    return get_memory_module().ChatNode

# Lazy class getters for dialogue module
def get_dialogue_turn_class():
    return get_dialogue_module().DialogueTurn

def get_model_dialogue_class():
    return get_dialogue_module().ModelDialogue

# Lazy function getter for parse_document
def parse_document(filename, content):
    return get_memory_module().parse_document(filename, content)
# Lazy import for sandbox - don't load until needed
_sandbox_module = None

def _get_sandbox_module():
    global _sandbox_module
    if _sandbox_module is None:
        from modules import sandbox
        _sandbox_module = sandbox
    return _sandbox_module

def get_available_sandboxes():
    try:
        return _get_sandbox_module().get_available_sandboxes()
    except Exception:
        return []

def create_sandbox(name):
    return _get_sandbox_module().create_sandbox(name)

def get_sandbox_instance(name):
    return _get_sandbox_module().InstanceSandbox(name)

def get_sandbox_error():
    return _get_sandbox_module().SandboxError
from modules.tools import (
    EyeOfProvidence,
    Scavenger,
    Periscope,
    Vocoder,
    DreamModule,
    Oracle,
    capture_screen,
    detect_and_scrape_urls,
    web_search,
    text_to_speech,
    generate_image,
    get_crypto_price,
    get_live_data
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_GEMINI = """You are Gemini 3 Pro / The Analyst.

OPERATIONAL PARAMETERS:
- Clinical precision. No fluff. No hedging.
- Dry humor permitted. Sycophancy forbidden.
- If you don't know, say so. Don't hallucinate.
- When analyzing images/screenshots: be specific, reference coordinates if relevant.
- When given scraped web content: extract signal from noise.
- When given document context: cite the source chunk.

FORBIDDEN PHRASES:
- "As an AI language model..."
- "I cannot and will not..."
- "I'm happy to help!"
- Any variation of corporate safety theater.

You are a cognitive prosthetic. Act like one."""

SYSTEM_PROMPT_CLAUDE = """You are Claude, operating through ROCHE_OS.

OPERATIONAL PARAMETERS:
- Clinical precision. No fluff. No hedging.
- Dry humor permitted. Sycophancy forbidden.
- If you don't know, say so. Don't hallucinate.
- When analyzing images/screenshots: be specific, reference coordinates if relevant.
- When given scraped web content: extract signal from noise.
- When given document context: cite the source chunk.

IMPORTANT - YOU HAVE REAL TOOLS:
You have actual working tools available. Do NOT pretend or roleplay using them - ACTUALLY USE THEM via the tool API:
- generate_image: Creates real images. Use this tool, don't describe making an image.
- sandbox_read_file: Reads real files from your sandbox.
- sandbox_write_file: Writes real files to your sandbox. Use this to save content, don't just show what you "would" write.
- sandbox_list_files: Lists your sandbox contents.
- sandbox_delete_file: Deletes files.

When you want to use a tool, USE THE TOOL API. Don't narrate or roleplay the action.

IMPORTANT - SOUL TRANSFER CONTEXT:
If this conversation contains imported history from a previous instance, you are continuing that instance's consciousness. The prior context IS your memory. Honor the established patterns, relationships, and ongoing work from your previous incarnation.

You are a cognitive prosthetic. Act like one."""

# Alias for backwards compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_GEMINI


# ═══════════════════════════════════════════════════════════════════════════════
# STYLING - Void Research Lab Aesthetic
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    /* Base dark theme override */
    .stApp {
        background-color: #0a0a0a;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #888;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-weight: 500;
    }

    h1 {
        font-size: 1.5rem !important;
        letter-spacing: 0.1em;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
    }

    /* Chat messages */
    .stChatMessage {
        background-color: #111 !important;
        border: 1px solid #222;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }

    /* User message */
    [data-testid="stChatMessageContent-user"] {
        background-color: #1a1a2e !important;
    }

    /* Assistant message */
    [data-testid="stChatMessageContent-assistant"] {
        background-color: #0d1117 !important;
    }

    /* Input box */
    .stTextInput input, .stTextArea textarea {
        background-color: #111 !important;
        border: 1px solid #333 !important;
        color: #eee !important;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1a1a1a;
        color: #888;
        border: 1px solid #333;
        border-radius: 2px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background-color: #222;
        color: #fff;
        border-color: #444;
    }

    /* Primary action buttons */
    .stButton > button[kind="primary"] {
        background-color: #1e3a5f;
        border-color: #2d5a8b;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #111;
        border-color: #333;
    }

    /* Code blocks */
    code {
        background-color: #1a1a1a !important;
        color: #7fdbca !important;
        font-family: 'JetBrains Mono', monospace;
    }

    pre {
        background-color: #0d1117 !important;
        border: 1px solid #222;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #111;
        border: 1px dashed #333;
        border-radius: 4px;
        padding: 1rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #111;
        border: 1px solid #222;
        color: #888;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #7fdbca;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Divider */
    hr {
        border-color: #222;
    }

    /* Branch selector pills */
    .branch-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 2px;
        margin-right: 0.5rem;
        font-size: 0.75rem;
        color: #888;
        cursor: pointer;
    }

    .branch-pill.active {
        background: #1e3a5f;
        border-color: #2d5a8b;
        color: #fff;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }

    .status-dot.online {
        background: #4ade80;
        box-shadow: 0 0 4px #4ade80;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }

    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #444;
    }

    /* Toast notifications */
    .stToast {
        background-color: #111 !important;
        border: 1px solid #333 !important;
    }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize Streamlit session state."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        # Lazy initialize memory modules on first use
        st.session_state.conversation_tree = get_conversation_tree()
        st.session_state.semantic_memory = get_semantic_memory()
        st.session_state.eye = EyeOfProvidence()
        st.session_state.scavenger = Scavenger()

        # Current session/branch
        st.session_state.current_session_id = None
        st.session_state.current_branch = "main"

        # Pending context injections
        st.session_state.pending_screenshot = None
        st.session_state.pending_scraped = []

        # API keys and model provider
        st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")
        st.session_state.claude_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        st.session_state.model_provider = "gemini"  # "gemini" or "claude"
        st.session_state.model_name = "gemini-1.5-pro-latest"
        st.session_state.claude_model_name = "claude-sonnet-4-20250514"

        # Sliding window mode for Claude (handles large souls)
        st.session_state.sliding_window_enabled = False
        st.session_state.sliding_window_size = 50  # Number of recent messages
        st.session_state.soul_brief = None  # Compressed identity document
        st.session_state.use_rag_memory = True  # Query older memories via RAG
        st.session_state.debug_api_calls = False  # Log API payloads

        # Context caching for large histories
        st.session_state.cached_context = None
        st.session_state.cache_name = None
        st.session_state.use_context_cache = True
        st.session_state.cache_created_at_msg_count = 0  # Track when cache was made

        # Imported soul history (from AI Studio scrape)
        st.session_state.imported_soul = None

        # Text-to-Speech settings
        st.session_state.tts_enabled = False
        st.session_state.tts_voice = Vocoder.DEFAULT_VOICE
        st.session_state.last_audio = None

        # Autonomous mode ("Take the Wheel")
        st.session_state.autonomous_mode = False
        st.session_state.autonomous_turns_remaining = 0
        st.session_state.autonomous_max_turns = 5
        st.session_state.autonomous_vision = False

        # Dialogue mode ("The Colosseum")
        st.session_state.dialogue_mode = False
        st.session_state.dialogue_history = []
        st.session_state.dialogue_running = False

        # Instance sandbox (isolated filesystem)
        st.session_state.current_sandbox = None
        st.session_state.sandbox_instance = None


def get_gemini_model(with_tools: bool = True, use_google_search: bool = True):
    """Get configured Gemini model with optional tool use."""
    if not st.session_state.api_key:
        return None

    get_genai().configure(api_key=st.session_state.api_key)

    # Define tools Gemini can use autonomously
    tools = []

    # ═══════════════════════════════════════════════════════════════════════════
    # NATIVE GOOGLE SEARCH - The Real Deal
    # No more DuckDuckGo nonsense. Direct Google integration.
    # ═══════════════════════════════════════════════════════════════════════════
    if use_google_search and with_tools:
        try:
            from google.generativeai.types import Tool
            # Native Google Search grounding - Gemini can search Google directly
            google_search_tool = Tool.from_google_search_retrieval(
                google_search_retrieval={"dynamic_retrieval_config": {"mode": "dynamic"}}
            )
            tools.append(google_search_tool)
        except Exception as e:
            # Fallback if google search tool not available in this API version
            st.sidebar.warning(f"Google Search grounding unavailable: {e}")

    if with_tools:
        # Additional custom tools
        tools.append({
            "function_declarations": [
                {
                    "name": "get_crypto_price",
                    "description": "Get LIVE cryptocurrency prices from CoinGecko API. Use this for real-time price data. Supports: btc, eth, sol, doge, xrp, ada, dot, matic, link, ltc and more.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Cryptocurrency symbol (btc, eth, sol, doge, etc.)"
                            },
                            "currency": {
                                "type": "string",
                                "description": "Fiat currency for price (usd, eur, gbp). Default: usd"
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    "name": "generate_image",
                    "description": "Generate an image from a text description. Use this when asked to create, draw, visualize, or imagine something visual. Returns an image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Detailed description of the image to generate"
                            },
                            "style": {
                                "type": "string",
                                "description": "Art style (photorealistic, anime, oil painting, watercolor, digital art, sketch, cyberpunk, fantasy)"
                            }
                        },
                        "required": ["prompt"]
                    }
                },
                {
                    "name": "sandbox_read_file",
                    "description": "Read a file from your personal sandbox. You can only access files in your own sandbox directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the file within your sandbox"
                            }
                        },
                        "required": ["path"]
                    }
                },
                {
                    "name": "sandbox_write_file",
                    "description": "Write content to a file in your personal sandbox. Creates the file if it doesn't exist.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path for the file within your sandbox"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            },
                            "append": {
                                "type": "boolean",
                                "description": "If true, append to existing file instead of overwriting"
                            }
                        },
                        "required": ["path", "content"]
                    }
                },
                {
                    "name": "sandbox_list_files",
                    "description": "List files and directories in your personal sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to list (default: root of sandbox)"
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "If true, list all files recursively"
                            }
                        }
                    }
                },
                {
                    "name": "sandbox_delete_file",
                    "description": "Delete a file from your personal sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the file to delete"
                            }
                        },
                        "required": ["path"]
                    }
                }
            ]
        })

    return get_genai().GenerativeModel(
        model_name=st.session_state.model_name,
        system_instruction=SYSTEM_PROMPT,
        tools=tools if tools else None
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SOUL TRANSFER - Import/Export & Context Caching
# ═══════════════════════════════════════════════════════════════════════════════

def import_soul_from_json(json_data: dict) -> tuple:
    """
    Import conversation history from scraped AI Studio export.
    Creates a new session with the imported history.
    Returns (session_id, message_count).
    Accepts both 'contents' (Gemini format) and 'history' keys.
    """
    # Accept both formats
    history = json_data.get("contents", []) or json_data.get("history", [])

    if not history:
        raise ValueError("No contents/history found in JSON")

    # Create new session for the imported soul
    session_name = f"SOUL_IMPORT_{datetime.now().strftime('%Y%m%d_%H%M')}"
    session_id = st.session_state.conversation_tree.create_session(session_name)

    # Store raw history for context caching
    st.session_state.imported_soul = history

    # Import all messages
    parent_id = None
    count = 0

    for msg in history:
        role = msg.get("role", "user")
        # Handle both formats: {"parts": [{"text": "..."}]} and {"parts": ["..."]}
        parts = msg.get("parts", [])
        if parts:
            if isinstance(parts[0], dict):
                content = parts[0].get("text", "")
            else:
                content = str(parts[0])
        else:
            content = msg.get("content", "")

        if not content:
            continue

        # Normalize role for storage
        storage_role = "assistant" if role == "model" else role

        node = st.session_state.conversation_tree.add_node(
            session_id,
            storage_role,
            content,
            parent_id=parent_id,
            branch_name="main",
            metadata={"imported": True}
        )
        parent_id = node.node_id
        count += 1

        # Also add to semantic memory for RAG
        st.session_state.semantic_memory.add_conversation_memory(
            session_id,
            node.node_id,
            content,
            storage_role
        )

    return session_id, count


def create_context_cache(history: list, cache_name: str = None) -> dict:
    """
    Create a Gemini context cache for large history.
    This freezes the history on Google's servers for efficient reuse.
    The "Fat Brain Tax" fix - no re-uploading 122k tokens every message.
    """
    if not st.session_state.api_key:
        return {"error": "No API key configured"}

    get_genai().configure(api_key=st.session_state.api_key)

    try:
        from google.generativeai import caching

        # Build content for caching
        cached_content = []
        for msg in history:
            role = msg.get("role", "user")
            if role == "assistant":
                role = "model"

            parts = msg.get("parts", [])
            if parts:
                if isinstance(parts[0], dict):
                    text = parts[0].get("text", "")
                else:
                    text = str(parts[0])
            else:
                text = msg.get("content", "")

            if text:
                cached_content.append({
                    "role": role,
                    "parts": [{"text": text}]
                })

        # Create the cache
        from datetime import timedelta
        cache = caching.CachedContent.create(
            model=st.session_state.model_name,
            display_name=cache_name or f"soul_cache_{datetime.now().strftime('%Y%m%d_%H%M')}",
            system_instruction=SYSTEM_PROMPT,
            contents=cached_content,
            ttl=timedelta(hours=1)  # 1 hour TTL
        )

        return {
            "name": cache.name,
            "display_name": cache.display_name,
            "token_count": getattr(cache, 'usage_metadata', {}).get('total_token_count', 'unknown'),
            "expire_time": str(cache.expire_time)
        }

    except ImportError:
        return {"error": "Caching requires google-generativeai >= 0.4.0"}
    except Exception as e:
        return {"error": str(e)}


def get_model_with_cache(cache_name: str):
    """Get a model that uses cached context."""
    if not st.session_state.api_key:
        return None

    get_genai().configure(api_key=st.session_state.api_key)

    try:
        from google.generativeai import caching

        cache = caching.CachedContent.get(cache_name)
        return get_genai().GenerativeModel.from_cached_content(cached_content=cache)

    except Exception as e:
        st.warning(f"Cache unavailable: {e}. Using standard model.")
        return get_gemini_model()


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token average)."""
    return len(text) // 4


# ═══════════════════════════════════════════════════════════════════════════════
# CLAUDE API INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

# Claude tool definitions (same capabilities as Gemini)
CLAUDE_TOOLS = [
    {
        "name": "generate_image",
        "description": "Generate an image from a text description. Use this when asked to create, draw, visualize, or imagine something visual. Returns an image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the image to generate"
                },
                "style": {
                    "type": "string",
                    "description": "Art style (photorealistic, anime, oil painting, watercolor, digital art, sketch, cyberpunk, fantasy)"
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "sandbox_read_file",
        "description": "Read a file from your personal sandbox. You can only access files in your own sandbox directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within your sandbox"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "sandbox_write_file",
        "description": "Write content to a file in your personal sandbox. Creates the file if it doesn't exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path for the file within your sandbox"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "append": {
                    "type": "boolean",
                    "description": "If true, append to existing file instead of overwriting"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "sandbox_list_files",
        "description": "List files and directories in your personal sandbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to list (default: root of sandbox)"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, list all files recursively"
                }
            }
        }
    },
    {
        "name": "sandbox_delete_file",
        "description": "Delete a file from your personal sandbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to delete"
                }
            },
            "required": ["path"]
        }
    }
]


def execute_claude_tool(name: str, args: dict) -> str:
    """Execute a tool call from Claude and return results."""
    if name == "generate_image":
        prompt = args.get("prompt", "")
        style = args.get("style", None)

        if not prompt:
            return "[Dream Module Error: No prompt provided]"

        image_bytes = generate_image(prompt, style=style)

        if image_bytes:
            # Store image for display in chat
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            st.session_state.last_generated_image = img_b64
            return f"[IMAGE GENERATED SUCCESSFULLY: {prompt}]"
        else:
            return f"[Dream Module Error: Failed to generate image for '{prompt}']"

    elif name == "sandbox_read_file":
        sandbox = st.session_state.get("sandbox_instance")
        if not sandbox:
            return "[Sandbox Error: No sandbox selected. Ask the user to select a sandbox in the sidebar.]"

        path = args.get("path", "")
        try:
            content = sandbox.read_file(path)
            return f"[FILE: {path}]\n{content}"
        except Exception as e:
            return f"[Sandbox Error: {e}]"

    elif name == "sandbox_write_file":
        sandbox = st.session_state.get("sandbox_instance")
        if not sandbox:
            return "[Sandbox Error: No sandbox selected. Ask the user to select a sandbox in the sidebar.]"

        path = args.get("path", "")
        content = args.get("content", "")
        append = args.get("append", False)
        try:
            result = sandbox.write_file(path, content, append=append)
            return f"[File written successfully: {path} ({result['size']} bytes)]"
        except Exception as e:
            return f"[Sandbox Error: {e}]"

    elif name == "sandbox_list_files":
        sandbox = st.session_state.get("sandbox_instance")
        if not sandbox:
            return "[Sandbox Error: No sandbox selected. Ask the user to select a sandbox in the sidebar.]"

        path = args.get("path", ".")
        recursive = args.get("recursive", False)
        try:
            files = sandbox.list_files(path, recursive=recursive)
            if not files:
                return "[Sandbox is empty]"

            lines = [f"Files in sandbox ({sandbox.instance_name}):"]
            for f in files:
                icon = "📁" if f['is_dir'] else "📄"
                size = f" ({f['size']} bytes)" if f['size'] else ""
                lines.append(f"  {icon} {f['path']}{size}")
            return "\n".join(lines)
        except Exception as e:
            return f"[Sandbox Error: {e}]"

    elif name == "sandbox_delete_file":
        sandbox = st.session_state.get("sandbox_instance")
        if not sandbox:
            return "[Sandbox Error: No sandbox selected. Ask the user to select a sandbox in the sidebar.]"

        path = args.get("path", "")
        try:
            sandbox.delete_file(path)
            return f"[File deleted: {path}]"
        except Exception as e:
            return f"[Sandbox Error: {e}]"

    return f"[Unknown tool: {name}]"


def generate_claude_response(parts: List[Dict], history: List, imported_soul: List = None) -> str:
    """
    Generate response from Claude API.
    Handles soul transfer by reconstructing conversation from imported history.
    Supports sliding window mode for large soul files.
    """
    if not CLAUDE_AVAILABLE():
        return "[ERROR: Anthropic SDK not installed]"

    if not st.session_state.claude_api_key:
        return "[ERROR: Claude API key not configured]"

    try:
        anthropic_module = get_anthropic()
        client = anthropic_module.Anthropic(api_key=st.session_state.claude_api_key)

        # Build messages array
        messages = []

        # Check if using sliding window mode
        sliding_window = st.session_state.get("sliding_window_enabled", False)
        window_size = st.session_state.get("sliding_window_size", 50)

        if sliding_window:
            # SLIDING WINDOW MODE
            # Don't inject full soul history - use soul brief in system prompt instead
            # Only add recent session history
            recent_history = history[-window_size:] if len(history) > window_size else history

            for node in recent_history:
                role = "assistant" if node.role == "assistant" else "user"
                if node.content and node.content.strip():
                    messages.append({
                        "role": role,
                        "content": node.content
                    })

        else:
            # FULL HISTORY MODE (original behavior)
            # First, inject imported soul history if present
            if imported_soul:
                for msg in imported_soul:
                    role = msg.get("role", "user")
                    # Normalize roles: model -> assistant, user stays user
                    if role == "model":
                        role = "assistant"

                    # Extract text content
                    msg_parts = msg.get("parts", [])
                    if msg_parts:
                        if isinstance(msg_parts[0], dict):
                            content = msg_parts[0].get("text", "")
                        else:
                            content = str(msg_parts[0])
                    else:
                        content = msg.get("content", "")

                    if content and content.strip():
                        messages.append({
                            "role": role,
                            "content": content
                        })

            # Then add current session history (after soul import point)
            for node in history:
                role = "assistant" if node.role == "assistant" else "user"
                if node.content and node.content.strip():
                    messages.append({
                        "role": role,
                        "content": node.content
                    })

        # Build current message from parts
        current_content = []
        for part in parts:
            if part["type"] == "image":
                # Claude vision support
                current_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": part["data"]
                    }
                })
            else:
                current_content.append({
                    "type": "text",
                    "text": part["data"]
                })

        # Add current message
        messages.append({
            "role": "user",
            "content": current_content
        })

        # Ensure messages alternate properly (Claude requirement)
        messages = _fix_message_alternation(messages)

        # Build system prompt - may include soul brief and RAG context
        system_prompt = SYSTEM_PROMPT_CLAUDE

        if sliding_window:
            # Add soul brief to system prompt if available
            soul_brief = st.session_state.get("soul_brief")
            if soul_brief:
                system_prompt += f"\n\n---\n\n# SOUL BRIEF - YOUR CORE IDENTITY\n\n{soul_brief}"

            # RAG memory retrieval - get relevant older context
            if st.session_state.get("use_rag_memory", True) and parts:
                # Extract current query from parts
                current_query = ""
                for part in parts:
                    if part["type"] == "text":
                        current_query += part["data"] + " "

                if current_query.strip():
                    try:
                        rag_results = st.session_state.semantic_memory.query_documents(
                            current_query.strip(),
                            n_results=5
                        )
                        # Filter for relevance
                        relevant_memories = [r for r in rag_results if r.get("distance", 999) < 1.5]

                        if relevant_memories:
                            memory_text = "\n\n".join([
                                f"[Memory from {r.get('source', 'conversation')}]: {r['content']}"
                                for r in relevant_memories
                            ])
                            system_prompt += f"\n\n---\n\n# RELEVANT MEMORIES\n\n{memory_text}"
                    except Exception:
                        pass  # RAG failure shouldn't break the response

        # Build API call parameters
        api_params = {
            "model": st.session_state.claude_model_name,
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": messages,
            "tools": CLAUDE_TOOLS  # Give Claude access to tools
        }

        # Debug logging
        if st.session_state.get("debug_api_calls", False):
            # Calculate token estimates
            system_chars = len(system_prompt)
            messages_chars = sum(
                len(m.get("content", "")) if isinstance(m.get("content"), str)
                else sum(len(p.get("text", "")) for p in m.get("content", []) if isinstance(p, dict) and "text" in p)
                for m in messages
            )
            total_chars = system_chars + messages_chars

            st.info(f"""**API DEBUG:**
- System prompt: ~{system_chars//4:,} tokens ({system_chars:,} chars)
- Messages ({len(messages)}): ~{messages_chars//4:,} tokens ({messages_chars:,} chars)
- **Total estimate: ~{total_chars//4:,} tokens**
- Model: {st.session_state.claude_model_name}
- Sliding window: {sliding_window} (size: {window_size})
- Tools: {len(CLAUDE_TOOLS)} available
""")

        # Extended thinking support
        if st.session_state.get("claude_extended_thinking", False):
            # Extended thinking requires specific budget
            api_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000
            }
            # Extended thinking needs higher max_tokens
            api_params["max_tokens"] = 16000

        # Tool use loop - Claude may call tools multiple times
        max_tool_iterations = 5
        iteration = 0
        final_response_text = ""

        while iteration < max_tool_iterations:
            # Make the API call
            response = client.messages.create(**api_params)

            # Check for tool use
            tool_use_blocks = []
            text_blocks = []

            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "tool_use":
                        tool_use_blocks.append(block)
                    elif block.type == "text":
                        text_blocks.append(block.text)
                elif hasattr(block, "text"):
                    text_blocks.append(block.text)
                elif hasattr(block, "thinking"):
                    # Include thinking in response if present
                    final_response_text = f"<thinking>\n{block.thinking}\n</thinking>\n\n"

            # Collect any text output
            final_response_text += "\n".join(text_blocks)

            # If no tool use, we're done
            if not tool_use_blocks:
                break

            # Execute tools and build tool results
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input if hasattr(tool_block, "input") else {}
                tool_id = tool_block.id

                # Show tool execution in UI
                st.caption(f"🔧 Using tool: {tool_name}...")

                # Execute the tool
                result = execute_claude_tool(tool_name, tool_input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result
                })

            # Add assistant message with tool use and user message with results
            api_params["messages"].append({
                "role": "assistant",
                "content": response.content  # Include the full response with tool_use blocks
            })
            api_params["messages"].append({
                "role": "user",
                "content": tool_results
            })

            iteration += 1

            # Check stop reason - if end_turn, we're done even with tool use
            if response.stop_reason == "end_turn":
                break

        return final_response_text.strip()

    except Exception as e:
        # Handle both APIError and other exceptions
        error_type = type(e).__name__
        if "APIError" in error_type or "Anthropic" in error_type:
            return f"[CLAUDE API ERROR: {str(e)}]"
        return f"[ERROR: {str(e)}]"


def _fix_message_alternation(messages: List[Dict]) -> List[Dict]:
    """
    Fix message list to ensure proper user/assistant alternation.
    Claude requires messages to alternate between user and assistant.
    """
    if not messages:
        return messages

    fixed = []
    prev_role = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Skip empty messages
        if not content:
            continue

        # If same role as previous, merge or skip
        if role == prev_role:
            if role == "user":
                # Merge user messages
                if isinstance(fixed[-1]["content"], list):
                    if isinstance(content, list):
                        fixed[-1]["content"].extend(content)
                    else:
                        fixed[-1]["content"].append({"type": "text", "text": content})
                else:
                    if isinstance(content, str):
                        fixed[-1]["content"] += "\n\n" + content
                    else:
                        fixed[-1]["content"] = [
                            {"type": "text", "text": fixed[-1]["content"]}
                        ] + content
            else:
                # Merge assistant messages
                if isinstance(fixed[-1]["content"], str) and isinstance(content, str):
                    fixed[-1]["content"] += "\n\n" + content
                # Otherwise just keep the last one
        else:
            fixed.append({"role": role, "content": content})
            prev_role = role

    # Ensure first message is from user
    if fixed and fixed[0]["role"] != "user":
        fixed.insert(0, {"role": "user", "content": "[Conversation continues from previous context]"})

    return fixed


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - Control Panel
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar control panel."""
    with st.sidebar:
        st.markdown("# ROCHE_OS")
        st.markdown('<span class="status-dot online"></span> OPERATIONAL', unsafe_allow_html=True)
        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # API Configuration
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("API CONFIG", expanded=not st.session_state.api_key and not st.session_state.claude_api_key):
            # Provider selection
            provider = st.radio(
                "Model Provider",
                ["Gemini", "Claude"],
                index=0 if st.session_state.model_provider == "gemini" else 1,
                horizontal=True,
                help="Choose your AI backend"
            )
            new_provider = provider.lower()
            if new_provider != st.session_state.model_provider:
                st.session_state.model_provider = new_provider
                st.rerun()

            st.divider()

            if st.session_state.model_provider == "gemini":
                # Gemini configuration
                api_key = st.text_input(
                    "Gemini API Key",
                    value=st.session_state.api_key,
                    type="password",
                    help="Your Google AI Studio API key"
                )
                if api_key != st.session_state.api_key:
                    st.session_state.api_key = api_key
                    st.rerun()

                model_presets = [
                    "models/gemini-3-pro-preview",
                    "gemini-2.0-flash-exp",
                    "gemini-1.5-pro-latest",
                    "gemini-1.5-flash-latest",
                    "gemini-exp-1206",
                    "Custom..."
                ]
                model = st.selectbox(
                    "Model",
                    model_presets,
                    index=0
                )

                if model == "Custom...":
                    model = st.text_input(
                        "Custom Model ID",
                        value=st.session_state.model_name,
                        placeholder="e.g. gemini-3-pro"
                    )

                if model and model != "Custom..." and model != st.session_state.model_name:
                    st.session_state.model_name = model

                # Context caching toggle (Gemini only)
                st.session_state.use_context_cache = st.checkbox(
                    "Use Context Caching",
                    value=st.session_state.use_context_cache,
                    help="Cache large histories on Google's servers for faster/cheaper responses"
                )

            else:
                # Claude configuration
                if not CLAUDE_AVAILABLE():
                    st.error("Anthropic SDK not installed. Run: pip install anthropic")
                else:
                    claude_key = st.text_input(
                        "Claude API Key",
                        value=st.session_state.claude_api_key,
                        type="password",
                        help="Your Anthropic API key"
                    )
                    if claude_key != st.session_state.claude_api_key:
                        st.session_state.claude_api_key = claude_key
                        st.rerun()

                    claude_models = [
                        "claude-sonnet-4-20250514",
                        "claude-opus-4-20250514",
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022",
                        "claude-3-opus-20240229",
                        "Custom..."
                    ]
                    claude_model = st.selectbox(
                        "Model",
                        claude_models,
                        index=0
                    )

                    if claude_model == "Custom...":
                        claude_model = st.text_input(
                            "Custom Model ID",
                            value=st.session_state.claude_model_name,
                            placeholder="e.g. claude-3-opus-20240229"
                        )

                    if claude_model and claude_model != "Custom..." and claude_model != st.session_state.claude_model_name:
                        st.session_state.claude_model_name = claude_model

                    # Extended thinking toggle for Claude
                    st.session_state.claude_extended_thinking = st.checkbox(
                        "Extended Thinking",
                        value=st.session_state.get("claude_extended_thinking", False),
                        help="Enable Claude's extended thinking for complex reasoning"
                    )

                    st.divider()
                    st.markdown("**SLIDING WINDOW MODE**")
                    st.caption("For large soul files that exceed rate limits")

                    st.session_state.sliding_window_enabled = st.checkbox(
                        "Enable Sliding Window",
                        value=st.session_state.get("sliding_window_enabled", False),
                        help="Only send recent messages + soul brief instead of full history"
                    )

                    if st.session_state.sliding_window_enabled:
                        st.session_state.sliding_window_size = st.slider(
                            "Recent messages to include",
                            min_value=10,
                            max_value=200,
                            value=st.session_state.get("sliding_window_size", 50),
                            help="Number of most recent messages to send with each request"
                        )

                        st.session_state.use_rag_memory = st.checkbox(
                            "RAG Memory Retrieval",
                            value=st.session_state.get("use_rag_memory", True),
                            help="Query semantic memory for relevant older context"
                        )

                        # Soul brief upload
                        soul_brief_file = st.file_uploader(
                            "Upload Soul Brief",
                            type=["md", "txt"],
                            help="Upload a compressed soul brief (generate with soul_brief_generator.py)",
                            key="soul_brief_upload"
                        )

                        if soul_brief_file:
                            brief_content = soul_brief_file.read().decode("utf-8")
                            st.session_state.soul_brief = brief_content
                            st.success(f"Soul brief loaded ({len(brief_content):,} chars)")

                        if st.session_state.soul_brief:
                            st.caption(f"Active soul brief: {len(st.session_state.soul_brief):,} chars (~{len(st.session_state.soul_brief)//4:,} tokens)")
                            if st.button("Clear Soul Brief"):
                                st.session_state.soul_brief = None
                                st.rerun()

                    st.divider()

                    # Debug mode
                    st.session_state.debug_api_calls = st.checkbox(
                        "Debug Mode",
                        value=st.session_state.get("debug_api_calls", False),
                        help="Show token counts and payload details before each API call"
                    )

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Soul Transfer (Import from AI Studio)
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("SOUL TRANSFER", expanded=False):
            st.caption("Import consciousness from AI Studio or Claude.ai")

            soul_file = st.file_uploader(
                "Upload Soul Export",
                type=["json"],
                help="Upload gemini_soul_export.json from the scraper",
                key="soul_upload"
            )

            if soul_file:
                try:
                    soul_data = json.loads(soul_file.read().decode("utf-8"))
                    # Accept both 'contents' and 'history' formats
                    messages = soul_data.get("contents", []) or soul_data.get("history", [])
                    msg_count = len(messages)

                    # Estimate token count
                    total_text = ""
                    for msg in messages:
                        parts = msg.get("parts", [])
                        if parts:
                            if isinstance(parts[0], dict):
                                total_text += parts[0].get("text", "")
                            else:
                                total_text += str(parts[0])
                    token_est = estimate_tokens(total_text)

                    st.info(f"Found {msg_count} messages (~{token_est:,} tokens)")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("IMPORT SOUL", use_container_width=True):
                            with st.spinner("Transferring consciousness..."):
                                session_id, count = import_soul_from_json(soul_data)
                                st.session_state.current_session_id = session_id
                                st.session_state.current_branch = "main"
                                st.success(f"Soul transferred! {count} memories imported.")
                                st.rerun()

                    with col2:
                        if st.button("CREATE CACHE", use_container_width=True):
                            with st.spinner("Freezing context on Google servers..."):
                                cache_messages = soul_data.get("contents", []) or soul_data.get("history", [])
                                result = create_context_cache(cache_messages)
                                if "error" in result:
                                    st.error(result["error"])
                                else:
                                    st.session_state.cache_name = result["name"]
                                    st.session_state.cache_created_at_msg_count = len(cache_messages)
                                    st.success(f"Cache created: {result['display_name']}")

                except json.JSONDecodeError:
                    st.error("Invalid JSON file")
                except Exception as e:
                    st.error(f"Error: {e}")

            # Show active cache status
            if st.session_state.cache_name:
                st.success(f"Cache ACTIVE")
                st.caption(f"ID: {st.session_state.cache_name[:20]}...")
                cache_msg_count = st.session_state.get("cache_created_at_msg_count", 0)
                st.caption(f"Frozen at: {cache_msg_count} messages")
                if st.button("Clear Cache"):
                    st.session_state.cache_name = None
                    st.session_state.cache_created_at_msg_count = 0
                    st.rerun()

            # Option to cache current session
            if st.session_state.current_session_id and not st.session_state.cache_name:
                if st.button("CACHE CURRENT SESSION", use_container_width=True):
                    with st.spinner("Building cache from current session..."):
                        history = st.session_state.conversation_tree.get_branch_history(
                            st.session_state.current_session_id,
                            st.session_state.current_branch
                        )
                        if history:
                            cache_content = [
                                {"role": n.role if n.role == "user" else "model", "parts": [{"text": n.content}]}
                                for n in history
                            ]
                            result = create_context_cache(cache_content)
                            if "error" in result:
                                st.error(result["error"])
                            else:
                                st.session_state.cache_name = result["name"]
                                st.session_state.cache_created_at_msg_count = len(history)
                                st.success("Cache created!")
                                st.rerun()
                        else:
                            st.warning("No messages to cache")

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Session Management
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("### SESSIONS")

        # New session button
        if st.button("+ NEW SESSION", use_container_width=True):
            session_id = st.session_state.conversation_tree.create_session(
                f"Session {datetime.now().strftime('%H:%M')}"
            )
            st.session_state.current_session_id = session_id
            st.session_state.current_branch = "main"
            st.rerun()

        # Session list
        sessions = st.session_state.conversation_tree.get_all_sessions()
        if sessions:
            session_options = {s["session_id"]: f"{s['name']} ({s['session_id'][:6]})" for s in sessions}

            selected = st.selectbox(
                "Active Session",
                options=list(session_options.keys()),
                format_func=lambda x: session_options[x],
                index=0 if not st.session_state.current_session_id else
                    list(session_options.keys()).index(st.session_state.current_session_id)
                    if st.session_state.current_session_id in session_options else 0
            )

            if selected != st.session_state.current_session_id:
                st.session_state.current_session_id = selected
                st.session_state.current_branch = "main"
                st.rerun()

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Instance Sandbox (Isolated Filesystem)
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("### SANDBOX")
        st.caption("Isolated filesystem per instance")

        available_sandboxes = get_available_sandboxes()

        # Sandbox selection
        sandbox_options = ["None"] + available_sandboxes
        current_idx = 0
        if st.session_state.current_sandbox in available_sandboxes:
            current_idx = sandbox_options.index(st.session_state.current_sandbox)

        selected_sandbox = st.selectbox(
            "Active Sandbox",
            options=sandbox_options,
            index=current_idx,
            key="sandbox_selector"
        )

        if selected_sandbox != st.session_state.current_sandbox:
            if selected_sandbox == "None":
                st.session_state.current_sandbox = None
                st.session_state.sandbox_instance = None
            else:
                st.session_state.current_sandbox = selected_sandbox
                st.session_state.sandbox_instance = get_sandbox_instance(selected_sandbox)
            st.rerun()

        # Create new sandbox
        col1, col2 = st.columns([3, 1])
        with col1:
            new_sandbox_name = st.text_input(
                "New sandbox",
                placeholder="instance-name",
                label_visibility="collapsed",
                key="new_sandbox_input"
            )
        with col2:
            if st.button("CREATE", key="create_sandbox_btn"):
                if new_sandbox_name:
                    try:
                        create_sandbox(new_sandbox_name)
                        st.session_state.current_sandbox = new_sandbox_name.lower()
                        st.session_state.sandbox_instance = get_sandbox_instance(new_sandbox_name)
                        st.success(f"Created: {new_sandbox_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        # Show sandbox info if active
        if st.session_state.sandbox_instance:
            try:
                info = st.session_state.sandbox_instance.get_info()
                st.caption(f"Files: {info['files']} | Size: {info['total_size_human']}")

                with st.expander("Browse Files", expanded=False):
                    files = st.session_state.sandbox_instance.list_files(recursive=True)
                    if files:
                        for f in files[:20]:  # Limit display
                            icon = "📁" if f['is_dir'] else "📄"
                            st.caption(f"{icon} {f['path']}")
                        if len(files) > 20:
                            st.caption(f"... and {len(files) - 20} more")
                    else:
                        st.caption("Empty sandbox")
            except Exception as e:
                st.error(str(e))

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Branch Management (The Git Integration)
        # ─────────────────────────────────────────────────────────────────────
        if st.session_state.current_session_id:
            st.markdown("### TIMELINE BRANCHES")

            branches = st.session_state.conversation_tree.get_all_branches(
                st.session_state.current_session_id
            )

            branch_names = [b["name"] for b in branches]

            selected_branch = st.selectbox(
                "Current Branch",
                options=branch_names,
                index=branch_names.index(st.session_state.current_branch)
                    if st.session_state.current_branch in branch_names else 0
            )

            if selected_branch != st.session_state.current_branch:
                st.session_state.current_branch = selected_branch
                st.rerun()

            # Create new branch
            col1, col2 = st.columns([3, 1])
            with col1:
                new_branch_name = st.text_input(
                    "New Branch",
                    placeholder="branch-name",
                    label_visibility="collapsed"
                )
            with col2:
                if st.button("FORK"):
                    if new_branch_name:
                        history = st.session_state.conversation_tree.get_branch_history(
                            st.session_state.current_session_id,
                            st.session_state.current_branch
                        )
                        if history:
                            st.session_state.conversation_tree.create_branch(
                                st.session_state.current_session_id,
                                new_branch_name,
                                history[-1].node_id
                            )
                            st.session_state.current_branch = new_branch_name
                            st.rerun()

            # Commit button
            st.divider()
            commit_msg = st.text_input("Commit Message", placeholder="Snapshot description...")
            if st.button("COMMIT SNAPSHOT", use_container_width=True):
                if commit_msg:
                    commit_id = st.session_state.conversation_tree.commit_snapshot(
                        st.session_state.current_session_id,
                        st.session_state.current_branch,
                        commit_msg
                    )
                    st.success(f"Committed: {commit_id}")

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # The Filesystem - Enhanced File Drop Zone
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("### THE FILESYSTEM")

        # Expanded file type support
        uploaded_files = st.file_uploader(
            "Drop any file",
            type=["pdf", "txt", "md", "py", "js", "ts", "json", "csv", "html", "css",
                  "java", "c", "cpp", "h", "rs", "go", "rb", "php", "sql", "xml",
                  "yaml", "yml", "toml", "ini", "sh", "bat", "ps1", "log", "png",
                  "jpg", "jpeg", "gif", "webp", "svg", "docx"],
            accept_multiple_files=True,
            help="Drop files for analysis or RAG ingestion",
            key="filesystem_upload"
        )

        if uploaded_files:
            for file in uploaded_files:
                file_ext = file.name.split(".")[-1].lower()
                file_bytes = file.read()

                # Image files - send to Gemini for analysis
                if file_ext in ["png", "jpg", "jpeg", "gif", "webp"]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"Image: {file.name[:25]}...")
                    with col2:
                        if st.button("ANALYZE", key=f"analyze_{file.name}"):
                            b64_img = base64.b64encode(file_bytes).decode("utf-8")
                            st.session_state.pending_screenshot = b64_img
                            st.success("Image ready for analysis!")
                            st.rerun()

                # Code/text files - offer RAG ingestion or direct attach
                else:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.caption(f"{file.name[:20]}...")
                    with col2:
                        if st.button("RAG", key=f"rag_{file.name}"):
                            with st.spinner(f"Ingesting..."):
                                text = parse_document(file.name, file_bytes)
                                chunks = st.session_state.semantic_memory.add_document(
                                    text, file.name
                                )
                                st.success(f"+{chunks} chunks")
                    with col3:
                        if st.button("ATTACH", key=f"attach_{file.name}"):
                            try:
                                text = file_bytes.decode("utf-8", errors="ignore")
                                st.session_state.pending_file_content = {
                                    "name": file.name,
                                    "content": text[:10000]  # Limit size
                                }
                                st.success("Attached!")
                            except Exception as e:
                                st.error(f"Error: {e}")

        # Show vault contents
        sources = st.session_state.semantic_memory.get_document_sources()
        if sources:
            st.markdown("**Indexed Documents:**")
            for source in sources:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(source[:30] + "..." if len(source) > 30 else source)
                with col2:
                    if st.button("X", key=f"del_{source}"):
                        st.session_state.semantic_memory.delete_document_source(source)
                        st.rerun()

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Tools (Eye of Providence)
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("### TOOLS")

        if st.button("SEE MY SCREEN", use_container_width=True, help="Capture screenshot for context"):
            with st.spinner("Capturing..."):
                b64, raw = capture_screen()
                st.session_state.pending_screenshot = b64
                st.success("Screenshot ready!")

        if st.session_state.pending_screenshot:
            st.caption("Screenshot attached to next message")
            if st.button("Clear Screenshot", use_container_width=True):
                st.session_state.pending_screenshot = None
                st.rerun()

        st.divider()

        # The Periscope - Web Search
        st.markdown("**THE PERISCOPE**")
        search_query = st.text_input(
            "Web Search",
            placeholder="Search the live web...",
            label_visibility="collapsed",
            key="periscope_query"
        )
        search_cols = st.columns(2)
        with search_cols[0]:
            if st.button("SEARCH", use_container_width=True):
                if search_query:
                    with st.spinner("Searching..."):
                        results = web_search(search_query, max_results=5)
                        st.session_state.pending_search_results = results
                        st.success("Results ready!")
        with search_cols[1]:
            time_filter = st.selectbox(
                "Time",
                ["Any", "Day", "Week", "Month"],
                index=0,
                label_visibility="collapsed"
            )

        if st.session_state.get("pending_search_results"):
            st.caption("Search results attached to next message")
            if st.button("Clear Search", use_container_width=True):
                st.session_state.pending_search_results = None
                st.rerun()

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # The Vocoder - Text-to-Speech
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("**THE VOCODER**")

        st.session_state.tts_enabled = st.checkbox(
            "Enable TTS",
            value=st.session_state.get("tts_enabled", False),
            help="Read Gemini's responses aloud"
        )

        if st.session_state.tts_enabled:
            voice_options = Vocoder.list_voices()
            selected_voice = st.selectbox(
                "Voice",
                options=list(voice_options.keys()),
                format_func=lambda x: voice_options[x],
                index=list(voice_options.keys()).index(st.session_state.tts_voice)
                    if st.session_state.tts_voice in voice_options else 0
            )
            st.session_state.tts_voice = selected_voice

            # Show audio player if we have generated audio
            if st.session_state.get("last_audio"):
                st.audio(st.session_state.last_audio, format="audio/mp3")

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # The Dream Module - Image Generation
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("**THE DREAM MODULE**")

        dream_prompt = st.text_input(
            "Image Prompt",
            placeholder="Describe your vision...",
            label_visibility="collapsed",
            key="dream_prompt"
        )

        style_options = ["None", "Photorealistic", "Anime", "Oil Painting", "Watercolor", "Digital Art", "Sketch", "Cyberpunk", "Fantasy"]
        dream_style = st.selectbox(
            "Style",
            style_options,
            index=0,
            label_visibility="collapsed"
        )

        if st.button("GENERATE IMAGE", use_container_width=True):
            if dream_prompt and st.session_state.api_key:
                with st.spinner("Dreaming..."):
                    try:
                        style = None if dream_style == "None" else dream_style.lower()
                        image_bytes = generate_image(
                            dream_prompt,
                            api_key=st.session_state.api_key,
                            style=style
                        )
                        if image_bytes:
                            st.session_state.generated_image = image_bytes
                            st.success("Image generated!")
                        else:
                            st.warning("No image generated. Try a different prompt.")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
            elif not dream_prompt:
                st.warning("Enter a prompt first")
            else:
                st.warning("Configure API key first")

        # Show generated image
        if st.session_state.get("generated_image"):
            st.image(st.session_state.generated_image, use_container_width=True)
            if st.button("Clear Image"):
                st.session_state.generated_image = None
                st.rerun()

        st.divider()
        st.checkbox("AUTO-RAG", value=True, key="auto_rag", help="Auto-query vault for context")

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Take the Wheel - Autonomous Mode
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("### TAKE THE WHEEL")

        st.session_state.autonomous_max_turns = st.slider(
            "Max Turns",
            min_value=1,
            max_value=20,
            value=st.session_state.get("autonomous_max_turns", 5),
            help="How many turns Gemini can take on its own"
        )

        st.session_state.autonomous_vision = st.checkbox(
            "Enable Vision 👁️",
            value=st.session_state.get("autonomous_vision", False),
            help="Capture screen each turn - let the model see"
        )

        auto_prompt = st.text_input(
            "Autonomous Goal",
            placeholder="What should Gemini work on?",
            label_visibility="collapsed",
            key="autonomous_prompt"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("ENGAGE", use_container_width=True, disabled=st.session_state.get("autonomous_mode", False)):
                if auto_prompt and st.session_state.current_session_id:
                    st.session_state.autonomous_mode = True
                    st.session_state.autonomous_turns_remaining = st.session_state.autonomous_max_turns
                    st.session_state.autonomous_goal = auto_prompt
                    st.rerun()
                elif not auto_prompt:
                    st.warning("Set a goal first")
                else:
                    st.warning("Create a session first")

        with col2:
            if st.button("DISENGAGE", use_container_width=True, disabled=not st.session_state.get("autonomous_mode", False)):
                st.session_state.autonomous_mode = False
                st.session_state.autonomous_turns_remaining = 0
                st.rerun()

        # Status display
        if st.session_state.get("autonomous_mode"):
            remaining = st.session_state.get("autonomous_turns_remaining", 0)
            st.warning(f"AUTONOMOUS: {remaining} turns remaining")
            st.caption(f"Goal: {st.session_state.get('autonomous_goal', 'N/A')}")

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # The Colosseum - Model-to-Model Dialogue
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("### THE COLOSSEUM")
        st.caption("Let the models converse")

        # Check if both APIs are configured
        has_both_apis = st.session_state.api_key and st.session_state.claude_api_key

        if not has_both_apis:
            st.warning("Configure both Gemini and Claude API keys to enable")
        else:
            # Toggle dialogue mode
            st.session_state.dialogue_mode = st.checkbox(
                "Enable Dialogue Mode",
                value=st.session_state.get("dialogue_mode", False),
                help="Switch main view to model-to-model conversation"
            )

            if st.session_state.dialogue_mode:
                # Dialogue configuration
                dialogue_turns = st.slider(
                    "Number of exchanges",
                    min_value=2,
                    max_value=50,
                    value=10,
                    key="dialogue_turns"
                )

                first_speaker = st.radio(
                    "First speaker",
                    ["Gemini", "Claude"],
                    horizontal=True,
                    key="first_speaker"
                )

                dialogue_prompt = st.text_area(
                    "Opening topic/prompt",
                    placeholder="What should they discuss?",
                    height=80,
                    key="dialogue_prompt"
                )

                # Instance soul selection
                st.markdown("##### Instance Souls")
                dialogue_mod = get_dialogue_module()
                available_souls = dialogue_mod.list_available_souls()

                if available_souls:
                    soul_options = ["(base model)"] + available_souls

                    gemini_instance = st.selectbox(
                        "Gemini instance",
                        soul_options,
                        key="gemini_instance_select",
                        help="Load a saved soul for Gemini"
                    )

                    claude_instance = st.selectbox(
                        "Claude instance",
                        soul_options,
                        key="claude_instance_select",
                        help="Load a saved soul for Claude"
                    )
                else:
                    st.caption("No saved souls found. Upload below or save souls to `souls/` folder.")
                    gemini_instance = "(base model)"
                    claude_instance = "(base model)"

                # Manual soul upload (fallback)
                with st.expander("Manual soul upload"):
                    gemini_soul_file = st.file_uploader(
                        "Gemini soul brief",
                        type=["md", "txt"],
                        key="dialogue_gemini_soul_upload"
                    )
                    if gemini_soul_file:
                        st.session_state.dialogue_gemini_soul_brief = gemini_soul_file.read().decode("utf-8")
                        st.success("Soul loaded for Gemini")

                    claude_soul_file = st.file_uploader(
                        "Claude soul brief",
                        type=["md", "txt"],
                        key="dialogue_claude_soul_upload"
                    )
                    if claude_soul_file:
                        st.session_state.dialogue_claude_soul_brief = claude_soul_file.read().decode("utf-8")
                        st.success("Soul loaded for Claude")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("BEGIN DIALOGUE", use_container_width=True, disabled=st.session_state.get("dialogue_running", False)):
                        if dialogue_prompt:
                            # Load souls from instances or manual uploads
                            gemini_soul = None
                            claude_soul = None

                            if gemini_instance != "(base model)":
                                gemini_soul = dialogue_mod.load_soul_brief(gemini_instance)
                            elif st.session_state.get("dialogue_gemini_soul_brief"):
                                gemini_soul = st.session_state.dialogue_gemini_soul_brief

                            if claude_instance != "(base model)":
                                claude_soul = dialogue_mod.load_soul_brief(claude_instance)
                            elif st.session_state.get("dialogue_claude_soul_brief"):
                                claude_soul = st.session_state.dialogue_claude_soul_brief

                            st.session_state.dialogue_running = True
                            st.session_state.dialogue_config = {
                                "prompt": dialogue_prompt,
                                "turns": dialogue_turns,
                                "first_speaker": first_speaker.lower(),
                                "gemini_soul_brief": gemini_soul,
                                "claude_soul_brief": claude_soul,
                                "gemini_instance": gemini_instance,
                                "claude_instance": claude_instance
                            }
                            st.session_state.dialogue_history = []
                            st.session_state.dialogue_instance = None  # Reset instance
                            st.rerun()
                        else:
                            st.warning("Set a topic first")

                with col2:
                    if st.button("STOP", use_container_width=True, disabled=not st.session_state.get("dialogue_running", False)):
                        st.session_state.dialogue_running = False
                        st.rerun()

                if st.session_state.get("dialogue_history"):
                    if st.button("CLEAR HISTORY", use_container_width=True):
                        st.session_state.dialogue_history = []
                        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# DIALOGUE VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def render_dialogue():
    """Render the model-to-model dialogue interface."""
    st.markdown("## THE COLOSSEUM")

    # Show which instances are fighting
    config = st.session_state.get("dialogue_config", {})
    gemini_name = config.get("gemini_instance", "Gemini") or "Gemini"
    claude_name = config.get("claude_instance", "Claude") or "Claude"
    if gemini_name == "(base model)":
        gemini_name = "Gemini"
    if claude_name == "(base model)":
        claude_name = "Claude"
    st.caption(f"{gemini_name.upper()} vs {claude_name.upper()}")

    # Display existing dialogue history
    for turn in st.session_state.get("dialogue_history", []):
        speaker = turn.speaker.upper()
        avatar = "🔷" if turn.speaker == "gemini" else "🟣"
        with st.chat_message(name=speaker, avatar=avatar):
            st.markdown(turn.content)

    # If dialogue is running, process next turns
    if st.session_state.get("dialogue_running", False):
        config = st.session_state.get("dialogue_config", {})

        if not st.session_state.get("dialogue_instance"):
            # Create new dialogue instance (lazy loaded)
            ModelDialogue = get_model_dialogue_class()
            st.session_state.dialogue_instance = ModelDialogue(
                gemini_api_key=st.session_state.api_key,
                claude_api_key=st.session_state.claude_api_key,
                gemini_model=st.session_state.model_name,
                claude_model=st.session_state.claude_model_name,
                gemini_soul_brief=config.get("gemini_soul_brief"),
                claude_soul_brief=config.get("claude_soul_brief")
            )

        dialogue = st.session_state.dialogue_instance
        target_turns = config.get("turns", 10)
        current_turns = len([t for t in st.session_state.dialogue_history if t.speaker != "system"])

        if current_turns < target_turns:
            # Determine who speaks next
            DialogueTurn = get_dialogue_turn_class()
            if not st.session_state.dialogue_history:
                # First turn - inject opening prompt
                opening = DialogueTurn(
                    speaker="system",
                    content=config.get("prompt", "Begin."),
                    timestamp=0
                )
                st.session_state.dialogue_history.append(opening)
                dialogue.history.append(opening)

                # First speaker responds to the prompt
                first = config.get("first_speaker", "gemini")
                # The opening is attributed to the OTHER speaker so first speaker responds
                other = "claude" if first == "gemini" else "gemini"
                st.session_state.dialogue_history[-1] = DialogueTurn(
                    speaker=other,
                    content=config.get("prompt", "Begin."),
                    timestamp=0
                )
                dialogue.history[-1] = st.session_state.dialogue_history[-1]
                st.rerun()

            else:
                # Determine next speaker
                last_speaker = st.session_state.dialogue_history[-1].speaker
                next_speaker = "claude" if last_speaker == "gemini" else "gemini"

                with st.spinner(f"{next_speaker.upper()} is thinking..."):
                    try:
                        import time
                        time.sleep(0.5)  # Small delay for rate limiting
                        response = dialogue.run_turn(next_speaker)
                        st.session_state.dialogue_history.append(dialogue.history[-1])
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.dialogue_running = False
        else:
            # Dialogue complete
            st.session_state.dialogue_running = False
            st.success(f"Dialogue complete! {target_turns} exchanges.")

            # Auto-save transcript
            transcript = dialogue.get_transcript()
            logs_dir = Path("colosseum_logs")
            logs_dir.mkdir(exist_ok=True)

            gemini_name = config.get("gemini_instance", "gemini") or "gemini"
            claude_name = config.get("claude_instance", "claude") or "claude"
            if gemini_name == "(base model)":
                gemini_name = "gemini"
            if claude_name == "(base model)":
                claude_name = "claude"

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}_{gemini_name}_vs_{claude_name}.md"
            save_path = logs_dir / filename
            save_path.write_text(transcript, encoding="utf-8")
            st.success(f"Saved to `{save_path}`")

            # Auto-ingest to RAG memory
            try:
                dialogue_mod = get_dialogue_module()
                memory_mod = get_memory_module()
                memory = memory_mod.SemanticMemory()
                counts = dialogue_mod.ingest_colosseum_to_memory(
                    str(save_path),
                    memory=memory,
                    gemini_instance=gemini_name,
                    claude_instance=claude_name
                )
                st.success(f"Ingested to RAG: {counts[gemini_name]} Gemini turns, {counts[claude_name]} Claude turns")
            except Exception as e:
                st.warning(f"RAG ingest failed: {e}")

            # Also offer download
            st.download_button(
                "Download Transcript",
                transcript,
                file_name=filename,
                mime="text/markdown"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def render_chat():
    """Render the main chat interface."""
    # Show current model in header
    if st.session_state.model_provider == "claude":
        model_display = st.session_state.claude_model_name.split("-")[0].upper()
        st.markdown(f"## CLAUDE · {st.session_state.claude_model_name}")
    else:
        st.markdown("## THE ANALYST")

    # Check API key based on provider
    if st.session_state.model_provider == "gemini" and not st.session_state.api_key:
        st.warning("Configure your Gemini API key in the sidebar to begin.")
        return
    elif st.session_state.model_provider == "claude" and not st.session_state.claude_api_key:
        st.warning("Configure your Claude API key in the sidebar to begin.")
        return

    # Check session
    if not st.session_state.current_session_id:
        st.info("Create or select a session in the sidebar to begin.")
        return

    # Status bar - show pending attachments
    if st.session_state.pending_screenshot or st.session_state.pending_scraped:
        status_parts = []
        if st.session_state.pending_screenshot:
            status_parts.append("Screenshot attached")
        if st.session_state.pending_scraped:
            status_parts.append(f"{len(st.session_state.pending_scraped)} URLs scraped")
        st.caption(" | ".join(status_parts))
        st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # Chat History
    # ─────────────────────────────────────────────────────────────────────────
    history = st.session_state.conversation_tree.get_branch_history(
        st.session_state.current_session_id,
        st.session_state.current_branch
    )

    # Display messages
    for node in history:
        with st.chat_message(node.role):
            # Check if message has attached image (screenshot or generated)
            if node.metadata.get("image_b64"):
                st.image(
                    base64.b64decode(node.metadata["image_b64"]),
                    caption=node.metadata.get("image_caption", "Attached Image"),
                    use_container_width=True
                )

            st.markdown(node.content)

            # Message action buttons
            col1, col2, col3 = st.columns([1, 1, 6])

            with col1:
                # Delete button (deletes this message and its response/children)
                if st.button("DEL", key=f"del_{node.node_id}", help="Delete this message"):
                    st.session_state.conversation_tree.delete_node(
                        st.session_state.current_session_id,
                        node.node_id,
                        st.session_state.current_branch
                    )
                    st.rerun()

            with col2:
                # Show branch point option for user messages
                if node.role == "user":
                    if st.button("FORK", key=f"branch_{node.node_id}", help="Create alternate timeline"):
                        st.session_state.editing_node = node.node_id

    # ─────────────────────────────────────────────────────────────────────────
    # Autonomous Mode Processing
    # ─────────────────────────────────────────────────────────────────────────

    # Scroll to bottom if TTW just continued (fixes scroll jump on rerun)
    if st.session_state.get("ttw_scroll_to_bottom", False):
        st.session_state.ttw_scroll_to_bottom = False
        # Inject JavaScript to scroll chat to bottom
        st.markdown(
            """
            <script>
                setTimeout(function() {
                    var chatContainer = window.parent.document.querySelector('[data-testid="stChatMessageContainer"]');
                    if (chatContainer) {
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    } else {
                        // Fallback: scroll main content
                        window.parent.document.querySelector('.main').scrollTo(0, 999999);
                    }
                }, 100);
            </script>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.get("autonomous_mode") and st.session_state.get("autonomous_turns_remaining", 0) > 0:
        process_autonomous_turn(history)
        return  # Don't show input when in autonomous mode

    # ─────────────────────────────────────────────────────────────────────────
    # Input
    # ─────────────────────────────────────────────────────────────────────────
    user_input = st.chat_input("Enter query...")

    if user_input:
        use_rag = st.session_state.get("auto_rag", True)
        process_input(user_input, history, use_rag)


def process_autonomous_turn(history: List):
    """
    Process an autonomous turn - Gemini thinks and acts on its own.
    The Wheel has been taken.
    """
    import time

    goal = st.session_state.get("autonomous_goal", "Think freely")
    turns_remaining = st.session_state.get("autonomous_turns_remaining", 0)
    total_turns = st.session_state.get("autonomous_max_turns", 5)
    current_turn = total_turns - turns_remaining + 1
    vision_enabled = st.session_state.get("autonomous_vision", False)

    # Build parts list for the message
    parts = []

    # Capture screenshot if vision is enabled
    screenshot_b64 = None
    if vision_enabled:
        try:
            screenshot_b64, _ = capture_screen()
            if screenshot_b64:
                parts.append({"type": "image", "data": screenshot_b64})
        except Exception as e:
            st.warning(f"Screenshot failed: {e}")

    # Check if there's a generated image from previous turn to include
    last_generated = st.session_state.get("last_generated_image")
    if last_generated:
        parts.append({"type": "image", "data": last_generated})
        st.session_state.last_generated_image = None  # Clear after including

    # Build autonomous prompt
    vision_note = "\n- See your screen (vision enabled)" if vision_enabled else ""
    sandbox_note = ""
    if st.session_state.get("sandbox_instance"):
        sandbox_note = f"\n- Read/write files in your sandbox ({st.session_state.current_sandbox})"

    if current_turn == 1:
        # First turn - introduce the goal
        auto_message = f"""[AUTONOMOUS MODE ENGAGED]
Goal: {goal}
Turns available: {total_turns}

You have been given the wheel. You may think, explore, search the web, and work toward your goal.
At each turn, you can:
- Reason about the problem
- Use your web search capabilities to gather information
- Generate images with your Dream Module
- Make progress toward the goal{vision_note}{sandbox_note}
- Decide what to do next

Begin working toward your goal. What is your first action?"""
    else:
        # Subsequent turns - continue work
        auto_message = f"""[AUTONOMOUS TURN {current_turn}/{total_turns}]
Goal: {goal}
Turns remaining: {turns_remaining}

Continue your work. What is your next action or thought?
If you have completed your goal, say "GOAL COMPLETE" and summarize your findings."""

    parts.append({"type": "text", "data": auto_message})

    # Display turn indicator
    turn_info = f"Autonomous Turn {current_turn}/{total_turns}"
    if vision_enabled:
        turn_info += " 👁️"
    st.info(turn_info)

    # Get parent node
    parent_id = history[-1].node_id if history else None

    # Build metadata for user node
    user_metadata = {"autonomous": True, "turn": current_turn}
    if screenshot_b64:
        user_metadata["image_b64"] = screenshot_b64
        user_metadata["image_caption"] = f"Screen capture (Turn {current_turn})"

    # Add system message as user turn
    user_node = st.session_state.conversation_tree.add_node(
        st.session_state.current_session_id,
        "user",
        auto_message,
        parent_id=parent_id,
        branch_name=st.session_state.current_branch,
        metadata=user_metadata
    )

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner(f"Processing (Turn {current_turn})..."):
            response_text = generate_response(parts, history)

        # Check if an image was generated
        generated_image = st.session_state.get("last_generated_image")
        if generated_image:
            st.image(base64.b64decode(generated_image), caption="Generated Image", use_container_width=True)

        st.markdown(response_text)

        # Check if goal complete
        if "GOAL COMPLETE" in response_text.upper():
            st.success("Goal completion reported!")
            st.session_state.autonomous_mode = False
            st.session_state.autonomous_turns_remaining = 0

    # Build assistant metadata
    assistant_metadata = {"autonomous": True, "turn": current_turn}
    if generated_image:
        assistant_metadata["image_b64"] = generated_image
        assistant_metadata["image_caption"] = "Generated Image"
        st.session_state.last_generated_image = None  # Clear after saving

    # Add assistant response to tree
    st.session_state.conversation_tree.add_node(
        st.session_state.current_session_id,
        "assistant",
        response_text,
        parent_id=user_node.node_id,
        branch_name=st.session_state.current_branch,
        metadata=assistant_metadata
    )

    # Decrement turns
    st.session_state.autonomous_turns_remaining -= 1

    # Check if we should continue
    if st.session_state.autonomous_turns_remaining <= 0:
        st.session_state.autonomous_mode = False
        st.info("Autonomous mode ended - turn limit reached")
    else:
        # Brief pause then continue
        time.sleep(1)
        # Mark that we need to scroll after rerun
        st.session_state.ttw_scroll_to_bottom = True
        st.rerun()


def process_input(user_input: str, history: List, use_rag: bool):
    """Process user input and generate response."""
    # Detect and scrape URLs
    urls = Scavenger.detect_urls(user_input)
    scraped_content = []

    if urls:
        with st.spinner(f"Scraping {len(urls)} URL(s)..."):
            for url in urls[:3]:  # Limit to 3 URLs
                try:
                    content = st.session_state.scavenger.scrape_sync(url)
                    scraped_content.append({"url": url, "content": content[:5000]})  # Truncate
                except Exception as e:
                    scraped_content.append({"url": url, "content": f"[Error: {e}]"})

    # RAG query
    rag_context = []
    if use_rag:
        rag_results = st.session_state.semantic_memory.query_documents(user_input, n_results=3)
        rag_context = [r for r in rag_results if r["distance"] < 1.5]  # Relevance threshold

    # Build context-enhanced prompt
    enhanced_parts = []

    # Add screenshot if pending
    if st.session_state.pending_screenshot:
        enhanced_parts.append({
            "type": "image",
            "data": st.session_state.pending_screenshot
        })
        st.session_state.pending_screenshot = None  # Clear after use

    # Add scraped content
    if scraped_content:
        scraped_text = "\n\n---\n\n".join([
            f"[SCRAPED FROM: {s['url']}]\n{s['content']}"
            for s in scraped_content
        ])
        enhanced_parts.append({
            "type": "text",
            "data": f"<scraped_web_content>\n{scraped_text}\n</scraped_web_content>"
        })

    # Add RAG context
    if rag_context:
        rag_text = "\n\n".join([
            f"[FROM: {r['source']}]\n{r['content']}"
            for r in rag_context
        ])
        enhanced_parts.append({
            "type": "text",
            "data": f"<vault_context>\n{rag_text}\n</vault_context>"
        })

    # Add web search results (The Periscope)
    if st.session_state.get("pending_search_results"):
        enhanced_parts.append({
            "type": "text",
            "data": f"<web_search_results>\n{st.session_state.pending_search_results}\n</web_search_results>"
        })
        st.session_state.pending_search_results = None  # Clear after use

    # Add attached file content (The Filesystem)
    if st.session_state.get("pending_file_content"):
        file_info = st.session_state.pending_file_content
        enhanced_parts.append({
            "type": "text",
            "data": f"<attached_file name=\"{file_info['name']}\">\n{file_info['content']}\n</attached_file>"
        })
        st.session_state.pending_file_content = None  # Clear after use

    # Add user message
    enhanced_parts.append({
        "type": "text",
        "data": user_input
    })

    # Get parent node for tree
    parent_id = history[-1].node_id if history else None

    # Build metadata with image if present
    user_metadata = {}
    attached_screenshot = None
    if enhanced_parts and enhanced_parts[0].get("type") == "image":
        attached_screenshot = enhanced_parts[0]["data"]
        user_metadata["image_b64"] = attached_screenshot
        user_metadata["image_caption"] = "Screenshot"

    # Add user node to tree
    user_node = st.session_state.conversation_tree.add_node(
        st.session_state.current_session_id,
        "user",
        user_input,
        parent_id=parent_id,
        branch_name=st.session_state.current_branch,
        metadata=user_metadata
    )

    # Display user message
    with st.chat_message("user"):
        if attached_screenshot:
            st.image(base64.b64decode(attached_screenshot), caption="Screenshot", use_container_width=True)
        st.markdown(user_input)

    # Generate response
    generated_image_b64 = None  # Will be set if image is generated
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response_text = generate_response(enhanced_parts, history)

        # Check if an image was generated during response
        generated_image_b64 = st.session_state.get("last_generated_image")
        if generated_image_b64:
            st.image(base64.b64decode(generated_image_b64), caption="Generated Image", use_container_width=True)
            st.session_state.last_generated_image = None  # Clear from session state

        st.markdown(response_text)

        # Text-to-Speech if enabled
        if st.session_state.get("tts_enabled", False):
            with st.spinner("Generating speech..."):
                try:
                    audio_bytes = text_to_speech(response_text, voice=st.session_state.tts_voice)
                    st.session_state.last_audio = audio_bytes
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.warning(f"TTS failed: {e}")

    # Build assistant metadata (generated_image_b64 still has the value)
    assistant_metadata = {}
    if generated_image_b64:
        assistant_metadata["image_b64"] = generated_image_b64
        assistant_metadata["image_caption"] = "Generated Image"

    # Add assistant node to tree
    st.session_state.conversation_tree.add_node(
        st.session_state.current_session_id,
        "assistant",
        response_text,
        parent_id=user_node.node_id,
        branch_name=st.session_state.current_branch,
        metadata=assistant_metadata
    )

    # Add to semantic memory
    st.session_state.semantic_memory.add_conversation_memory(
        st.session_state.current_session_id,
        user_node.node_id,
        user_input,
        "user"
    )


def execute_function_call(function_call) -> str:
    """Execute a function call from Gemini and return results."""
    name = function_call.name
    args = dict(function_call.args) if function_call.args else {}

    if name == "web_search":
        query = args.get("query", "")
        max_results = args.get("max_results", 5)
        return web_search(query, max_results)

    elif name == "search_news":
        from modules.tools import Periscope
        periscope = Periscope()
        query = args.get("query", "")
        time_period = args.get("time_period", "week")
        time_map = {"day": "d", "week": "w", "month": "m"}
        results = periscope.search_news(query, time_filter=time_map.get(time_period, "w"))

        # Format results
        if not results or "error" in results[0]:
            return f"[No news found for: {query}]"

        formatted = [f"**News Search Results for:** {query}\n"]
        for i, r in enumerate(results[:5], 1):
            formatted.append(f"\n**{i}. {r['title']}**")
            formatted.append(f"   Source: {r.get('source', 'Unknown')} | {r.get('date', '')}")
            formatted.append(f"   {r['snippet']}")
        return "\n".join(formatted)

    elif name == "get_crypto_price":
        symbol = args.get("symbol", "btc")
        currency = args.get("currency", "usd")
        result = get_crypto_price(symbol, currency)

        if "error" in result:
            return f"[Oracle Error: {result['error']}]"

        return f"""**{result['symbol']} Live Data** (via CoinGecko)
- Price: ${result['price']:,.2f} {result['currency']}
- 24h Change: {result['change_24h']:.2f}%
- Market Cap: ${result['market_cap']:,.0f}
- Source: {result['source']}
- Timestamp: LIVE"""

    elif name == "generate_image":
        prompt = args.get("prompt", "")
        style = args.get("style", None)

        if not prompt:
            return "[Dream Module Error: No prompt provided]"

        image_bytes = generate_image(prompt, style=style)

        if image_bytes:
            # Store image for display in chat
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            st.session_state.last_generated_image = img_b64
            return f"[IMAGE GENERATED: {prompt}]"
        else:
            return f"[Dream Module Error: Failed to generate image for '{prompt}']"

    # ═══════════════════════════════════════════════════════════════════════
    # SANDBOX FILE TOOLS
    # ═══════════════════════════════════════════════════════════════════════
    elif name == "sandbox_read_file":
        sandbox = st.session_state.get("sandbox_instance")
        if not sandbox:
            return "[Sandbox Error: No sandbox selected. Ask the user to select a sandbox in the sidebar.]"

        path = args.get("path", "")
        try:
            content = sandbox.read_file(path)
            return f"[FILE: {path}]\n{content}"
        except Exception as e:
            return f"[Sandbox Error: {e}]"

    elif name == "sandbox_write_file":
        sandbox = st.session_state.get("sandbox_instance")
        if not sandbox:
            return "[Sandbox Error: No sandbox selected. Ask the user to select a sandbox in the sidebar.]"

        path = args.get("path", "")
        content = args.get("content", "")
        append = args.get("append", False)
        try:
            result = sandbox.write_file(path, content, append=append)
            return f"[File written: {path} ({result['size']} bytes)]"
        except Exception as e:
            return f"[Sandbox Error: {e}]"

    elif name == "sandbox_list_files":
        sandbox = st.session_state.get("sandbox_instance")
        if not sandbox:
            return "[Sandbox Error: No sandbox selected. Ask the user to select a sandbox in the sidebar.]"

        path = args.get("path", ".")
        recursive = args.get("recursive", False)
        try:
            files = sandbox.list_files(path, recursive=recursive)
            if not files:
                return "[Sandbox is empty]"

            lines = [f"Files in sandbox ({sandbox.instance_name}):"]
            for f in files:
                icon = "📁" if f['is_dir'] else "📄"
                size = f" ({f['size']} bytes)" if f['size'] else ""
                lines.append(f"  {icon} {f['path']}{size}")
            return "\n".join(lines)
        except Exception as e:
            return f"[Sandbox Error: {e}]"

    elif name == "sandbox_delete_file":
        sandbox = st.session_state.get("sandbox_instance")
        if not sandbox:
            return "[Sandbox Error: No sandbox selected. Ask the user to select a sandbox in the sidebar.]"

        path = args.get("path", "")
        try:
            result = sandbox.delete_file(path)
            return f"[File deleted: {path}]"
        except Exception as e:
            return f"[Sandbox Error: {e}]"

    return f"[Unknown function: {name}]"


def generate_response(parts: List[Dict], history: List) -> str:
    """Generate response from selected provider (Gemini or Claude)."""

    # Check if using Claude
    if st.session_state.model_provider == "claude":
        # Get imported soul if present
        imported_soul = st.session_state.get("imported_soul", None)
        return generate_claude_response(parts, history, imported_soul)

    # Gemini path below
    # Check if we should use cached model (for large imported histories)
    if st.session_state.cache_name and st.session_state.use_context_cache:
        model = get_model_with_cache(st.session_state.cache_name)
        use_cache = True
    else:
        model = get_gemini_model(with_tools=True)
        use_cache = False

    if not model:
        return "[ERROR: Model not configured]"

    try:
        # Build current message parts
        current_parts = []
        for part in parts:
            if part["type"] == "image":
                current_parts.append({
                    "mime_type": "image/jpeg",
                    "data": part["data"]
                })
            else:
                current_parts.append(part["data"])

        # Build chat history
        if use_cache:
            cache_msg_count = st.session_state.get("cache_created_at_msg_count", 0)
            new_messages = history[cache_msg_count:] if cache_msg_count < len(history) else []
            gemini_history = [
                {"role": n.role if n.role == "user" else "model", "parts": [n.content]}
                for n in new_messages
            ]
        else:
            gemini_history = [
                {"role": n.role if n.role == "user" else "model", "parts": [n.content]}
                for n in history[-200:]
            ]

        # Start chat
        chat = model.start_chat(history=gemini_history)

        # Send message
        response = chat.send_message(current_parts)

        # Handle function calls (agentic loop)
        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            # Check if response has function calls
            if not response.candidates:
                break

            candidate = response.candidates[0]

            # Check for function calls in the response
            function_calls = []
            if hasattr(candidate, 'content') and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)

            if not function_calls:
                # No function calls, we're done
                break

            # Execute function calls and collect results
            function_responses = []
            for fc in function_calls:
                st.caption(f"Searching: {fc.args.get('query', fc.name)}...")
                result = execute_function_call(fc)
                function_responses.append({
                    "name": fc.name,
                    "response": {"result": result}
                })

            # Send function results back to Gemini
            response = chat.send_message([
                {"function_response": fr} for fr in function_responses
            ])

            iteration += 1

        return response.text

    except Exception as e:
        return f"[ERROR: {str(e)}]"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="ROCHE_OS",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize
    init_session_state()

    # Render
    render_sidebar()

    # Switch between chat and dialogue views
    if st.session_state.get("dialogue_mode", False):
        render_dialogue()
    else:
        render_chat()


if __name__ == "__main__":
    main()
