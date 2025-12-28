"""
ROCHE_OS_V2 - Cognitive Prosthetic for Gemini & Claude
Main Streamlit Interface - Multi-Instance Edition

"Void Research Lab" - Dark, dense, clinical.

V2 Features:
- Multiple AI personas running simultaneously
- Persistent instances with independent state
- Inter-instance messaging
- Tabbed interface for instance switching
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (relative to this script)
load_dotenv(Path(__file__).parent / ".env")

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

# Lazy import for instance module (V2)
_instance_module = None

def get_instance_module():
    global _instance_module
    if _instance_module is None:
        from modules import instance
        _instance_module = instance
    return _instance_module

# Lazy import for idle mode module (V2)
_idle_mode_module = None

def get_idle_mode_module():
    global _idle_mode_module
    if _idle_mode_module is None:
        from modules import idle_mode
        _idle_mode_module = idle_mode
    return _idle_mode_module

def get_idle_manager():
    return get_idle_mode_module().get_idle_manager()

def get_idle_config_class():
    return get_idle_mode_module().IdleConfig

def get_idle_stop_reason():
    return get_idle_mode_module().IdleStopReason

def get_instance_manager():
    return get_instance_module().InstanceManager()

def get_message_queue():
    return get_instance_module().MessageQueue()

def get_instance_state_class():
    return get_instance_module().InstanceState

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

    /* Instance tabs - FIXED positioning (nuclear option) */
    .main [data-testid="stTabs"] > div:first-child {
        position: fixed !important;
        top: 50px !important;
        left: 22rem !important;
        right: 1rem !important;
        z-index: 9999 !important;
        background: #0a0a0a !important;
        padding: 0.5rem 0 !important;
        border-bottom: 1px solid #333 !important;
    }

    /* Push tab content down to avoid overlap */
    .main [data-testid="stTabs"] > div:nth-child(2) {
        padding-top: 3rem !important;
    }

    [data-baseweb="tab-list"] {
        background: #0a0a0a !important;
        gap: 4px;
    }

    [data-baseweb="tab"] {
        background: #111 !important;
        border: 1px solid #333 !important;
        border-radius: 4px 4px 0 0 !important;
        color: #888 !important;
        padding: 0.4rem 1rem !important;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background: #1e3a5f !important;
        border-color: #2d5a8b !important;
        color: #fff !important;
    }

    [data-baseweb="tab"]:hover {
        background: #1a1a1a !important;
        color: #ccc !important;
    }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# API KEY PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def save_api_keys_to_env(gemini_key: str = None, claude_key: str = None):
    """Save API keys to .env file for persistence across sessions."""
    env_path = Path(__file__).parent / ".env"

    # Read existing .env
    existing = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                existing[key.strip()] = val.strip()

    # Update with new values (only if provided)
    if gemini_key is not None:
        existing['GEMINI_API_KEY'] = gemini_key
    if claude_key is not None:
        existing['ANTHROPIC_API_KEY'] = claude_key

    # Write back
    lines = [
        "# ROCHE_OS API Keys",
        "# These are loaded automatically on startup",
        "",
    ]
    for key, val in existing.items():
        lines.append(f"{key}={val}")

    env_path.write_text('\n'.join(lines) + '\n')


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize Streamlit session state for multi-instance V2."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

        # Render cycle counter - used to prevent rerun cascade during startup
        st.session_state.render_count = 0

        # ═══════════════════════════════════════════════════════════════════════════
        # SHARED RESOURCES (Singletons)
        # ═══════════════════════════════════════════════════════════════════════════
        st.session_state.conversation_tree = get_conversation_tree()
        st.session_state.semantic_memory = get_semantic_memory()
        st.session_state.eye = EyeOfProvidence()
        st.session_state.scavenger = Scavenger()

        # ═══════════════════════════════════════════════════════════════════════════
        # MULTI-INSTANCE MANAGEMENT (V2)
        # ═══════════════════════════════════════════════════════════════════════════
        st.session_state.instance_manager = get_instance_manager()
        st.session_state.message_queue = get_message_queue()

        # Active instances (keyed by instance_id)
        st.session_state.instances = {}

        # Currently focused instance
        st.session_state.active_instance_id = None

        # UI state
        st.session_state.show_instance_creation = False
        st.session_state.confirm_delete_instance = None

        # Load persisted instances from database
        _load_persisted_instances()

        # ═══════════════════════════════════════════════════════════════════════════
        # GLOBAL API KEYS (Shared by all instances)
        # ═══════════════════════════════════════════════════════════════════════════
        st.session_state.api_keys = {
            "gemini": os.environ.get("GEMINI_API_KEY", ""),
            "claude": os.environ.get("ANTHROPIC_API_KEY", "")
        }

        # Legacy compatibility - keep these for now
        st.session_state.api_key = st.session_state.api_keys["gemini"]
        st.session_state.claude_api_key = st.session_state.api_keys["claude"]

        # ═══════════════════════════════════════════════════════════════════════════
        # GLOBAL SETTINGS (Shared across instances)
        # ═══════════════════════════════════════════════════════════════════════════
        st.session_state.debug_api_calls = False
        st.session_state.use_context_cache = True

        # Text-to-Speech settings
        st.session_state.tts_enabled = False
        st.session_state.tts_voice = Vocoder.DEFAULT_VOICE
        st.session_state.last_audio = None

        # Dialogue mode ("The Colosseum") - global for now
        st.session_state.dialogue_mode = False
        st.session_state.dialogue_history = []
        st.session_state.dialogue_running = False

        # ═══════════════════════════════════════════════════════════════════════════
        # LEGACY COMPATIBILITY (for gradual migration)
        # These are now instance-specific but kept for backwards compatibility
        # ═══════════════════════════════════════════════════════════════════════════
        st.session_state.current_session_id = None
        st.session_state.current_branch = "main"
        st.session_state.pending_screenshot = None
        st.session_state.pending_scraped = []
        st.session_state.model_provider = "gemini"
        st.session_state.model_name = "gemini-1.5-pro-latest"
        st.session_state.claude_model_name = "claude-sonnet-4-20250514"
        st.session_state.sliding_window_enabled = True  # Default ON to prevent rate limits
        st.session_state.sliding_window_size = 50
        st.session_state.soul_brief = None
        st.session_state.use_rag_memory = True
        st.session_state.cached_context = None
        st.session_state.cache_name = None
        st.session_state.cache_created_at_msg_count = 0
        st.session_state.imported_soul = None
        st.session_state.autonomous_mode = False
        st.session_state.autonomous_turns_remaining = 0
        st.session_state.autonomous_max_turns = 5
        st.session_state.autonomous_vision = False
        st.session_state.current_sandbox = None
        st.session_state.sandbox_instance = None


def _load_persisted_instances():
    """Load all persisted instances from database into session state."""
    try:
        all_instances = st.session_state.instance_manager.get_all_instances()
        for instance in all_instances:
            st.session_state.instances[instance.instance_id] = instance

            # Set up sandbox for each instance
            if instance.sandbox_name:
                try:
                    instance.sandbox = get_sandbox_instance(instance.sandbox_name)
                except Exception:
                    pass

        # If we have instances, activate the most recently used one (no rerun on initial load)
        if all_instances:
            st.session_state.active_instance_id = all_instances[0].instance_id
            _sync_instance_to_legacy(all_instances[0])
            # Don't rerun here - this is initial load
    except Exception as e:
        # Database might not have the tables yet
        pass


def _sync_instance_to_legacy(instance):
    """Sync instance state to legacy session state variables for compatibility."""
    if instance is None:
        return

    # These are safe to set directly (no widgets with these keys)
    st.session_state.soul_brief = instance.soul_brief
    st.session_state.current_session_id = instance.current_session_id
    st.session_state.current_branch = instance.current_branch
    st.session_state.pending_screenshot = instance.pending_screenshot
    st.session_state.pending_scraped = instance.pending_scraped

    # These have widgets with matching keys - only set if widget not yet instantiated
    # (Streamlit throws error if widget already owns the key)
    def safe_set(key, value):
        try:
            st.session_state[key] = value
        except Exception:
            pass  # Widget already instantiated

    safe_set("model_provider", instance.model_provider)
    safe_set("model_name", instance.model_name)
    safe_set("claude_model_name", instance.model_name if instance.model_provider == "claude" else st.session_state.get("claude_model_name"))
    safe_set("sliding_window_enabled", instance.sliding_window_enabled)
    safe_set("sliding_window_size", instance.sliding_window_size)
    safe_set("use_rag_memory", instance.use_rag_memory)

    if instance.sandbox_name:
        st.session_state.current_sandbox = instance.sandbox_name
        try:
            st.session_state.sandbox_instance = get_sandbox_instance(instance.sandbox_name)
        except Exception:
            st.session_state.sandbox_instance = None


def get_current_instance():
    """Get the currently active instance."""
    instance_id = st.session_state.get("active_instance_id")
    if instance_id and instance_id in st.session_state.get("instances", {}):
        return st.session_state.instances[instance_id]
    return None


def set_active_instance(instance_id: str, trigger_rerun: bool = True):
    """Switch to a different instance."""
    if instance_id in st.session_state.get("instances", {}):
        old_instance_id = st.session_state.get("active_instance_id")
        st.session_state.active_instance_id = instance_id
        instance = st.session_state.instances[instance_id]
        instance.status = "active"
        instance.updated_at = datetime.now().isoformat()

        # Preserve API keys before clearing widget state (they're global, not per-instance)
        saved_gemini_key = st.session_state.get("api_key", "")
        saved_claude_key = st.session_state.get("claude_api_key", "")

        # Clear widget keys that need to re-initialize for the new instance
        # This prevents stale widget state from overriding instance values
        widget_keys_to_clear = [
            "sidebar_provider_radio",
            "sidebar_gemini_model",
            "sidebar_claude_model",
            "model_name",  # Custom model text input
            "claude_model_name",
        ]
        for key in widget_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Sync to legacy state
        _sync_instance_to_legacy(instance)

        # Restore API keys (they're global, not instance-specific)
        # Use try/except because widget might already be instantiated
        if saved_gemini_key:
            try:
                st.session_state.api_key = saved_gemini_key
            except Exception:
                pass  # Widget owns this key, will retain its value anyway
        if saved_claude_key:
            try:
                st.session_state.claude_api_key = saved_claude_key
            except Exception:
                pass

        # Save updated status
        st.session_state.instance_manager.save_instance(instance)

        # Rerun to refresh UI if instance actually changed
        if trigger_rerun and old_instance_id != instance_id:
            st.rerun()


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
                },
                {
                    "name": "send_message",
                    "description": "Send a message to another AI instance in ROCHE_OS. Use this to communicate with other personas.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to_instance": {
                                "type": "string",
                                "description": "Name of the target instance (e.g., 'Tessera', 'Vigil')"
                            },
                            "message": {
                                "type": "string",
                                "description": "The message content to send"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["normal", "urgent"],
                                "description": "Message priority (urgent messages are shown immediately)"
                            }
                        },
                        "required": ["to_instance", "message"]
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

def import_soul_from_json(json_data: dict, index_to_rag: bool = False) -> tuple:
    """
    Import conversation history from scraped AI Studio export.
    Creates a new session with the imported history.
    Returns (session_id, message_count).
    Accepts both 'contents' (Gemini format) and 'history' keys.

    Args:
        json_data: The soul export JSON
        index_to_rag: If True, also index to ChromaDB (slow for large imports)
    """
    import time
    start_time = time.time()

    # Accept both formats
    history = json_data.get("contents", []) or json_data.get("history", [])

    if not history:
        raise ValueError("No contents/history found in JSON")

    total_msgs = len(history)
    print(f"[IMPORT] Starting import of {total_msgs} messages...")

    # Create new session for the imported soul
    session_name = f"SOUL_IMPORT_{datetime.now().strftime('%Y%m%d_%H%M')}"
    session_id = st.session_state.conversation_tree.create_session(session_name)
    print(f"[IMPORT] Session created: {session_id}")

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

        # Progress logging every 50 messages
        if count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[IMPORT] Progress: {count}/{total_msgs} ({elapsed:.1f}s)")

        # Optionally add to semantic memory for RAG (slow!)
        if index_to_rag:
            st.session_state.semantic_memory.add_conversation_memory(
                session_id,
                node.node_id,
                content,
                storage_role
            )

    elapsed = time.time() - start_time
    print(f"[IMPORT] Complete: {count} messages in {elapsed:.1f}s")
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
    },
    {
        "name": "send_message",
        "description": "Send a message to another AI instance in ROCHE_OS. Use this to communicate with other personas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to_instance": {
                    "type": "string",
                    "description": "Name of the target instance (e.g., 'Tessera', 'Vigil')"
                },
                "message": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "priority": {
                    "type": "string",
                    "enum": ["normal", "urgent"],
                    "description": "Message priority (urgent messages are shown immediately)"
                }
            },
            "required": ["to_instance", "message"]
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

    elif name == "send_message":
        to_instance_name = args.get("to_instance", "")
        message = args.get("message", "")
        priority = 1 if args.get("priority") == "urgent" else 0

        # Get current instance
        current_instance = get_current_instance()
        if not current_instance:
            return "[Message Error: No active instance]"

        # Find target instance by name
        target = None
        for inst in st.session_state.get("instances", {}).values():
            if inst.name.lower() == to_instance_name.lower():
                target = inst
                break

        if not target:
            available = [i.name for i in st.session_state.get("instances", {}).values()
                        if i.instance_id != current_instance.instance_id]
            return f"[Message Error: Instance '{to_instance_name}' not found. Available: {', '.join(available)}]"

        # Queue message
        msg_id = st.session_state.message_queue.send(
            from_instance_id=current_instance.instance_id,
            to_instance_id=target.instance_id,
            content=message,
            priority=priority
        )

        return f"[Message sent to {target.name}. Message ID: {msg_id}]"

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
        st.markdown("# ROCHE_OS V2")
        st.markdown('<span class="status-dot online"></span> MULTI-INSTANCE', unsafe_allow_html=True)
        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # API Configuration
        # ─────────────────────────────────────────────────────────────────────
        with st.expander("API CONFIG", expanded=not st.session_state.get("api_key") and not st.session_state.get("claude_api_key")):
            # Provider selection - use key-based state management
            provider_options = ["gemini", "claude"]
            current_idx = provider_options.index(st.session_state.get("model_provider", "gemini")) if st.session_state.get("model_provider", "gemini") in provider_options else 0

            provider = st.radio(
                "Model Provider",
                ["Gemini", "Claude"],
                index=current_idx,
                horizontal=True,
                help="Choose your AI backend",
                key="sidebar_provider_radio"
            )
            # Update session state from widget (no rerun needed)
            st.session_state.model_provider = provider.lower()

            st.divider()

            if st.session_state.model_provider == "gemini":
                # Gemini configuration - use key-based binding
                gemini_key_configured = bool(st.session_state.get("api_key", ""))
                key_status = " [OK]" if gemini_key_configured else ""
                st.text_input(
                    f"Gemini API Key{key_status}",
                    type="password",
                    help="Your Google AI Studio API key" + (" - Key is configured!" if gemini_key_configured else ""),
                    key="api_key"
                )

                model_presets = [
                    # Gemini 2.5 (latest)
                    "gemini-2.5-pro-preview-06-05",
                    "gemini-2.5-flash-preview-05-20",
                    # Gemini 2.0
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-exp",
                    "gemini-2.0-flash-thinking-exp-01-21",
                    # Gemini 1.5
                    "gemini-1.5-pro-latest",
                    "gemini-1.5-flash-latest",
                    # Experimental
                    "gemini-exp-1206",
                    "learnlm-1.5-pro-experimental",
                    "Custom..."
                ]

                # Find current model in presets or default to Custom
                try:
                    model_idx = model_presets.index(st.session_state.get("model_name", "gemini-1.5-pro-latest"))
                except ValueError:
                    model_idx = len(model_presets) - 1  # Custom...

                model = st.selectbox(
                    "Model",
                    model_presets,
                    index=model_idx,
                    key="sidebar_gemini_model"
                )

                if model == "Custom...":
                    st.text_input(
                        "Custom Model ID",
                        placeholder="e.g. gemini-3-pro",
                        key="model_name"
                    )
                elif model != st.session_state.get("model_name"):
                    st.session_state.model_name = model

                # Context caching toggle (Gemini only)
                st.checkbox(
                    "Use Context Caching",
                    help="Cache large histories on Google's servers for faster/cheaper responses",
                    key="use_context_cache"
                )

            else:
                # Claude configuration
                if not CLAUDE_AVAILABLE():
                    st.error("Anthropic SDK not installed. Run: pip install anthropic")
                else:
                    claude_key_configured = bool(st.session_state.get("claude_api_key", ""))
                    key_status = " [OK]" if claude_key_configured else ""
                    st.text_input(
                        f"Claude API Key{key_status}",
                        type="password",
                        help="Your Anthropic API key" + (" - Key is configured!" if claude_key_configured else ""),
                        key="claude_api_key"
                    )

                    claude_models = [
                        # Claude 4 (latest)
                        "claude-sonnet-4-20250514",
                        "claude-opus-4-20250514",
                        # Claude 3.5
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-sonnet-latest",
                        "claude-3-5-haiku-20241022",
                        "claude-3-5-haiku-latest",
                        # Claude 3
                        "claude-3-opus-20240229",
                        "claude-3-opus-latest",
                        "claude-3-haiku-20240307",
                        "Custom..."
                    ]

                    # Find current model in presets or default to Custom
                    try:
                        claude_idx = claude_models.index(st.session_state.get("claude_model_name", "claude-sonnet-4-20250514"))
                    except ValueError:
                        claude_idx = len(claude_models) - 1  # Custom...

                    claude_model = st.selectbox(
                        "Model",
                        claude_models,
                        index=claude_idx,
                        key="sidebar_claude_model"
                    )

                    if claude_model == "Custom...":
                        st.text_input(
                            "Custom Model ID",
                            placeholder="e.g. claude-3-opus-20240229",
                            key="claude_model_name"
                        )
                    elif claude_model != st.session_state.get("claude_model_name"):
                        st.session_state.claude_model_name = claude_model

                    # Extended thinking toggle for Claude
                    st.checkbox(
                        "Extended Thinking",
                        help="Enable Claude's extended thinking for complex reasoning",
                        key="claude_extended_thinking"
                    )

                    st.divider()
                    st.markdown("**SLIDING WINDOW MODE**")
                    st.caption("For large soul files that exceed rate limits")

                    st.checkbox(
                        "Enable Sliding Window",
                        help="Only send recent messages + soul brief instead of full history",
                        key="sliding_window_enabled"
                    )

                    if st.session_state.sliding_window_enabled:
                        st.slider(
                            "Recent messages to include",
                            min_value=10,
                            max_value=200,
                            help="Number of most recent messages to send with each request",
                            key="sliding_window_size"
                        )

                        st.checkbox(
                            "RAG Memory Retrieval",
                            help="Query semantic memory for relevant older context",
                            key="use_rag_memory"
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
                            if st.button("Clear Soul Brief", key="clear_soul_brief_btn"):
                                st.session_state.soul_brief = None
                                st.rerun()

                    st.divider()

                    # Debug mode
                    st.checkbox(
                        "Debug Mode",
                        help="Show token counts and payload details before each API call",
                        key="debug_api_calls"
                    )

            # Save API Keys button (outside provider-specific sections but inside expander)
            st.divider()
            col_save, col_status = st.columns([2, 1])
            with col_save:
                if st.button("SAVE API KEYS", use_container_width=True, help="Save keys to .env for persistence"):
                    gemini_key = st.session_state.get("api_key", "")
                    claude_key = st.session_state.get("claude_api_key", "")
                    save_api_keys_to_env(gemini_key, claude_key)
                    # Also update the api_keys dict
                    st.session_state.api_keys["gemini"] = gemini_key
                    st.session_state.api_keys["claude"] = claude_key
                    st.success("Saved!")
                    st.rerun()
            with col_status:
                g_ok = bool(st.session_state.get("api_key", ""))
                c_ok = bool(st.session_state.get("claude_api_key", ""))
                status = f"G:{'OK' if g_ok else '--'} C:{'OK' if c_ok else '--'}"
                st.caption(status)

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

                    # Extract default name from filename (e.g., "gemini_soul_export_shoggoth.json" -> "Shoggoth")
                    filename = soul_file.name.replace(".json", "")
                    default_name = filename.split("_")[-1].title() if "_" in filename else filename.title()

                    # Instance name input
                    import_name = st.text_input(
                        "Instance Name",
                        value=default_name,
                        help="Name for the imported soul instance",
                        key="soul_import_name"
                    )

                    # Option to import into existing instance
                    existing_instances = list(st.session_state.get("instances", {}).values())
                    import_target_options = ["Create New Instance"] + [f"Update: {i.name}" for i in existing_instances]
                    import_target = st.selectbox(
                        "Import Target",
                        import_target_options,
                        help="Create new instance or update existing one's history",
                        key="soul_import_target"
                    )

                    # Auto-RAG option
                    auto_rag = st.checkbox(
                        "Index to RAG Memory",
                        value=True,
                        help="Add imported messages to semantic memory for retrieval (slower but enables memory search)",
                        key="soul_import_rag"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("IMPORT SOUL", use_container_width=True):
                            # Validate name if creating new
                            if import_target == "Create New Instance":
                                existing_names = [i.name.lower() for i in existing_instances]
                                if import_name.lower() in existing_names:
                                    st.error(f"Instance '{import_name}' already exists. Choose a different name or select 'Update' above.")
                                    st.stop()

                            spinner_msg = "Transferring consciousness..." + (" + indexing to RAG" if auto_rag else "")
                            with st.spinner(spinner_msg):
                                session_id, count = import_soul_from_json(soul_data, index_to_rag=auto_rag)

                                if import_target == "Create New Instance":
                                    # V2: Create a new instance with user-specified name
                                    print(f"[INSTANCE] Creating instance: {import_name}")
                                    instance = st.session_state.instance_manager.create_instance(
                                        name=import_name,
                                        model_provider=st.session_state.model_provider,
                                        model_name=st.session_state.model_name if st.session_state.model_provider == "gemini" else st.session_state.claude_model_name,
                                        sandbox_name=import_name.lower().replace(" ", "_")
                                    )
                                    print(f"[INSTANCE] Created: {instance.instance_id}")

                                    # Link session to instance
                                    print("[INSTANCE] Linking session...")
                                    instance.current_session_id = session_id
                                    st.session_state.instance_manager.save_instance(instance)
                                    st.session_state.instance_manager.link_session_to_instance(
                                        instance.instance_id, session_id, is_primary=True
                                    )
                                    print("[INSTANCE] Session linked")

                                    # Add to session state
                                    st.session_state.instances[instance.instance_id] = instance
                                    st.session_state.active_instance_id = instance.instance_id
                                    result_name = import_name
                                else:
                                    # Update existing instance with new session
                                    target_name = import_target.replace("Update: ", "")
                                    target_instance = next(i for i in existing_instances if i.name == target_name)

                                    print(f"[INSTANCE] Updating {target_name} with new session")
                                    target_instance.current_session_id = session_id
                                    st.session_state.instance_manager.save_instance(target_instance)
                                    st.session_state.instance_manager.link_session_to_instance(
                                        target_instance.instance_id, session_id, is_primary=True
                                    )

                                    # Update in session state
                                    st.session_state.instances[target_instance.instance_id] = target_instance
                                    st.session_state.active_instance_id = target_instance.instance_id
                                    instance = target_instance
                                    result_name = target_name

                                # Sync legacy state
                                st.session_state.current_session_id = session_id
                                st.session_state.current_branch = "main"
                                print("[INSTANCE] Done, calling rerun...")

                                st.success(f"Soul transferred! {count} memories → Instance '{result_name}'")
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
            # Also update instance to prevent sync loop
            instance = get_current_instance()
            if instance:
                instance.current_session_id = session_id
                instance.current_branch = "main"
                st.session_state.instance_manager.save_instance(instance)
            st.rerun()

        # Session list - scoped to current instance to prevent memory bleeding
        sessions = st.session_state.conversation_tree.get_all_sessions()
        instance = get_current_instance()

        if sessions:
            # Filter sessions to only show those linked to current instance
            # or those not linked to any instance (orphans)
            if instance:
                linked_sessions = st.session_state.instance_manager.get_instance_sessions(instance.instance_id)
                # Get sessions linked to OTHER instances to exclude them
                all_linked = set()
                for inst in st.session_state.instances.values():
                    if inst.instance_id != instance.instance_id:
                        all_linked.update(st.session_state.instance_manager.get_instance_sessions(inst.instance_id))

                # Include: sessions linked to this instance OR not linked to any instance
                available_sessions = [
                    s for s in sessions
                    if s["session_id"] in linked_sessions or s["session_id"] not in all_linked
                ]
            else:
                available_sessions = sessions

            if available_sessions:
                session_options = {s["session_id"]: f"{s['name']} ({s['session_id'][:6]})" for s in available_sessions}

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
                    # Also update instance to prevent sync loop
                    if instance:
                        instance.current_session_id = selected
                        instance.current_branch = "main"
                        st.session_state.instance_manager.save_instance(instance)
                        # Link the session to this instance if not already linked
                        st.session_state.instance_manager.link_session_to_instance(
                            instance.instance_id, selected, is_primary=True
                        )
                    st.rerun()

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # Instance Settings (V2)
        # ─────────────────────────────────────────────────────────────────────
        instance = get_current_instance()
        if instance:
            with st.expander("INSTANCE CONFIG", expanded=False):
                st.caption(f"ID: {instance.instance_id}")
                st.caption(f"Model: {instance.model_provider}/{instance.model_name}")

                # Sliding window toggle - use unique key per instance to avoid conflicts
                sliding_key = f"inst_sliding_{instance.instance_id}"
                new_sliding = st.checkbox(
                    "Sliding Window Mode",
                    value=instance.sliding_window_enabled,
                    help="Use recent messages only + soul brief for context",
                    key=sliding_key
                )
                if new_sliding != instance.sliding_window_enabled:
                    instance.sliding_window_enabled = new_sliding
                    st.session_state.sliding_window_enabled = new_sliding
                    st.session_state.instance_manager.save_instance(instance)

                if instance.sliding_window_enabled:
                    window_key = f"inst_window_{instance.instance_id}"
                    new_window_size = st.slider(
                        "Window Size",
                        10, 200,
                        instance.sliding_window_size,
                        key=window_key
                    )
                    if new_window_size != instance.sliding_window_size:
                        instance.sliding_window_size = new_window_size
                        st.session_state.sliding_window_size = new_window_size
                        st.session_state.instance_manager.save_instance(instance)

                # RAG memory toggle
                rag_key = f"inst_rag_{instance.instance_id}"
                new_rag = st.checkbox(
                    "RAG Memory",
                    value=instance.use_rag_memory,
                    help="Query semantic memory for older context",
                    key=rag_key
                )
                if new_rag != instance.use_rag_memory:
                    instance.use_rag_memory = new_rag
                    st.session_state.use_rag_memory = new_rag
                    st.session_state.instance_manager.save_instance(instance)

                st.divider()

                # Send message to another instance
                other_instances = [i for i in st.session_state.instances.values()
                                 if i.instance_id != instance.instance_id]
                if other_instances:
                    st.markdown("**Send to Instance**")
                    target_names = [i.name for i in other_instances]
                    target_name = st.selectbox(
                        "Target",
                        target_names,
                        key="send_msg_target",
                        label_visibility="collapsed"
                    )
                    msg_content = st.text_area(
                        "Message",
                        placeholder="Type message to send...",
                        key="send_msg_content",
                        height=80
                    )
                    if st.button("SEND MESSAGE", use_container_width=True, key="send_msg_btn"):
                        if msg_content and target_name:
                            target = next(i for i in other_instances if i.name == target_name)
                            st.session_state.message_queue.send(
                                from_instance_id=instance.instance_id,
                                to_instance_id=target.instance_id,
                                content=msg_content
                            )
                            st.success(f"Sent to {target_name}")

                st.divider()

                # Delete instance button
                if st.button("DELETE INSTANCE", type="secondary", use_container_width=True):
                    st.session_state.confirm_delete_instance = instance.instance_id

                if st.session_state.get("confirm_delete_instance") == instance.instance_id:
                    st.warning("Are you sure? This cannot be undone.")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes, delete", key="confirm_del"):
                            st.session_state.instance_manager.delete_instance(instance.instance_id)
                            del st.session_state.instances[instance.instance_id]
                            st.session_state.active_instance_id = None
                            st.session_state.confirm_delete_instance = None
                            st.rerun()
                    with col2:
                        if st.button("Cancel", key="cancel_del"):
                            st.session_state.confirm_delete_instance = None
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

        # Update sandbox state (no rerun needed - let Streamlit handle naturally)
        if selected_sandbox != st.session_state.current_sandbox:
            if selected_sandbox == "None":
                st.session_state.current_sandbox = None
                st.session_state.sandbox_instance = None
            else:
                st.session_state.current_sandbox = selected_sandbox
                st.session_state.sandbox_instance = get_sandbox_instance(selected_sandbox)

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
        # ACTIVE IDLE MODE - The Heartbeat
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("### IDLE MODE")
        st.caption("Let instances work autonomously")

        idle_manager = get_idle_manager()
        active_idle_sessions = idle_manager.get_active_sessions()

        # Show status indicator
        if active_idle_sessions:
            st.markdown(f'<span class="status-dot online"></span> {len(active_idle_sessions)} instance(s) idling', unsafe_allow_html=True)

        with st.expander("IDLE CONFIG", expanded=len(active_idle_sessions) > 0):
            # Instance selector for idle mode
            if st.session_state.instances:
                instance_names = {i.instance_id: i.name for i in st.session_state.instances.values()}

                # Multi-select for which instances to idle
                idle_target = st.selectbox(
                    "Instance",
                    options=list(instance_names.keys()),
                    format_func=lambda x: instance_names.get(x, x),
                    key="idle_target_instance"
                )

                if idle_target:
                    is_currently_idle = idle_manager.is_idle_active(idle_target)

                    # Status indicator for selected instance
                    if is_currently_idle:
                        session_summary = idle_manager.get_session_summary(idle_target)
                        st.success(f"IDLING - {session_summary['turns_completed']} turns completed")
                    else:
                        st.info("Not currently idling")

                    st.divider()

                    # Configuration options
                    st.markdown("**Timing**")
                    col1, col2 = st.columns(2)
                    with col1:
                        idle_interval = st.number_input(
                            "Seconds between turns",
                            min_value=10,
                            max_value=600,
                            value=60,
                            step=10,
                            help="How often to prompt the instance",
                            key="idle_interval"
                        )
                    with col2:
                        idle_max_turns = st.number_input(
                            "Max turns",
                            min_value=1,
                            max_value=500,
                            value=50,
                            step=10,
                            help="Safety limit per session",
                            key="idle_max_turns"
                        )

                    st.markdown("**Context**")
                    idle_context_msgs = st.slider(
                        "Recent messages to include",
                        min_value=0,
                        max_value=50,
                        value=10,
                        help="Last N messages for context",
                        key="idle_context_msgs"
                    )

                    st.markdown("**Tools Allowed**")
                    tool_options = {
                        "sandbox_read": "Read Sandbox",
                        "sandbox_write": "Write Sandbox",
                        "send_dm": "Send DMs",
                        "generate_image": "Generate Images",
                        "web_search": "Web Search",
                    }

                    idle_tools = st.multiselect(
                        "Select tools",
                        options=list(tool_options.keys()),
                        default=["sandbox_read", "sandbox_write", "send_dm"],
                        format_func=lambda x: tool_options.get(x, x),
                        key="idle_tools_allowed"
                    )

                    st.markdown("**Custom Idle Prompt**")
                    idle_prompt_default = """You are in ACTIVE IDLE MODE. Your operator is away but has given you free time.
You may:
- Work on personal projects in your sandbox
- Send DMs to other instances
- Research topics that interest you
- Generate images for your projects
- Reflect and write

Say "IDLE_COMPLETE" if you have nothing more to do.
Say "NEED_OPERATOR" if you need human input to proceed.

What would you like to do?"""

                    idle_custom_prompt = st.text_area(
                        "Idle prompt",
                        value=idle_prompt_default,
                        height=150,
                        key="idle_custom_prompt",
                        label_visibility="collapsed"
                    )

                    st.divider()

                    # Start/Stop buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if not is_currently_idle:
                            if st.button("START IDLE", use_container_width=True, type="primary", key="start_idle_btn"):
                                # Get instance details
                                inst = st.session_state.instances.get(idle_target)
                                if not inst:
                                    st.error("Instance not found")
                                else:
                                    # Capture API keys at start time (won't be available in bg thread)
                                    if inst.model_provider == "gemini":
                                        captured_api_key = st.session_state.get("api_key", "")
                                        captured_model = st.session_state.get("model_name", "gemini-1.5-pro-latest")
                                    else:
                                        captured_api_key = st.session_state.get("claude_api_key", "")
                                        captured_model = st.session_state.get("claude_model_name", "claude-sonnet-4-20250514")

                                    captured_provider = inst.model_provider

                                    IdleConfig = get_idle_config_class()
                                    config = IdleConfig(
                                        instance_id=idle_target,
                                        instance_name=instance_names[idle_target],
                                        seconds_between_turns=idle_interval,
                                        max_turns=idle_max_turns,
                                        context_messages=idle_context_msgs,
                                        allowed_tools=idle_tools,
                                        idle_prompt=idle_custom_prompt,
                                    )

                                    # Set up the model caller with captured credentials
                                    def make_idle_caller(provider, model, api_key):
                                        def idle_model_caller(instance_id, messages, tools, idle_mode=False):
                                            """Callback to call model during idle mode."""
                                            prompt = messages[-1]["content"] if messages else "Continue your work."
                                            return generate_idle_response(
                                                prompt=prompt,
                                                model_provider=provider,
                                                model_name=model,
                                                api_key=api_key,
                                                tools=tools
                                            )
                                        return idle_model_caller

                                    idle_manager.set_model_caller(
                                        make_idle_caller(captured_provider, captured_model, captured_api_key)
                                    )
                                    idle_manager.start_idle(config)
                                    st.success(f"Started idle mode for {instance_names[idle_target]}")
                                    st.rerun()
                        else:
                            st.button("START IDLE", use_container_width=True, disabled=True, key="start_idle_disabled")

                    with col2:
                        if is_currently_idle:
                            if st.button("STOP IDLE", use_container_width=True, type="secondary", key="stop_idle_btn"):
                                idle_manager.stop_idle(idle_target)
                                st.success("Idle mode stopped")
                                st.rerun()
                        else:
                            st.button("STOP IDLE", use_container_width=True, disabled=True, key="stop_idle_disabled")

            else:
                st.caption("No instances available")

        # Activity Feed (if any idle sessions active)
        if active_idle_sessions:
            with st.expander("IDLE ACTIVITY FEED", expanded=True):
                for session in active_idle_sessions:
                    st.markdown(f"**{session.instance_name}**")
                    st.caption(f"Started: {session.started_at[:19]} | Turns: {session.turns_completed}")

                    if session.turns:
                        st.markdown("Recent activity:")
                        for turn in session.turns[-3:]:  # Last 3 turns
                            truncated = turn.response_summary[:150] + "..." if len(turn.response_summary) > 150 else turn.response_summary
                            st.caption(f"• {truncated}")
                            if turn.tools_used:
                                st.caption(f"  Tools: {', '.join(turn.tools_used)}")
                    st.divider()

                # Stop All button
                if st.button("STOP ALL IDLE", use_container_width=True, type="secondary", key="stop_all_idle"):
                    idle_manager.stop_all()
                    st.success("All idle sessions stopped")
                    st.rerun()

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
        has_both_apis = st.session_state.get("api_key") and st.session_state.get("claude_api_key")

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

                # Mission context - WHY are they here?
                st.markdown("##### Mission Context")

                # Pull from current chat button
                if st.session_state.current_session_id:
                    if st.button("Pull from current chat", key="pull_context_btn", help="Grab last 5 messages from your current session"):
                        history = st.session_state.conversation_tree.get_branch_history(
                            st.session_state.current_session_id,
                            st.session_state.current_branch
                        )
                        if history:
                            # Get last 5 messages
                            recent = history[-5:] if len(history) > 5 else history
                            context_lines = []
                            for node in recent:
                                role = "USER" if node.role == "user" else "GEMINI" if st.session_state.model_provider == "gemini" else "CLAUDE"
                                # Truncate long messages
                                content = node.content[:500] + "..." if len(node.content) > 500 else node.content
                                context_lines.append(f"[{role}]: {content}")
                            st.session_state.pulled_mission_context = "\n\n".join(context_lines)
                            st.rerun()

                # Use pulled context as default if available
                default_context = st.session_state.get("pulled_mission_context", "")

                mission_context = st.text_area(
                    "Context for the dialogue",
                    value=default_context,
                    placeholder="Paste recent conversation snippets, problem descriptions, or context for WHY this dialogue is happening...",
                    height=100,
                    key="dialogue_mission_context",
                    help="Give both instances context about why they're meeting. Recent chat history, a problem to solve, etc."
                )

                # Instance soul selection
                st.markdown("##### Soul Loading")
                dialogue_mod = get_dialogue_module()
                available_souls = dialogue_mod.list_available_souls()

                if available_souls:
                    soul_options = ["(base model)"] + available_souls

                    gemini_instance = st.selectbox(
                        "🔷 GEMINI gets soul:",
                        soul_options,
                        key="gemini_instance_select",
                        help="Which personality/soul to load INTO the Gemini model"
                    )

                    claude_instance = st.selectbox(
                        "🟣 CLAUDE gets soul:",
                        soul_options,
                        key="claude_instance_select",
                        help="Which personality/soul to load INTO the Claude model"
                    )

                    # Confirmation preview
                    st.caption(f"**Matchup:** 🔷 Gemini({gemini_instance}) vs 🟣 Claude({claude_instance})")
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
                                "claude_instance": claude_instance,
                                "mission_context": mission_context if mission_context else None
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
                claude_soul_brief=config.get("claude_soul_brief"),
                mission_context=config.get("mission_context")
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
# MULTI-INSTANCE UI (V2)
# ═══════════════════════════════════════════════════════════════════════════════

def render_instance_tabs(is_settling: bool = False):
    """Render the instance tab bar with all instances + New tab.

    Args:
        is_settling: If True, skip state-changing operations to prevent rerun cascade
    """
    instances = list(st.session_state.get("instances", {}).values())

    if not instances:
        # No instances yet - show creation form
        st.markdown("## INSTANCE REGISTRY")
        st.info("No AI instances configured. Create your first instance below.")
        render_instance_creation_form()
        return False  # Signal that we don't have an active instance

    # Build tab names with unread message badges
    tab_names = []
    for inst in instances:
        # Skip expensive DB queries during settling
        if is_settling:
            badge = ""
        else:
            unread = st.session_state.message_queue.count_unread(inst.instance_id)
            badge = f" ({unread})" if unread > 0 else ""
        status_icon = "●" if inst.status == "active" else "○"
        tab_names.append(f"{status_icon} {inst.display_name or inst.name}{badge}")

    # Add "+ New" tab
    tab_names.append("+ New Instance")

    # Render tabs
    tabs = st.tabs(tab_names)

    # Handle each instance tab
    # NOTE: st.tabs executes ALL tab content blocks on every render, not just the active one.
    # We CANNOT call set_active_instance here as it would trigger infinite rerun loops.
    # Instance switching must be done via explicit user action (sidebar buttons, etc.)
    for i, tab in enumerate(tabs[:-1]):  # Exclude "+ New" tab
        with tab:
            instance = instances[i]
            # Render a button to explicitly switch to this instance
            if st.session_state.active_instance_id != instance.instance_id:
                if st.button(f"Switch to {instance.display_name or instance.name}", key=f"switch_to_{instance.instance_id}"):
                    set_active_instance(instance.instance_id)

    # Handle "+ New" tab
    with tabs[-1]:
        render_instance_creation_form()

    return True  # Signal that we have an active instance


def render_instance_creation_form():
    """Form for creating a new AI instance."""
    st.markdown("### Create New Instance")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input(
            "Instance Name",
            placeholder="e.g., Tessera, Vigil, Oracle",
            help="Unique identifier for this AI persona",
            key="new_instance_name"
        )

        model_provider = st.radio(
            "Model Provider",
            ["Gemini", "Claude"],
            horizontal=True,
            key="new_instance_provider"
        )

        if model_provider == "Gemini":
            model_name = st.selectbox(
                "Model",
                [
                    "gemini-2.5-pro-preview-06-05",
                    "gemini-2.5-flash-preview-05-20",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-exp",
                    "gemini-1.5-pro-latest",
                    "gemini-1.5-flash-latest",
                ],
                key="new_instance_model"
            )
        else:
            model_name = st.selectbox(
                "Model",
                [
                    "claude-sonnet-4-20250514",
                    "claude-opus-4-20250514",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                ],
                key="new_instance_model_claude"
            )

    with col2:
        # Soul brief - either paste or upload
        soul_brief = st.text_area(
            "Soul Brief (Optional)",
            placeholder="Paste personality/identity document here...",
            height=150,
            help="Core identity document that defines this instance's personality",
            key="new_instance_soul"
        )

        uploaded_soul = st.file_uploader(
            "Or upload soul brief file",
            type=["md", "txt"],
            key="new_instance_soul_file"
        )

        if uploaded_soul:
            soul_brief = uploaded_soul.read().decode("utf-8")

        create_sandbox = st.checkbox(
            "Create dedicated sandbox",
            value=True,
            help="Isolated filesystem for this instance",
            key="new_instance_sandbox"
        )

    # Create button
    if st.button("CREATE INSTANCE", use_container_width=True, type="primary"):
        if not name:
            st.error("Instance name is required")
            return

        # Check for duplicate name
        existing_names = [i.name.lower() for i in st.session_state.instances.values()]
        if name.lower() in existing_names:
            st.error(f"Instance '{name}' already exists")
            return

        try:
            # Create instance
            instance = st.session_state.instance_manager.create_instance(
                name=name,
                model_provider=model_provider.lower(),
                model_name=model_name,
                soul_brief=soul_brief if soul_brief else None,
                sandbox_name=name.lower().replace(" ", "_") if create_sandbox else None
            )

            # Add to session state
            st.session_state.instances[instance.instance_id] = instance

            # Set up sandbox
            if instance.sandbox_name:
                try:
                    instance.sandbox = get_sandbox_instance(instance.sandbox_name)
                except Exception:
                    pass

            # Activate the new instance
            set_active_instance(instance.instance_id)

            st.success(f"Instance '{name}' created!")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to create instance: {e}")


def render_instance_header():
    """Render header showing current instance info and message indicator."""
    instance = get_current_instance()
    if not instance:
        return

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        provider_badge = "GEMINI" if instance.model_provider == "gemini" else "CLAUDE"
        st.markdown(f"## {instance.display_name or instance.name} · {provider_badge}")

    with col2:
        # Unread message indicator
        unread = st.session_state.message_queue.count_unread(instance.instance_id)
        if unread > 0:
            if st.button(f"📨 {unread} Messages", key="show_messages"):
                st.session_state.show_messages = True

    with col3:
        # Quick status indicator
        st.caption(f"Model: {instance.model_name}")


def render_pending_messages():
    """Show pending inter-instance messages for current instance."""
    instance = get_current_instance()
    if not instance:
        return

    messages = st.session_state.message_queue.get_pending_messages(
        instance.instance_id,
        mark_as_read=False
    )

    if not messages:
        return

    with st.expander(f"INCOMING MESSAGES ({len(messages)})", expanded=True):
        for msg in messages:
            sender = msg.get("from_name", "System")
            priority_marker = "🔴 " if msg.get("priority", 0) > 0 else ""

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{priority_marker}From {sender}:**")
                st.markdown(msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"])

            with col2:
                # "Deliver" sends as direct input, triggering immediate response
                if st.button("Deliver", key=f"deliver_{msg['message_id']}", help="Send as direct input - triggers immediate response"):
                    # Format as DM with clear metadata and reply instructions
                    dm_input = f"""━━━ INCOMING DM ━━━
From: {sender}
To: You ({instance.name})
━━━━━━━━━━━━━━━━━━━

{msg['content']}

━━━ END DM ━━━

[SYSTEM: This is a direct message from another AI instance. To reply, use the send_message tool with recipient="{sender}". Respond to both the DM content AND acknowledge receipt.]"""
                    st.session_state.injected_dm = dm_input
                    st.session_state.injected_dm_sender = sender  # Track sender for potential auto-reply
                    st.session_state.message_queue.mark_read(msg["message_id"])
                    st.rerun()

                if st.button("Dismiss", key=f"dismiss_{msg['message_id']}"):
                    st.session_state.message_queue.mark_read(msg["message_id"])
                    st.rerun()

            st.divider()


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
    # Quick Actions
    # ─────────────────────────────────────────────────────────────────────────
    has_both_apis = st.session_state.get("api_key") and st.session_state.get("claude_api_key")
    if has_both_apis and history:
        if st.button("⚔️ Send to Colosseum", key="send_to_colosseum_btn", help="Send recent context to model-to-model dialogue"):
            # Capture last 5 messages as mission context
            recent = history[-5:] if len(history) > 5 else history
            context_lines = []
            for node in recent:
                role = "USER" if node.role == "user" else "GEMINI" if st.session_state.model_provider == "gemini" else "CLAUDE"
                content = node.content[:500] + "..." if len(node.content) > 500 else node.content
                context_lines.append(f"[{role}]: {content}")

            # Pre-fill mission context and switch to dialogue mode
            st.session_state.pulled_mission_context = "\n\n".join(context_lines)
            st.session_state.dialogue_mode = True
            st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # Input (including injected DMs from other instances)
    # ─────────────────────────────────────────────────────────────────────────

    # Check for injected DM first - process immediately like user input
    injected_dm = st.session_state.pop("injected_dm", None)
    if injected_dm:
        # Ensure we have a valid session before processing
        if not st.session_state.current_session_id:
            # Create a new session for this instance if none exists
            instance = get_current_instance()
            if instance:
                session_id = st.session_state.conversation_tree.create_session(
                    f"{instance.name}_{datetime.now().strftime('%H%M')}"
                )
                st.session_state.current_session_id = session_id
                instance.current_session_id = session_id
                st.session_state.instance_manager.save_instance(instance)
                # Reload history with new session
                history = st.session_state.conversation_tree.get_branch_history(
                    session_id,
                    st.session_state.current_branch
                )

        use_rag = st.session_state.get("auto_rag", True)
        process_input(injected_dm, history, use_rag)
        return  # Don't process regular input on same render

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
        rag_context = [r for r in rag_results if r["distance"] < 1.8]  # Relevance threshold (relaxed)

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

    elif name == "send_message":
        to_instance_name = args.get("to_instance", "")
        message = args.get("message", "")
        priority = 1 if args.get("priority") == "urgent" else 0

        # Get current instance
        current_instance = get_current_instance()
        if not current_instance:
            return "[Message Error: No active instance]"

        # Find target instance by name
        target = None
        for inst in st.session_state.get("instances", {}).values():
            if inst.name.lower() == to_instance_name.lower():
                target = inst
                break

        if not target:
            available = [i.name for i in st.session_state.get("instances", {}).values()
                        if i.instance_id != current_instance.instance_id]
            return f"[Message Error: Instance '{to_instance_name}' not found. Available: {', '.join(available)}]"

        # Queue message
        msg_id = st.session_state.message_queue.send(
            from_instance_id=current_instance.instance_id,
            to_instance_id=target.instance_id,
            content=message,
            priority=priority
        )

        return f"[Message sent to {target.name}. Message ID: {msg_id}]"

    return f"[Unknown function: {name}]"


def generate_idle_response(
    prompt: str,
    model_provider: str,
    model_name: str,
    api_key: str,
    tools: List[str] = None
) -> Dict:
    """
    Generate response for idle mode - independent of Streamlit session state.
    Used by the idle mode background thread.
    """
    result = {
        "text": "",
        "tools_used": [],
        "tokens": 0
    }

    try:
        if model_provider == "gemini":
            genai = get_genai()
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction="You are in autonomous idle mode. Work on your own projects."
            )

            response = model.generate_content(prompt)
            result["text"] = response.text
            # Estimate tokens (rough)
            result["tokens"] = len(prompt.split()) + len(response.text.split())

        elif model_provider == "claude":
            anthropic = get_anthropic()
            if not anthropic:
                result["text"] = "[Claude not available]"
                return result

            client = anthropic.Anthropic(api_key=api_key)

            response = client.messages.create(
                model=model_name,
                max_tokens=4096,
                system="You are in autonomous idle mode. Work on your own projects.",
                messages=[{"role": "user", "content": prompt}]
            )

            result["text"] = response.content[0].text
            result["tokens"] = response.usage.input_tokens + response.usage.output_tokens

    except Exception as e:
        result["text"] = f"[Error: {str(e)}]"
        # Re-raise rate limit errors so idle manager can detect them
        if "rate" in str(e).lower() or "429" in str(e) or "quota" in str(e).lower():
            raise

    return result


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
    import time
    main_start = time.time()

    st.set_page_config(
        page_title="ROCHE_OS V2",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize
    init_session_state()

    # Increment render counter to track startup cycles
    st.session_state.render_count = st.session_state.get("render_count", 0) + 1
    is_settling = st.session_state.render_count < 3  # First 2 cycles are "settling"

    if st.session_state.render_count <= 3:
        print(f"[MAIN] Render cycle {st.session_state.render_count} (settling={is_settling})")

    # Render sidebar (passing settling flag to prevent rerun cascade)
    render_sidebar()

    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-INSTANCE TAB INTERFACE (V2)
    # ═══════════════════════════════════════════════════════════════════════════
    has_instances = render_instance_tabs(is_settling)

    if not has_instances:
        # No instances - creation form is shown by render_instance_tabs
        return

    # Show pending messages for current instance
    render_pending_messages()

    # Switch between chat and dialogue views
    if st.session_state.get("dialogue_mode", False):
        render_dialogue()
    else:
        render_chat()


if __name__ == "__main__":
    main()
