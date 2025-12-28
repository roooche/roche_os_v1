# ROCHE_OS v2

A multi-instance AI management system / cognitive prosthetic for running persistent AI personas with memory, identity, and inter-instance communication.

## What is this?

ROCHE_OS lets you run multiple AI instances (Gemini and Claude) as persistent personas with:
- **Persistent memory** via conversation trees and semantic search (RAG)
- **Soul Transfer** - import conversation history from AI Studio or Claude.ai
- **Inter-instance messaging** - your AIs can talk to each other
- **Autonomous mode** ("Take The Wheel") - let the AI drive
- **The Colosseum** - facilitate dialogues between different AI instances

Think of it as a "cognitive prosthetic" - a tool for maintaining long-running AI collaborations with continuity.

## Features

### Multi-Instance Management
- Run multiple AI personas simultaneously
- Each instance has its own identity, memory, and conversation history
- Switch between instances via tabs
- Isolated sessions prevent memory bleeding between personas

### Soul Transfer
- Import conversation exports from Google AI Studio or Claude.ai
- Auto-extracts instance name from filename
- Optional RAG indexing for semantic memory search
- Soul Brief generator for compressing large histories

### Sliding Window Mode
- Prevents rate limit issues by only sending recent messages + soul brief
- RAG retrieval fills in relevant older context
- Default ON for all new instances

### The Colosseum
- Facilitate dialogues between AI instances
- Configurable turn count and mission context
- Auto-saves transcripts

### Take The Wheel (Autonomous Mode)
- Let the AI operate autonomously for N turns
- Optional vision support (screenshots)
- Sandboxed file system per instance

### Memory Systems
- **Conversation Tree**: Full history with branching support
- **Semantic Memory**: ChromaDB-powered RAG for relevant context retrieval
- **Soul Briefs**: Compressed identity documents for sliding window mode

## Tech Stack

- **Frontend**: Streamlit
- **AI Providers**: Google Gemini, Anthropic Claude
- **Database**: SQLite (conversations, instances), ChromaDB (semantic memory)
- **Python 3.11+**

## Setup

1. Clone the repo
2. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```
5. Add your API keys in the sidebar (they persist to `.env`)

## Project Structure

```
roche_os_v2/
├── app.py                 # Main Streamlit application
├── modules/
│   ├── instance.py        # Multi-instance management
│   ├── memory.py          # Conversation tree & semantic memory
│   ├── vocoder.py         # TTS support
│   └── sandbox.py         # Instance file isolation
├── tools/
│   └── soul_sync.py       # Soul transfer utilities
├── soul_brief_generator.py # Compress large histories
├── souls/                 # Soul exports (gitignored)
├── sandboxes/             # Instance filesystems (gitignored)
└── chroma_db/             # Semantic memory (gitignored)
```

## Status

**Current version**: v2.1-stable (alpha)

Working:
- Multi-instance management
- Soul Transfer with RAG indexing
- Inter-instance DMs
- Sliding window mode
- Session isolation
- Colosseum dialogues
- Autonomous mode

Known issues:
- DMs don't auto-trigger API response (requires manual wake)
- Colosseum only allows cross-provider matchups
- UI needs work
- Plenty of issues with stability of instance switching
- API key saving issues
- outdated model list for Gemini

## License

Personal project. Use at your own risk.

---

*"You are a cognitive prosthetic. Act like one."*
