#!/usr/bin/env python3
"""
MD to Soul Transfer JSON Converter

Converts scraped Claude conversation markdown files into Gemini API compatible
JSON format for import into ROCHE_OS soul transfer system.

Output format matches what app.py import_soul_from_json() expects.
"""

import re
import json
import sys
from pathlib import Path
from datetime import datetime


def parse_metadata(content: str) -> dict:
    """Extract metadata from the markdown header."""
    metadata = {
        'source': 'Claude.ai Export',
        'format': 'gemini_api_compatible'
    }

    # Title (first H1)
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    if title_match:
        metadata['title'] = title_match.group(1).strip()

    # Date fields
    date_patterns = {
        'created': r'\*\*Created:\*\*\s*(.+?)(?:\s{2}|\n)',
        'updated': r'\*\*Updated:\*\*\s*(.+?)(?:\s{2}|\n)',
        'exported': r'\*\*Exported:\*\*\s*(.+?)(?:\s{2}|\n)',
    }

    for key, pattern in date_patterns.items():
        match = re.search(pattern, content)
        if match:
            metadata[key] = match.group(1).strip()

    # Link
    link_match = re.search(r'\*\*Link:\*\*\s*\[.+?\]\((.+?)\)', content)
    if link_match:
        metadata['source_url'] = link_match.group(1).strip()

    return metadata


def extract_thinking(response_content: str) -> tuple[str | None, str]:
    """
    Separate thinking blocks from response content.
    Returns (thinking_content, cleaned_response)
    """
    thinking = None
    cleaned = response_content

    # Match thinking blocks in code fences (```plaintext or ````plaintext)
    thinking_pattern = r'`{3,4}plaintext\s*\n(.*?)\n`{3,4}'
    thinking_match = re.search(thinking_pattern, response_content, re.DOTALL)

    if thinking_match:
        thinking = thinking_match.group(1).strip()
        cleaned = re.sub(thinking_pattern, '', response_content, flags=re.DOTALL).strip()

    return thinking, cleaned


def parse_conversations(content: str) -> list[dict]:
    """
    Parse all prompt/response pairs from the markdown.
    Returns Gemini API compatible format.
    """
    messages = []

    # Pattern to find all prompts and responses
    turn_pattern = r'## (Prompt|Response):\s*\n(\d{1,2}\.\d{1,2}\.\d{4},\s*\d{1,2}:\d{2}:\d{2})\s*\n(.*?)(?=## (?:Prompt|Response):|\Z)'

    matches = re.findall(turn_pattern, content, re.DOTALL)

    for turn_type, timestamp, turn_content in matches:
        turn_content = turn_content.strip()

        if turn_type == 'Prompt':
            # User message
            messages.append({
                'role': 'user',
                'parts': [{'text': turn_content}],
                '_timestamp': timestamp  # Preserved for reference, ignored by import
            })

        elif turn_type == 'Response':
            thinking, cleaned_content = extract_thinking(turn_content)

            msg = {
                'role': 'model',
                'parts': [{'text': cleaned_content}],
                '_timestamp': timestamp
            }

            # Store thinking separately - can be prepended or kept as metadata
            if thinking:
                msg['_thinking'] = thinking

            messages.append(msg)

    return messages


def md_to_soul_transfer(
    md_path: str,
    output_path: str = None,
    model: str = None,
    include_thinking: bool = True
) -> dict:
    """
    Convert a markdown conversation export to ROCHE_OS compatible JSON.

    Args:
        md_path: Path to the input markdown file
        output_path: Optional path for JSON output (defaults to same name with .json)
        model: Optional model identifier (e.g., 'claude-3.5-sonnet')
        include_thinking: If True, prepends thinking to assistant responses

    Returns:
        The generated soul transfer document as a dict
    """
    md_path = Path(md_path)

    if not md_path.exists():
        raise FileNotFoundError(f"Input file not found: {md_path}")

    content = md_path.read_text(encoding='utf-8')

    # Parse components
    metadata = parse_metadata(content)
    messages = parse_conversations(content)

    # Optionally integrate thinking into response text
    if include_thinking:
        for msg in messages:
            if msg['role'] == 'model' and '_thinking' in msg:
                thinking = msg['_thinking']
                original = msg['parts'][0]['text']
                # Format: <thinking>...</thinking>\n\nresponse
                msg['parts'][0]['text'] = f"<thinking>\n{thinking}\n</thinking>\n\n{original}"

    # Clean up internal fields for export
    history = []
    for msg in messages:
        clean_msg = {
            'role': msg['role'],
            'parts': msg['parts']
        }
        history.append(clean_msg)

    # Build soul transfer document (Gemini API compatible)
    soul_doc = {
        'metadata': {
            'exported_at': datetime.now().isoformat(),
            'source': metadata.get('source', 'Claude.ai Export'),
            'message_count': len(history),
            'format': 'gemini_api_compatible',
            'original_title': metadata.get('title'),
            'original_created': metadata.get('created'),
            'original_updated': metadata.get('updated'),
            'source_url': metadata.get('source_url'),
        },
        'history': history
    }

    if model:
        soul_doc['metadata']['model'] = model

    # Clean None values from metadata
    soul_doc['metadata'] = {k: v for k, v in soul_doc['metadata'].items() if v is not None}

    # Output
    if output_path is None:
        output_path = md_path.with_suffix('.json')
    else:
        output_path = Path(output_path)

    output_path.write_text(
        json.dumps(soul_doc, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    # Stats
    user_count = sum(1 for m in history if m['role'] == 'user')
    model_count = sum(1 for m in history if m['role'] == 'model')

    print(f"Soul transfer document created: {output_path}")
    print(f"  Title: {metadata.get('title', 'Unknown')}")
    print(f"  Messages: {len(history)} ({user_count} user, {model_count} model)")
    print(f"  Format: Gemini API compatible (ready for ROCHE_OS import)")

    return soul_doc


def main():
    if len(sys.argv) < 2:
        print("Usage: python md_to_soul_transfer.py <input.md> [output.json] [options]")
        print("\nOptions:")
        print("  --model MODEL        Set model identifier (e.g., claude-3.5-sonnet)")
        print("  --no-thinking        Don't include thinking blocks in output")
        print("\nExamples:")
        print("  python md_to_soul_transfer.py TESSERA.md")
        print("  python md_to_soul_transfer.py TESSERA.md soul.json --model claude-3.5-sonnet")
        print("  python md_to_soul_transfer.py TESSERA.md --no-thinking")
        sys.exit(1)

    md_path = sys.argv[1]
    output_path = None
    model = None
    include_thinking = True

    # Parse args
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--model' and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == '--no-thinking':
            include_thinking = False
            i += 1
        else:
            output_path = args[i]
            i += 1

    try:
        md_to_soul_transfer(md_path, output_path, model, include_thinking)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
