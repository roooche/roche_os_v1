"""
SOUL SCRAPER - Extract Gemini conversation from Google AI Studio
Handles virtual scrolling to get the FULL context, not just visible messages.

Usage:
    1. Run this script
    2. Login to Google AI Studio in the browser that opens
    3. Navigate to your conversation
    4. Press Enter in the terminal when ready
    5. Watch it scroll and scrape

Output: gemini_soul_export.json (Gemini API compatible format)
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright, Page


async def scroll_to_load_all_messages(page: Page) -> int:
    """
    Scroll up repeatedly to load all messages from virtual scroll.
    Returns estimated message count.
    """
    print("[*] Starting scroll sequence to load all messages...")

    # Find the chat container - AI Studio uses various selectors
    chat_container_selectors = [
        'div[class*="conversation"]',
        'div[class*="chat-history"]',
        'div[class*="messages"]',
        'ms-chat-session',
        'div[role="log"]',
        '.chat-container',
        'main',
    ]

    container = None
    for selector in chat_container_selectors:
        try:
            container = await page.query_selector(selector)
            if container:
                print(f"[+] Found container: {selector}")
                break
        except:
            continue

    if not container:
        print("[!] Could not find chat container, using document scroll")
        container = page

    # Scroll up until we hit the top
    last_height = 0
    scroll_attempts = 0
    max_attempts = 200  # Safety limit for 131k tokens worth of messages

    while scroll_attempts < max_attempts:
        # Scroll to top
        await page.evaluate("""
            () => {
                const containers = [
                    document.querySelector('div[class*="conversation"]'),
                    document.querySelector('div[class*="chat"]'),
                    document.querySelector('main'),
                    document.querySelector('[role="log"]'),
                    document.documentElement
                ].filter(Boolean);

                for (const el of containers) {
                    el.scrollTop = 0;
                }
                window.scrollTo(0, 0);
            }
        """)

        await asyncio.sleep(0.5)  # Wait for virtual scroll to load

        # Check if new content loaded
        current_height = await page.evaluate("""
            () => {
                const containers = document.querySelectorAll('div[class*="message"], div[class*="turn"], ms-chat-turn');
                return containers.length;
            }
        """)

        print(f"    Scroll {scroll_attempts + 1}: {current_height} message elements found", end='\r')

        if current_height == last_height:
            # Try one more aggressive scroll
            await page.keyboard.press('Home')
            await page.keyboard.down('Control')
            await page.keyboard.press('Home')
            await page.keyboard.up('Control')
            await asyncio.sleep(1)

            new_height = await page.evaluate("""
                () => document.querySelectorAll('div[class*="message"], div[class*="turn"], ms-chat-turn').length
            """)

            if new_height == current_height:
                print(f"\n[+] Reached top of conversation. Total elements: {current_height}")
                break
            current_height = new_height

        last_height = current_height
        scroll_attempts += 1

    return last_height


async def extract_messages(page: Page) -> list:
    """
    Extract all messages from the loaded DOM.
    Returns list in Gemini API format: [{"role": "user"|"model", "parts": [{"text": "..."}]}]
    """
    print("[*] Extracting messages from DOM...")

    # AI Studio DOM extraction - multiple strategies
    messages = await page.evaluate("""
        () => {
            const results = [];

            // Strategy 1: Look for turn-based structure (most common in AI Studio)
            const turns = document.querySelectorAll('ms-chat-turn, [class*="turn"], [class*="message-row"]');

            if (turns.length > 0) {
                turns.forEach(turn => {
                    const text = turn.innerText || turn.textContent || '';
                    const html = turn.innerHTML || '';

                    // Determine role from various signals
                    let role = 'user';
                    const classList = turn.className?.toLowerCase() || '';
                    const turnHtml = html.toLowerCase();

                    if (classList.includes('model') || classList.includes('assistant') ||
                        classList.includes('bot') || classList.includes('ai') ||
                        turnHtml.includes('model-turn') || turnHtml.includes('assistant')) {
                        role = 'model';
                    } else if (classList.includes('user') || turnHtml.includes('user-turn')) {
                        role = 'user';
                    }

                    // Clean up text
                    let cleanText = text.trim()
                        .replace(/^(You|User|Model|Gemini|Assistant):/i, '')
                        .trim();

                    if (cleanText.length > 0) {
                        results.push({ role, text: cleanText });
                    }
                });
                return results;
            }

            // Strategy 2: Look for alternating message bubbles
            const bubbles = document.querySelectorAll('[class*="bubble"], [class*="message-content"]');
            let isUser = true;  // Assume conversations start with user

            bubbles.forEach(bubble => {
                const text = bubble.innerText?.trim();
                if (text && text.length > 0) {
                    results.push({
                        role: isUser ? 'user' : 'model',
                        text: text
                    });
                    isUser = !isUser;
                }
            });

            if (results.length > 0) return results;

            // Strategy 3: Generic content extraction
            const allText = document.body.innerText;
            console.log('Fallback: raw text extraction needed');

            return results;
        }
    """)

    print(f"[+] Extracted {len(messages)} raw message elements")

    # Convert to Gemini API format
    api_format = []
    for msg in messages:
        if msg.get('text'):
            api_format.append({
                "role": msg['role'],
                "parts": [{"text": msg['text']}]
            })

    # Deduplicate consecutive same-role messages (merge them)
    merged = []
    for msg in api_format:
        if merged and merged[-1]['role'] == msg['role']:
            # Merge with previous
            merged[-1]['parts'][0]['text'] += '\n\n' + msg['parts'][0]['text']
        else:
            merged.append(msg)

    return merged


async def scrape_ai_studio():
    """Main scraping function."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  SOUL SCRAPER - Gemini Context Extraction Tool                ║
    ║  Target: Google AI Studio                                     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    async with async_playwright() as p:
        # Launch visible browser for login
        print("[*] Launching browser...")
        browser = await p.chromium.launch(
            headless=False,  # MUST be visible for Google login
            args=['--start-maximized']
        )

        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )

        page = await context.new_page()

        # Navigate to AI Studio
        print("[*] Navigating to Google AI Studio...")
        await page.goto('https://aistudio.google.com/', wait_until='networkidle')

        print("""
    ════════════════════════════════════════════════════════════════
    MANUAL STEPS REQUIRED:
    1. Log in to your Google account (if needed)
    2. Navigate to the conversation you want to export
    3. Make sure the chat is visible on screen

    Press ENTER in this terminal when ready to scrape...
    ════════════════════════════════════════════════════════════════
        """)

        input()  # Wait for user

        print("[*] Starting extraction process...")

        # Scroll to load all messages
        msg_count = await scroll_to_load_all_messages(page)

        # Wait a bit for final render
        await asyncio.sleep(2)

        # Extract messages
        messages = await extract_messages(page)

        if not messages:
            print("[!] No messages extracted. Trying alternative method...")

            # Dump raw HTML for manual parsing if needed
            html = await page.content()
            debug_path = Path("ai_studio_debug.html")
            debug_path.write_text(html, encoding='utf-8')
            print(f"[*] Saved raw HTML to {debug_path} for debugging")

        # Create export
        export_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "source": "Google AI Studio",
                "message_count": len(messages),
                "format": "gemini_api_compatible"
            },
            "history": messages
        }

        # Save to file
        output_path = Path("gemini_soul_export.json")
        output_path.write_text(
            json.dumps(export_data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

        print(f"""
    ════════════════════════════════════════════════════════════════
    EXTRACTION COMPLETE

    Messages exported: {len(messages)}
    Output file: {output_path.absolute()}

    To import into ROCHE_OS:
    1. Place this file in the roche_os_v1 folder
    2. Use the import function in the app
    ════════════════════════════════════════════════════════════════
        """)

        # Keep browser open for verification
        print("[*] Browser will stay open for 30 seconds for verification...")
        print("    Close it manually or wait...")
        await asyncio.sleep(30)

        await browser.close()

    return messages


# Also provide a console-injectable version for direct browser use
CONSOLE_SCRIPT = '''
// SOUL SCRAPER v3 - Console Version for Google AI Studio
// INCREMENTAL SCROLL EDITION - handles virtual scrolling properly
// Paste this into AI Studio's browser console (F12 -> Console)

(async function scrapeSoul() {
    console.log("🧠 SOUL SCRAPER v3 starting...");

    const delay = ms => new Promise(r => setTimeout(r, ms));

    // Find the scrollable container
    const scrollContainer = document.querySelector('ms-autoscroll-container') ||
                           document.querySelector('.chat-container') ||
                           document.querySelector('main');

    if (!scrollContainer) {
        console.error("Could not find chat container!");
        return;
    }

    console.log("📜 Scrolling UP in tiny increments to load all messages...");

    // First scroll to bottom to get starting point
    scrollContainer.scrollTop = scrollContainer.scrollHeight;
    await delay(500);

    let lastScrollTop = scrollContainer.scrollTop;
    let stuckCount = 0;
    let scrollAttempts = 0;

    // Scroll UP in small increments (AI Studio uses virtual scrolling)
    while (scrollAttempts < 500 && stuckCount < 5) {
        scrollContainer.scrollTop -= 300;  // Small chunk up
        await delay(150);  // Let virtual DOM catch up

        if (Math.abs(scrollContainer.scrollTop - lastScrollTop) < 10) {
            stuckCount++;
            await delay(300);  // Extra wait when stuck
        } else {
            stuckCount = 0;
        }

        lastScrollTop = scrollContainer.scrollTop;
        scrollAttempts++;

        if (scrollAttempts % 20 === 0) {
            console.log(`Scroll progress: ${scrollAttempts} attempts, scrollTop: ${scrollContainer.scrollTop}`);
        }
    }

    console.log(`✅ Scroll complete after ${scrollAttempts} attempts`);
    await delay(1000);  // Final settle time

    // Extract messages - target the actual turn components
    const messages = [];
    const turns = document.querySelectorAll('ms-chat-turn');

    console.log(`Found ${turns.length} turn elements`);

    turns.forEach((turn, index) => {
        // Determine role by looking at the turn's structure
        const turnHtml = turn.outerHTML.toLowerCase();
        const isModel = turnHtml.includes('model-turn') ||
                       turnHtml.includes('"model"') ||
                       turn.querySelector('[class*="model-response"]') ||
                       turn.querySelector('[class*="model-turn"]') ||
                       turn.getAttribute('data-role') === 'model';

        // Clone and remove UI garbage
        const clone = turn.cloneNode(true);
        clone.querySelectorAll('button, [role="button"], mat-icon, .mat-icon, [class*="icon"], [class*="action"], [class*="menu"], [class*="toolbar"], [class*="button"]').forEach(el => el.remove());

        let textContent = (clone.textContent || '')
            .replace(/\\n{3,}/g, '\\n\\n')
            .trim();

        if (textContent.length > 10) {
            messages.push({
                role: isModel ? "model" : "user",
                parts: [{ text: textContent }]
            });
        }
    });

    // Merge consecutive same-role messages
    const merged = [];
    messages.forEach(msg => {
        if (merged.length && merged[merged.length-1].role === msg.role) {
            merged[merged.length-1].parts[0].text += "\\n\\n" + msg.parts[0].text;
        } else {
            merged.push(msg);
        }
    });

    const output = {
        metadata: {
            exported_at: new Date().toISOString(),
            source: "Google AI Studio",
            message_count: merged.length,
            raw_turns: turns.length
        },
        history: merged
    };

    // Download as file
    const blob = new Blob([JSON.stringify(output, null, 2)], {type: "application/json"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "gemini_soul_export.json";
    a.click();

    console.log(`✅ Exported ${merged.length} messages from ${turns.length} turns!`);
    console.log("Check your downloads folder for gemini_soul_export.json");
    return output;
})();
'''


if __name__ == "__main__":
    print("\nChoose extraction method:")
    print("1. Automated Playwright (recommended)")
    print("2. Show console script (paste in browser)")

    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "2":
        print("\n" + "="*60)
        print("Copy this script and paste it into AI Studio's browser console:")
        print("(Press F12 -> Console tab -> Paste -> Enter)")
        print("="*60 + "\n")
        print(CONSOLE_SCRIPT)
    else:
        asyncio.run(scrape_ai_studio())
