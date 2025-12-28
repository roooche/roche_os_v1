"""
ROCHE_OS_V1 - Tools Module
The Eye of Providence (screenshots) + The Scavenger (web scraping).

Heavy imports (mss, PIL, BeautifulSoup) are lazy-loaded for faster startup.
"""

import base64
import re
import asyncio
from io import BytesIO
from typing import Optional, Tuple, List
from urllib.parse import urlparse

# Lazy-loaded modules (heavy imports deferred until first use)
_mss = None
_PIL_Image = None
_BeautifulSoup = None


def _get_mss():
    global _mss
    if _mss is None:
        import mss
        _mss = mss
    return _mss


def _get_pil_image():
    global _PIL_Image
    if _PIL_Image is None:
        from PIL import Image
        _PIL_Image = Image
    return _PIL_Image


def _get_beautifulsoup():
    global _BeautifulSoup
    if _BeautifulSoup is None:
        from bs4 import BeautifulSoup
        _BeautifulSoup = BeautifulSoup
    return _BeautifulSoup


class EyeOfProvidence:
    """
    Screen capture tool for multimodal context injection.
    One click to show Gemini what you're looking at.
    """

    def __init__(self):
        self._sct = None  # Lazy init

    @property
    def sct(self):
        """Lazy-load mss screen capture."""
        if self._sct is None:
            self._sct = _get_mss().mss()
        return self._sct

    def capture_primary_monitor(
        self,
        quality: int = 85,
        max_size: Tuple[int, int] = (1920, 1080)
    ) -> Tuple[str, bytes]:
        """
        Capture the primary monitor and return base64-encoded image.
        Returns: (base64_string, raw_bytes)
        """
        Image = _get_pil_image()

        # Get primary monitor (index 1, 0 is "all monitors combined")
        monitor = self.sct.monitors[1]

        # Capture
        screenshot = self.sct.grab(monitor)

        # Convert to PIL Image
        img = Image.frombytes(
            "RGB",
            (screenshot.width, screenshot.height),
            screenshot.rgb
        )

        # Resize if too large (save tokens/bandwidth)
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Convert to JPEG bytes
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        img_bytes = buffer.getvalue()

        # Encode to base64
        b64_string = base64.b64encode(img_bytes).decode("utf-8")

        return b64_string, img_bytes

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        quality: int = 85
    ) -> Tuple[str, bytes]:
        """Capture a specific region of the screen."""
        Image = _get_pil_image()

        region = {"left": x, "top": y, "width": width, "height": height}
        screenshot = self.sct.grab(region)

        img = Image.frombytes(
            "RGB",
            (screenshot.width, screenshot.height),
            screenshot.rgb
        )

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        img_bytes = buffer.getvalue()

        b64_string = base64.b64encode(img_bytes).decode("utf-8")

        return b64_string, img_bytes

    def get_monitors_info(self) -> List[dict]:
        """Get info about all connected monitors."""
        return [
            {
                "index": i,
                "left": m["left"],
                "top": m["top"],
                "width": m["width"],
                "height": m["height"]
            }
            for i, m in enumerate(self.sct.monitors)
        ]


class Scavenger:
    """
    Web scraper with headless browser support.
    Bypass the "I can't read this link" limitation.
    """

    # URL pattern for detection
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    )

    # Sites that need JS rendering
    JS_REQUIRED_DOMAINS = {
        "twitter.com",
        "x.com",
        "reddit.com",
        "linkedin.com",
        "facebook.com",
        "instagram.com"
    }

    def __init__(self):
        self._playwright = None
        self._browser = None

    @staticmethod
    def detect_urls(text: str) -> List[str]:
        """Detect URLs in text."""
        urls = Scavenger.URL_PATTERN.findall(text)
        # Normalize
        return [
            url if url.startswith("http") else f"https://{url}"
            for url in urls
        ]

    @staticmethod
    def needs_js_rendering(url: str) -> bool:
        """Check if URL needs JavaScript rendering."""
        try:
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix
            domain = domain.replace("www.", "")
            return domain in Scavenger.JS_REQUIRED_DOMAINS
        except Exception:
            return False

    async def _ensure_browser(self):
        """Lazy-load Playwright browser."""
        if self._browser is None:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)

    async def scrape_with_js(self, url: str, timeout: int = 30000) -> str:
        """Scrape URL with full JavaScript rendering."""
        await self._ensure_browser()

        page = await self._browser.new_page()

        try:
            await page.goto(url, timeout=timeout, wait_until="networkidle")

            # Wait for dynamic content
            await page.wait_for_timeout(2000)

            # Get rendered HTML
            content = await page.content()

            return self._extract_text(content, url)

        finally:
            await page.close()

    def scrape_simple(self, url: str, timeout: int = 30) -> str:
        """Scrape URL without JS rendering (faster)."""
        import urllib.request
        from urllib.error import URLError, HTTPError

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read().decode("utf-8", errors="ignore")
                return self._extract_text(content, url)
        except (URLError, HTTPError) as e:
            return f"[Scraping Error: {str(e)}]"

    def _extract_text(self, html: str, url: str) -> str:
        """Extract readable text from HTML."""
        BeautifulSoup = _get_beautifulsoup()
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        # Special handling for different sites
        domain = urlparse(url).netloc.lower().replace("www.", "")

        if domain in ["twitter.com", "x.com"]:
            return self._extract_twitter(soup)
        elif domain == "arxiv.org":
            return self._extract_arxiv(soup)
        elif domain == "github.com":
            return self._extract_github(soup)
        else:
            return self._extract_generic(soup)

    def _extract_twitter(self, soup) -> str:
        """Extract tweet content."""
        texts = []

        # Tweet text
        for tweet in soup.find_all("article"):
            text = tweet.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)

        if texts:
            return "\n\n---\n\n".join(texts[:10])  # Limit to 10 tweets

        # Fallback
        return self._extract_generic(soup)

    def _extract_arxiv(self, soup) -> str:
        """Extract arXiv paper abstract and metadata."""
        parts = []

        # Title
        title = soup.find("h1", class_="title")
        if title:
            parts.append(f"# {title.get_text(strip=True)}")

        # Authors
        authors = soup.find("div", class_="authors")
        if authors:
            parts.append(f"**Authors:** {authors.get_text(strip=True)}")

        # Abstract
        abstract = soup.find("blockquote", class_="abstract")
        if abstract:
            parts.append(f"**Abstract:**\n{abstract.get_text(strip=True)}")

        if parts:
            return "\n\n".join(parts)

        return self._extract_generic(soup)

    def _extract_github(self, soup) -> str:
        """Extract GitHub README content."""
        # README content
        readme = soup.find("article", class_="markdown-body")
        if readme:
            return readme.get_text(separator="\n", strip=True)

        # Repo description
        about = soup.find("p", class_="f4")
        if about:
            return about.get_text(strip=True)

        return self._extract_generic(soup)

    def _extract_generic(self, soup) -> str:
        """Generic text extraction."""
        # Try to find main content
        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", class_="content") or
            soup.find("div", id="content") or
            soup.body
        )

        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    async def scrape(self, url: str) -> str:
        """
        Smart scrape - automatically chooses method based on URL.
        """
        if self.needs_js_rendering(url):
            try:
                return await self.scrape_with_js(url)
            except Exception as e:
                # Fallback to simple scrape
                return self.scrape_simple(url)
        else:
            return self.scrape_simple(url)

    def scrape_sync(self, url: str) -> str:
        """Synchronous wrapper for scraping."""
        if self.needs_js_rendering(url):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.scrape_with_js(url))
                finally:
                    loop.close()
            except Exception as e:
                return self.scrape_simple(url)
        else:
            return self.scrape_simple(url)

    async def close(self):
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()


def scrape_url(url: str) -> str:
    """Convenience function for synchronous URL scraping."""
    scavenger = Scavenger()
    return scavenger.scrape_sync(url)


def capture_screen() -> Tuple[str, bytes]:
    """Convenience function for screen capture."""
    eye = EyeOfProvidence()
    return eye.capture_primary_monitor()


def detect_and_scrape_urls(text: str) -> List[Tuple[str, str]]:
    """
    Detect URLs in text and scrape them all.
    Returns list of (url, content) tuples.
    """
    urls = Scavenger.detect_urls(text)
    results = []

    scavenger = Scavenger()
    for url in urls:
        try:
            content = scavenger.scrape_sync(url)
            results.append((url, content))
        except Exception as e:
            results.append((url, f"[Error scraping {url}: {e}]"))

    return results


class Periscope:
    """
    The Periscope - Web Search Tool
    Gives Gemini eyes on the live internet.
    No more relying on training data cutoff.
    """

    def __init__(self):
        self._ddg = None

    def _ensure_ddg(self):
        """Lazy load DuckDuckGo search."""
        if self._ddg is None:
            try:
                from duckduckgo_search import DDGS
                self._ddg = DDGS()
            except ImportError:
                raise ImportError("Install duckduckgo-search: pip install duckduckgo-search")

    def _process_query_semantically(self, query: str) -> str:
        """
        Process natural language queries into search-optimized form.

        PHILOSOPHY: Stop being clever. Preserve ALL nouns. Remove only garbage.
        The previous attempts failed because they tried to extract "entities"
        but missed lowercase technical terms like "pandas", "numpy", "react".

        New approach:
        1. Strip question syntax
        2. Keep ALL nouns and technical terms
        3. Reorder so specific terms come first
        """
        import re

        q = query.strip().rstrip('?.')
        q_lower = q.lower()

        # ═══════════════════════════════════════════════════════════════════
        # GARBAGE WORDS - these add nothing to search results
        # ═══════════════════════════════════════════════════════════════════

        garbage = {
            # Question starters
            'what', 'what is', 'what are', 'what\'s', 'whats',
            'who', 'who is', 'who are', 'who\'s',
            'where', 'where is', 'where are', 'where\'s',
            'when', 'when is', 'when did', 'when was',
            'how', 'how do', 'how does', 'how to', 'how can',
            'why', 'why is', 'why does',
            'which', 'whose',
            # Articles and connectors
            'the', 'a', 'an', 'of', 'for', 'to', 'in', 'on', 'at', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'do', 'does', 'did',
            'can', 'could', 'would', 'should',
            'i', 'me', 'my', 'you', 'your',
            # Filler phrases
            'please', 'find', 'tell', 'show', 'give', 'get',
            'want', 'need', 'looking',
            'know', 'about', 'regarding',
            'currently', 'right now',
        }

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Remove question prefixes (keep the meat)
        # ═══════════════════════════════════════════════════════════════════

        question_prefixes = [
            'what is the current', 'what is the latest', 'what is the',
            'what are the', 'what\'s the',
            'can you tell me', 'can you find', 'can you show',
            'i want to know', 'i need to find', 'i\'m looking for',
            'tell me about', 'find me', 'show me', 'give me',
            'do you know', 'does anyone know',
            'where can i find', 'how do i find',
        ]

        for prefix in sorted(question_prefixes, key=len, reverse=True):
            if q_lower.startswith(prefix):
                q = q[len(prefix):].strip()
                q_lower = q.lower()
                break

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Tokenize and identify CONTENT words vs GARBAGE
        # ═══════════════════════════════════════════════════════════════════

        words = q.split()
        content_words = []
        technical_terms = []

        # Known technical/library terms (lowercase but important)
        tech_keywords = {
            'pandas', 'numpy', 'scipy', 'matplotlib', 'tensorflow', 'pytorch',
            'react', 'vue', 'angular', 'node', 'express', 'django', 'flask',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'linux', 'windows', 'macos', 'ubuntu', 'debian',
            'python', 'javascript', 'typescript', 'rust', 'golang', 'java',
            'api', 'sdk', 'cli', 'gui', 'sql', 'nosql', 'graphql',
            'version', 'release', 'stable', 'latest', 'update', 'changelog',
            'library', 'package', 'module', 'framework', 'plugin',
        }

        for word in words:
            word_lower = word.lower().strip('.,;:!?()[]{}"\'-')

            # Skip pure garbage
            if word_lower in garbage:
                continue

            # Technical terms get priority
            if word_lower in tech_keywords:
                technical_terms.append(word_lower)
            # Capitalized words (proper nouns/names)
            elif word[0].isupper() and len(word) > 1:
                content_words.append(word)
            # Numbers (versions, years)
            elif re.match(r'^[\d\.]+$', word_lower):
                content_words.append(word_lower)
            # Everything else that's not garbage
            elif len(word_lower) > 2:
                content_words.append(word_lower)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Build final query - SPECIFIC FIRST, then context
        # ═══════════════════════════════════════════════════════════════════

        # Priority order:
        # 1. Technical terms (pandas, react, etc.) - most specific
        # 2. Proper nouns (Cortical Labs, Voyager)
        # 3. Other content words (version, stable, etc.)

        final_parts = []

        # Add technical terms first (they're usually the actual subject)
        if technical_terms:
            final_parts.extend(technical_terms)

        # Add proper nouns
        proper_nouns = [w for w in content_words if w[0].isupper()]
        if proper_nouns:
            final_parts.extend(proper_nouns)

        # Add remaining content words
        other_words = [w for w in content_words if not w[0].isupper()]
        final_parts.extend(other_words)

        # If we still have nothing, just return cleaned original
        if not final_parts:
            return q

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: Format for DDG
        # ═══════════════════════════════════════════════════════════════════

        result = ' '.join(final_parts)

        # For very specific technical queries, quote the primary subject
        # e.g., "pandas version" not pandas version
        if len(technical_terms) == 1 and len(final_parts) <= 4:
            primary = technical_terms[0]
            rest = ' '.join([w for w in final_parts if w != primary])
            if rest:
                result = f'"{primary}" {rest}'

        return result

    # Domains to deprioritize (forums, Q&A that rarely have direct answers)
    LOW_QUALITY_DOMAINS = {
        'quora.com', 'reddit.com', 'stackexchange.com', 'answers.yahoo.com',
        'zhihu.com', 'ask.com', 'answers.com', 'wikihow.com',
        'medium.com',  # Often clickbait
        'pinterest.com', 'twitter.com', 'x.com', 'facebook.com',
    }

    # Domains to prioritize (authoritative, answer-shaped)
    HIGH_QUALITY_DOMAINS = {
        # Official docs
        'docs.python.org', 'pypi.org', 'npmjs.com', 'crates.io',
        'developer.mozilla.org', 'devdocs.io',
        # Reference
        'wikipedia.org', 'britannica.com', 'github.com',
        # News/Research
        'arxiv.org', 'nature.com', 'science.org', 'reuters.com', 'apnews.com',
        'techcrunch.com', 'arstechnica.com', 'wired.com',
        # Finance
        'finance.yahoo.com', 'bloomberg.com', 'marketwatch.com',
        # Official sources
        '.gov', '.edu', '.org',
    }

    def search(
        self,
        query: str,
        max_results: int = 5,
        time_filter: str = None
    ) -> List[dict]:
        """
        Search the web using DuckDuckGo.

        Args:
            query: Search query
            max_results: Number of results to return
            time_filter: 'd' (day), 'w' (week), 'm' (month), 'y' (year)

        Returns:
            List of {title, url, snippet} dicts
        """
        self._ensure_ddg()

        # Process query semantically to understand intent
        cleaned_query = self._process_query_semantically(query)

        try:
            # Request more results than needed so we can filter/rerank
            raw_results = list(self._ddg.text(
                cleaned_query,
                max_results=max_results * 3,  # Get extra for filtering
                timelimit=time_filter
            ))

            if not raw_results:
                # DDG returned nothing - report this clearly
                return [{
                    "error": f"DuckDuckGo returned no results for query: '{cleaned_query}' (original: '{query}')",
                    "debug": "This may indicate rate limiting, network issues, or an overly specific query."
                }]

            # Parse and score results
            scored_results = []
            for r in raw_results:
                url = r.get("href", r.get("link", ""))
                title = r.get("title", "")
                snippet = r.get("body", r.get("snippet", ""))

                if not url:
                    continue

                # Calculate quality score
                score = 50  # Base score

                # Check domain quality
                url_lower = url.lower()
                for domain in self.HIGH_QUALITY_DOMAINS:
                    if domain in url_lower:
                        score += 30
                        break

                for domain in self.LOW_QUALITY_DOMAINS:
                    if domain in url_lower:
                        score -= 40
                        break

                # Boost if title/snippet contains query terms
                query_terms = cleaned_query.lower().split()
                title_lower = title.lower()
                snippet_lower = snippet.lower()

                for term in query_terms:
                    if len(term) > 3:  # Skip short words
                        if term in title_lower:
                            score += 10
                        if term in snippet_lower:
                            score += 5

                # Penalize very short snippets (often not useful)
                if len(snippet) < 50:
                    score -= 10

                scored_results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "score": score
                })

            # Sort by score (highest first) and return top results
            scored_results.sort(key=lambda x: x["score"], reverse=True)

            return [
                {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
                for r in scored_results[:max_results]
            ]

        except Exception as e:
            import traceback
            return [{
                "error": f"Search failed: {str(e)}",
                "type": type(e).__name__,
                "debug": traceback.format_exc()[:500]
            }]

    def search_news(
        self,
        query: str,
        max_results: int = 5,
        time_filter: str = "w"
    ) -> List[dict]:
        """
        Search news articles.

        Args:
            query: Search query
            max_results: Number of results
            time_filter: 'd' (day), 'w' (week), 'm' (month)
        """
        self._ensure_ddg()

        try:
            results = list(self._ddg.news(
                query,
                max_results=max_results,
                timelimit=time_filter
            ))

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", r.get("link", "")),
                    "snippet": r.get("body", ""),
                    "source": r.get("source", ""),
                    "date": r.get("date", "")
                }
                for r in results
            ]
        except Exception as e:
            return [{"error": str(e)}]

    def search_and_summarize(self, query: str, max_results: int = 3) -> str:
        """
        Search and format results as readable text for the LLM.
        """
        results = self.search(query, max_results=max_results)

        if not results:
            return f"[No results found for: {query}]"

        if "error" in results[0]:
            return f"[Search error: {results[0]['error']}]"

        formatted = [f"**Web Search Results for:** {query}\n"]

        for i, r in enumerate(results, 1):
            formatted.append(f"\n**{i}. {r['title']}**")
            formatted.append(f"   URL: {r['url']}")
            formatted.append(f"   {r['snippet']}")

        return "\n".join(formatted)


def web_search(query: str, max_results: int = 5) -> str:
    """Convenience function for web search."""
    periscope = Periscope()
    return periscope.search_and_summarize(query, max_results)


def news_search(query: str, max_results: int = 5) -> List[dict]:
    """Convenience function for news search."""
    periscope = Periscope()
    return periscope.search_news(query, max_results)


class Oracle:
    """
    The Oracle - Direct Data Fetcher
    For when you need FACTS, not web pages.

    The Periscope finds articles ABOUT things.
    The Oracle gets the actual DATA.
    """

    # CoinGecko API (free, no key needed)
    COINGECKO_API = "https://api.coingecko.com/api/v3"

    # Common crypto symbol mappings
    CRYPTO_IDS = {
        'btc': 'bitcoin', 'bitcoin': 'bitcoin',
        'eth': 'ethereum', 'ethereum': 'ethereum',
        'xrp': 'ripple', 'ripple': 'ripple',
        'sol': 'solana', 'solana': 'solana',
        'ada': 'cardano', 'cardano': 'cardano',
        'doge': 'dogecoin', 'dogecoin': 'dogecoin',
        'dot': 'polkadot', 'polkadot': 'polkadot',
        'matic': 'matic-network', 'polygon': 'matic-network',
        'link': 'chainlink', 'chainlink': 'chainlink',
        'ltc': 'litecoin', 'litecoin': 'litecoin',
    }

    def get_crypto_price(self, symbol: str, currency: str = "usd") -> dict:
        """
        Get current cryptocurrency price from CoinGecko.
        Returns actual data, not web pages.
        """
        import urllib.request
        import json

        symbol_lower = symbol.lower().strip()
        coin_id = self.CRYPTO_IDS.get(symbol_lower, symbol_lower)

        url = f"{self.COINGECKO_API}/simple/price?ids={coin_id}&vs_currencies={currency}&include_24hr_change=true&include_market_cap=true"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))

            if coin_id in data:
                price_data = data[coin_id]
                return {
                    "symbol": symbol.upper(),
                    "coin_id": coin_id,
                    "price": price_data.get(currency, "N/A"),
                    "currency": currency.upper(),
                    "change_24h": price_data.get(f"{currency}_24h_change", "N/A"),
                    "market_cap": price_data.get(f"{currency}_market_cap", "N/A"),
                    "source": "CoinGecko API",
                    "timestamp": "live"
                }
            else:
                return {"error": f"Coin '{symbol}' not found. Try: btc, eth, sol, doge, etc."}

        except Exception as e:
            return {"error": f"CoinGecko API error: {str(e)}"}

    def get_multiple_crypto_prices(self, symbols: List[str], currency: str = "usd") -> List[dict]:
        """Get prices for multiple cryptocurrencies at once."""
        import urllib.request
        import json

        coin_ids = [self.CRYPTO_IDS.get(s.lower().strip(), s.lower().strip()) for s in symbols]
        ids_str = ",".join(coin_ids)

        url = f"{self.COINGECKO_API}/simple/price?ids={ids_str}&vs_currencies={currency}&include_24hr_change=true"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))

            results = []
            for symbol, coin_id in zip(symbols, coin_ids):
                if coin_id in data:
                    price_data = data[coin_id]
                    results.append({
                        "symbol": symbol.upper(),
                        "price": price_data.get(currency, "N/A"),
                        "change_24h": price_data.get(f"{currency}_24h_change", "N/A"),
                    })
                else:
                    results.append({"symbol": symbol.upper(), "error": "Not found"})

            return results

        except Exception as e:
            return [{"error": f"API error: {str(e)}"}]


def get_crypto_price(symbol: str, currency: str = "usd") -> dict:
    """Convenience function for crypto prices."""
    oracle = Oracle()
    return oracle.get_crypto_price(symbol, currency)


def get_live_data(query: str) -> Optional[str]:
    """
    Smart router - detects if query needs live data vs web search.
    Returns formatted data if it's a data query, None if it needs web search.
    """
    import re
    q_lower = query.lower()

    # Crypto price patterns
    crypto_patterns = [
        r'(?:price|value|cost|worth)\s+(?:of\s+)?(\w+)',
        r'(\w+)\s+(?:price|value|cost)',
        r'how much is\s+(\w+)',
        r'what is\s+(\w+)\s+(?:trading|worth|price)',
    ]

    oracle = Oracle()

    for pattern in crypto_patterns:
        match = re.search(pattern, q_lower)
        if match:
            potential_coin = match.group(1)
            if potential_coin in oracle.CRYPTO_IDS or potential_coin in ['bitcoin', 'ethereum', 'crypto']:
                result = oracle.get_crypto_price(potential_coin)
                if "error" not in result:
                    return f"""**{result['symbol']} Live Price**
- Price: ${result['price']:,.2f} {result['currency']}
- 24h Change: {result['change_24h']:.2f}%
- Market Cap: ${result['market_cap']:,.0f}
- Source: {result['source']} (live)"""

    return None  # Not a data query, use web search


class Vocoder:
    """
    The Vocoder - Text-to-Speech Engine
    Give Gemini a voice. Uses Microsoft Edge TTS (free, high quality).
    """

    # Popular voice options
    VOICES = {
        "en-US-GuyNeural": "Guy (US Male)",
        "en-US-JennyNeural": "Jenny (US Female)",
        "en-US-AriaNeural": "Aria (US Female)",
        "en-GB-RyanNeural": "Ryan (UK Male)",
        "en-GB-SoniaNeural": "Sonia (UK Female)",
        "en-AU-WilliamNeural": "William (AU Male)",
        "ja-JP-NanamiNeural": "Nanami (Japanese Female)",
        "de-DE-ConradNeural": "Conrad (German Male)",
        "fr-FR-DeniseNeural": "Denise (French Female)",
        "es-ES-AlvaroNeural": "Alvaro (Spanish Male)",
    }

    DEFAULT_VOICE = "en-US-GuyNeural"

    def __init__(self, voice: str = None):
        self.voice = voice or self.DEFAULT_VOICE

    async def synthesize(self, text: str, output_path: str = None) -> bytes:
        """
        Convert text to speech.

        Args:
            text: Text to convert
            output_path: Optional file path to save audio

        Returns:
            Audio bytes (MP3 format)
        """
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)

        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        if output_path:
            with open(output_path, "wb") as f:
                f.write(audio_data)

        return audio_data

    def synthesize_sync(self, text: str, output_path: str = None) -> bytes:
        """Synchronous wrapper for TTS."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.synthesize(text, output_path))
        finally:
            loop.close()

    @classmethod
    def list_voices(cls) -> dict:
        """Get available voice options."""
        return cls.VOICES


def text_to_speech(text: str, voice: str = None, output_path: str = None) -> bytes:
    """Convenience function for TTS."""
    vocoder = Vocoder(voice=voice)
    return vocoder.synthesize_sync(text, output_path)


class DreamModule:
    """
    The Dream Module - Image Generation
    Gives Gemini the ability to dream in pixels.
    Uses Pollinations.ai (free, no API key needed).
    """

    # Pollinations.ai - Free image generation
    POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}&nologo=true"

    def __init__(self, api_key: str = None):
        self.api_key = api_key  # Not needed for Pollinations

    def generate(
        self,
        prompt: str,
        style: str = None,
        aspect_ratio: str = "1:1"
    ) -> Optional[Tuple[bytes, str]]:
        """
        Generate an image from a text prompt using Pollinations.ai.

        Args:
            prompt: Text description of desired image
            style: Optional style modifier
            aspect_ratio: Image aspect ratio (1:1, 16:9, 9:16)

        Returns:
            Tuple of (image_bytes, mime_type) or None if failed
        """
        import urllib.request
        import urllib.parse

        # Build enhanced prompt
        full_prompt = prompt
        if style:
            full_prompt = f"{prompt}, {style} style, high quality, detailed"

        # Parse aspect ratio to dimensions
        dimensions = {
            "1:1": (1024, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
        }
        width, height = dimensions.get(aspect_ratio, (1024, 1024))

        try:
            # URL encode the prompt
            encoded_prompt = urllib.parse.quote(full_prompt)

            # Build URL
            url = self.POLLINATIONS_URL.format(
                prompt=encoded_prompt,
                width=width,
                height=height
            )

            # Fetch image
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 ROCHE_OS DreamModule"}
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                image_bytes = response.read()

            if image_bytes and len(image_bytes) > 1000:  # Valid image
                return (image_bytes, "image/jpeg")

            return None

        except Exception as e:
            print(f"Image generation error: {e}")
            return None

    def generate_variations(
        self,
        base_image: bytes,
        prompt: str,
        num_variations: int = 2
    ) -> List[Tuple[bytes, str]]:
        """Generate variations - just regenerates with same prompt."""
        results = []
        for _ in range(num_variations):
            result = self.generate(prompt)
            if result:
                results.append(result)
        return results


def generate_image(prompt: str, api_key: str = None, style: str = None) -> Optional[bytes]:
    """Convenience function for image generation."""
    dream = DreamModule(api_key=api_key)
    result = dream.generate(prompt, style=style)
    if result:
        return result[0]  # Return just the bytes
    return None
