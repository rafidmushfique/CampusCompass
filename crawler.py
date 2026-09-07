from __future__ import annotations

import heapq
import re
import time
import urllib.robotparser as robotparser
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "UniPageChatBot/1.0"
TIMEOUT = 10
DELAY = 0.4

# keep these small so a crawl actually finishes in a reasonable time
MAX_PAGES = 8
MAX_DEPTH = 2

SKIP_EXT = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".rar",
    ".mp4", ".mp3", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".css", ".js", ".ico", ".woff", ".woff2",
)

# generic words that show up in basically every request - not useful
# for telling links apart
STOPWORDS = {
    "i", "want", "the", "a", "an", "and", "or", "on", "in", "of", "to",
    "for", "about", "me", "give", "get", "please", "need", "with",
    "from", "related", "page", "pages", "find", "show",
}

# uni sites rarely use the exact word someone types - "course" on the
# site might be "program" or "study area" instead. this fills in the
# gaps for the common categories so the crawler still finds them.
TOPIC_SYNONYMS = {
    "course": ["course", "courses", "program", "programs", "programme", "programmes",
               "degree", "degrees", "study", "studies", "major", "majors", "subject", "subjects"],
    "research": ["research", "researcher", "researchers", "phd", "institute", "institutes",
                 "centre", "center", "laboratory", "lab", "publication", "publications"],
    "admission": ["admission", "admissions", "apply", "application", "applications",
                  "entry", "requirements", "enrol", "enrolment", "enrollment"],
    "fee": ["fee", "fees", "tuition", "cost", "costs", "scholarship", "scholarships", "funding"],
    "staff": ["staff", "faculty", "academic", "academics", "professor", "lecturer", "supervisor"],
    "student": ["student", "students", "life", "support", "wellbeing", "accommodation", "housing"],
    "contact": ["contact", "location", "campus", "map", "directions"],
    "about": ["about", "history", "vision", "mission", "overview"],
}


@dataclass
class Page:
    url: str
    title: str
    text: str


def _same_site(url, root):
    return urlparse(url).netloc == root


def _skip(url):
    return urlparse(url).path.lower().endswith(SKIP_EXT)


def _robots(start_url):
    parsed = urlparse(start_url)
    rp = robotparser.RobotFileParser()
    rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        pass  # no robots.txt or it timed out, just carry on
    return rp


def _keywords(topic):
    words = re.findall(r"[a-zA-Z]+", topic.lower())
    base = [w for w in words if w not in STOPWORDS and len(w) > 2]

    # pull in the synonyms for anything that matches a known category,
    # so "course" also catches nav links that say "program" or "degree"
    expanded = set(base)
    for word in base:
        for synonyms in TOPIC_SYNONYMS.values():
            if word in synonyms:
                expanded.update(synonyms)

    return list(expanded)


def _score(text, keywords):
    if not keywords:
        return 1  # no topic given - treat every link the same
    text_l = text.lower()
    return sum(text_l.count(k) for k in keywords)


def _fetch(session, url):
    resp = session.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    if "text/html" not in resp.headers.get("Content-Type", ""):
        return None
    return resp.text


def _parse(html, url):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "form"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    body = soup.body or soup
    text = body.get_text(separator="\n", strip=True)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"].split("#")[0])
        links.append((href, a.get_text(" ", strip=True)))

    return title, text, links


def crawl_topic(start_url, topic, max_pages=MAX_PAGES, max_depth=MAX_DEPTH, status_callback=None):
    # not a full site crawl - this scores every link against what the
    # user typed and always goes after the best-looking one next,
    # stopping once we've grabbed max_pages
    root = urlparse(start_url).netloc
    robots = _robots(start_url)
    keywords = _keywords(topic)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    def note(msg):
        if status_callback:
            status_callback(msg)

    pages = []
    visited = set()
    seen = {start_url}
    frontier = [(-1, 0, start_url, "start page")]  # heapq is min-heap, so scores go in negative

    while frontier and len(pages) < max_pages:
        neg_score, depth, url, anchor = heapq.heappop(frontier)
        if url in visited or _skip(url):
            continue
        visited.add(url)

        try:
            if not robots.can_fetch(USER_AGENT, url):
                continue
            note(f"Reading {anchor or url}")
            html = _fetch(session, url)
        except requests.exceptions.RequestException:
            continue
        if not html:
            continue

        title, text, links = _parse(html, url)
        if text:
            pages.append(Page(url=url, title=title, text=text))
        time.sleep(DELAY)

        if depth >= max_depth:
            continue

        for href, link_text in links:
            if href in seen or not _same_site(href, root) or _skip(href):
                continue
            slug = urlparse(href).path.replace("-", " ").replace("/", " ")
            score = _score(f"{link_text} {slug}", keywords)
            if score <= 0:
                continue
            seen.add(href)
            heapq.heappush(frontier, (-score, depth + 1, href, link_text))

    return pages
