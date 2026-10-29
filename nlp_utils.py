"""Shared text-processing helpers (used by train.py and app.py)."""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")
SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly")
SUSPICIOUS_TLDS = (".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf", ".gq", ".work", ".zip")

URGENCY_WORDS = [
    "urgent", "immediately", "asap", "act now", "within 24 hours", "expires",
    "suspended", "locked", "final notice", "last chance", "limited time", "verify",
    "confirm your", "update your", "click here", "login", "password", "ssn",
    "bank", "account", "winner", "prize", "claim", "free", "congratulations",
]


def clean_text(text: str) -> str:
    """Normalise raw email/SMS text for the TF-IDF vectoriser."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags
    text = re.sub(r"\d+", " numtoken ", text)      # numbers -> placeholder
    text = re.sub(r"[^a-z\s]", " ", text)          # drop punctuation/symbols
    return re.sub(r"\s+", " ", text).strip()


def extract_indicators(text: str) -> dict:
    """Rule-based red flags shown next to the model prediction in the demo."""
    low = text.lower()
    urls = URL_RE.findall(text)
    flags = []
    if IP_URL_RE.search(text):
        flags.append("Link uses a raw IP address instead of a domain")
    if any(s in low for s in SHORTENERS):
        flags.append("Contains a URL-shortener link (hides the real destination)")
    if any(u.lower().rstrip("/.,)").endswith(SUSPICIOUS_TLDS) or
           any(t + "/" in u.lower() for t in SUSPICIOUS_TLDS) for u in urls):
        flags.append("Link uses a TLD often abused in phishing")
    if re.search(r"https?://[^\s/]*@", text):
        flags.append("Link contains '@' (credential-style URL trick)")
    urgent = sorted({w for w in URGENCY_WORDS if w in low})
    if len(urgent) >= 3:
        flags.append(f"Heavy urgency / pressure language ({len(urgent)} trigger words)")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.3 and len(letters) > 30:
        flags.append("Unusually high share of CAPITAL LETTERS")
    if text.count("!") >= 3:
        flags.append("Excessive exclamation marks")
    return {"urls": urls, "urgency_words": urgent, "flags": flags}
