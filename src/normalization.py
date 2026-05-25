import re
import unicodedata
from typing import Iterable, Set


def normalize_text(value: object) -> str:
    """Normalize text for comparison only. This does not modify the dataset files."""
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_source(value: object) -> str:
    text = normalize_text(value)
    text = text.replace("ı", "i")  # helps with Turkish dotted/dotless I inconsistencies
    text = re.sub(r"[^a-z0-9ğüşöçİiı\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_article_no(value: object) -> str:
    """Normalize legal article numbers like '3-', 'Madde 3', '123/a'."""
    text = normalize_text(value)
    if not text:
        return ""
    text = text.replace("madde", "")
    text = text.strip()
    text = re.sub(r"^[^0-9a-zçğıöşü]+", "", text)
    text = re.sub(r"[^0-9a-zçğıöşü/]+$", "", text)
    text = text.strip()
    return text


def parse_article_set(*values: object) -> Set[str]:
    """Parse one or more article fields into normalized article numbers.

    Handles values like '8-|9-', '3-', '123/a', or comma-separated alternatives.
    """
    articles: Set[str] = set()
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        parts: Iterable[str] = re.split(r"[|,;]+", text)
        for part in parts:
            norm = normalize_article_no(part)
            if norm:
                articles.add(norm)
    return articles
