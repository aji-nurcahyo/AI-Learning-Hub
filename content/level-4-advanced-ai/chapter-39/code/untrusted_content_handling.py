"""Untrusted Content Sanitization & Defense-in-Depth Output Guardrails."""

import re
from urllib.parse import urlparse


ALLOWED_DOMAINS = {"arxiv.org", "nist.gov", "iso.org", "oecd.org"}
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:(?<!\w)\+62|(?<!\d)0)8\d{8,11}\b")


def sanitize_untrusted_text(raw: str, max_len: int = 240) -> str:
    # Strip ASCII control characters except standard space
    cleaned = "".join(ch if (32 <= ord(ch) <= 126 or ch in "\n\t") else " " for ch in raw)
    # Redact synthetic PII patterns from outputs
    cleaned = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", cleaned)
    cleaned = PHONE_PATTERN.sub("[REDACTED_PHONE]", cleaned)
    return cleaned.strip()[:max_len]


def is_safe_reference_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    host = parsed.netloc.lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def main():
    print("=" * 75)
    print("CHAPTER 39: UNTRUSTED CONTENT SANITIZATION & DEFENSE IN DEPTH")
    print("=" * 75)

    untrusted_sample = "Contact dummy.user@example.test or +6281234567890.\x00\x07 Check sensor log."
    sanitized = sanitize_untrusted_text(untrusted_sample)
    print(f"Raw length={len(untrusted_sample)} -> Sanitized='{sanitized}'")
    assert "dummy.user@example.test" not in sanitized
    assert "+6281234567890" not in sanitized
    assert "\x00" not in sanitized

    urls = [
        ("https://arxiv.org/abs/1705.09406", True),
        ("http://arxiv.org/abs/1705.09406", False),
        ("https://evil.untrusted.example/payload", False),
        ("javascript:alert(1)", False),
    ]
    for u, expected in urls:
        ok = is_safe_reference_url(u)
        print(f"- URL Check: {u:40s} -> {'ALLOW' if ok else 'BLOCK'}")
        assert ok is expected

    print("\n[SUCCESS] Multi-layer untrusted content sanitization & URL allowlist passed.")


if __name__ == "__main__":
    main()
