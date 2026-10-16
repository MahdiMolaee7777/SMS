"""Step 2 of the specification: clean the text and remove unnecessary noise.

The cleaning logic is implemented as a scikit-learn transformer so that it is
**part of the persisted pipeline**.  Training and inference therefore always
apply byte-for-byte the same transformation - a common source of silent
train/serve skew in student projects.

Design decisions
----------------
* Noisy entities (URLs, e-mail addresses, phone numbers, money amounts, other
  numbers) are replaced by *placeholder tokens* instead of being deleted.
  "Click here http://x.y" must still look spammy after cleaning, so the fact
  that a URL was present is preserved as the feature ``__url__``.
* Placeholders use underscores because scikit-learn's default token pattern
  ``(?u)\\b\\w\\w+\\b`` treats ``_`` as a word character, so ``__url__`` is kept
  as a single token ("url" alone would be dropped by ``min_df`` semantics and
  ``__`` boundaries keep it unique).
* Unicode is normalised (NFKC) and remaining non-ASCII characters are removed.
  The SMS Spam Collection contains a handful of mojibake characters
  (e.g. ``£`` rendered as ``ú``) that otherwise create duplicate vocabulary.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

from sklearn.base import BaseEstimator, TransformerMixin

from . import config

# --- Regular expressions ---------------------------------------------------

_URL_RE = re.compile(
    r"""(?ix)
    (?:
        \b(?:https?://|ftp://|www\d?\.)   # explicit scheme or www.
        [^\s<>"']+                        # followed by anything non-blank
      | \b[a-z0-9][a-z0-9.\-]*\.(?:com|net|org|edu|gov|info|biz|co|uk|us|ly|me|tv|cc|ru|in)
        (?:/[^\s<>"']*)?                  # bare domain + optional path
    )
    """
)

_EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

#: Phone numbers as they appear in the corpus: 7+ digits, optionally grouped by
#: spaces/dashes/dots, optionally with a leading international prefix.
_PHONE_RE = re.compile(r"(?:(?<=\s)|^)(?:\+?\d[\d\s\-().]{6,}\d)(?=\s|$)")

#: Currency amounts: "£1.50", "$500", "150p", "10,000 pounds" (symbol part only).
_MONEY_RE = re.compile(r"(?i)(?:[$£€¥]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:p|pence|pounds|dollars|euros|usd|gbp)\b)")

_NUMBER_RE = re.compile(r"\d+")

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_WHITESPACE_RE = re.compile(r"\s+")
_MULTISPACE_TOKEN_RE = re.compile(r"(?:\s*__\w+__\s*){2,}")


def clean_text(text: str) -> str:
    """Return the normalised form of a raw SMS body.

    The function is deterministic and cheap, which matters because it runs on
    every prediction request.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Unicode normalisation repairs mojibake such as "ú1.20" / "fine‚" before
    #    any pattern matching happens.
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase so "FREE" and "free" share one feature.
    text = text.lower()

    # 3. Replace noisy entities with explicit placeholder tokens. E-mails are
    #    handled before URLs so that "user@example.com" is not split by the
    #    bare-domain rule.
    text = _EMAIL_RE.sub(f" {config.EMAIL_TOKEN} ", text)
    text = _URL_RE.sub(f" {config.URL_TOKEN} ", text)
    text = _PHONE_RE.sub(f" {config.PHONE_TOKEN} ", text)
    text = _MONEY_RE.sub(f" {config.MONEY_TOKEN} ", text)
    text = _NUMBER_RE.sub(f" {config.NUMBER_TOKEN} ", text)

    # 4. Drop leftover non-ASCII characters (emoji, stray symbols, encoding junk).
    text = _NON_ASCII_RE.sub(" ", text)

    # 5. Collapse whitespace and remove repeated placeholder runs so that
    #    "£5 £5 £5" does not become "__money__ __money__ __money__".
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTISPACE_TOKEN_RE.sub(lambda m: f" {m.group(0).strip().split()[0]} ", text)
    return text.strip()


class TextCleaner(BaseEstimator, TransformerMixin):
    """scikit-learn transformer wrapper around :func:`clean_text`.

    Being a transformer (and not a pre-processing step performed *before*
    ``fit``) means the cleaning rules are stored inside the pickled pipeline and
    cannot drift between training and serving.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def fit(self, X: Sequence[str], y: Iterable[int] | None = None) -> "TextCleaner":
        self._validate_input(X)
        return self

    def transform(self, X: Sequence[str]) -> list[str]:
        self._validate_input(X)
        if not self.enabled:
            return [str(item) for item in X]
        return [clean_text(item) for item in X]

    @staticmethod
    def _validate_input(X: Sequence[str]) -> None:
        if X is None:
            raise ValueError("TextCleaner received None instead of a sequence of strings.")
        if isinstance(X, str):
            raise ValueError(
                "TextCleaner expects a sequence of strings (e.g. a list), not a single string."
            )
