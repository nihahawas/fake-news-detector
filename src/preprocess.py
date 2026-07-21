"""
preprocess.py
-------------
Text cleaning utilities for the fake news detector.

Uses NLTK's stopword list / lemmatizer when available, but falls back to a
built-in stopword list and a light stemmer-free cleanup so the project still
runs on machines where NLTK data hasn't been downloaded yet.
"""

import re
import string

# ---------------------------------------------------------------------------
# Try to use NLTK if it's installed and its data is downloaded; otherwise
# fall back gracefully so the pipeline never breaks.
# ---------------------------------------------------------------------------
_USE_NLTK = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))

    try:
        _lemmatizer = WordNetLemmatizer()
        _lemmatizer.lemmatize("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        _lemmatizer = WordNetLemmatizer()

    _USE_NLTK = True
except Exception:
    _USE_NLTK = False

# Fallback stopword list (covers the vast majority of common English stopwords)
_FALLBACK_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who
who's whom why why's with won't would wouldn't you you'd you'll you're you've
your yours yourself yourselves
""".split())

STOPWORDS = STOPWORDS if _USE_NLTK else _FALLBACK_STOPWORDS

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Lowercases, strips URLs/HTML/punctuation/numbers, removes stopwords, and
    (if NLTK is available) lemmatizes each token.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    if _USE_NLTK:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


if __name__ == "__main__":
    sample = "SHOCKING!!! Visit http://example.com — the TRUTH about the economy is finally OUT!"
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
