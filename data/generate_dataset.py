"""
generate_dataset.py
--------------------
Generates a synthetic (but realistically-patterned) fake/real news dataset
so the full ML pipeline can be trained and demoed immediately, with zero
external downloads.

IMPORTANT FOR PRODUCTION / RESUME-WORTHY RESULTS:
--------------------------------------------------
This synthetic data is great for getting the pipeline working end-to-end
right now. For a stronger, resume-worthy model, swap this out for a real
public dataset such as the Kaggle "Fake and Real News Dataset"
(search: kaggle fake-and-real-news-dataset, files Fake.csv / True.csv).
Just drop those two files in data/ and adjust train.py's DATA loading
section (instructions are in the README).
"""

import random
import csv
import os

random.seed(42)

# ---------------------------------------------------------------------------
# Vocabulary / template banks
# ---------------------------------------------------------------------------

TOPICS = [
    "the economy", "the new vaccine rollout", "the presidential election",
    "climate policy", "the stock market", "a local school district",
    "the national health system", "a tech company merger", "the housing market",
    "immigration policy", "a celebrity's private life", "the space program",
    "a major sports league", "the central bank's interest rate decision",
    "a new social media app", "the education budget", "a natural disaster",
    "a viral health trend", "government spending", "a political scandal",
]

REAL_SOURCES = [
    "Reuters", "the Associated Press", "a Treasury Department spokesperson",
    "officials at the Department of Health", "the university's research team",
    "a spokesperson for the company", "the mayor's office", "the World Health Organization",
    "the Bureau of Labor Statistics", "court documents", "a peer-reviewed study published this week",
    "the company's quarterly earnings report", "the national weather service",
    "election commission officials", "the central bank's press release",
]

REAL_VERBS = [
    "reported", "announced", "confirmed", "said in a statement",
    "released data showing", "told reporters", "clarified in a press briefing",
    "published findings indicating", "confirmed in an official statement",
]

REAL_TEMPLATES = [
    "{source} {verb} that {topic} saw a {pct}% change over the past {period}, according to figures released on {day}.",
    "In a statement on {day}, {source} {verb} new developments regarding {topic}, citing data collected over the past {period}.",
    "{source} {verb} an update on {topic} on {day}, noting that further analysis is expected within the next {period}.",
    "According to {source}, {topic} is expected to shift by roughly {pct}% following changes announced on {day}.",
    "{source} {verb} details about {topic} during a briefing held on {day}, based on a review spanning the last {period}.",
    "A report from {source} released {day} outlines how {topic} has changed, with officials attributing the shift to policy adjustments made over the last {period}.",
    "Data compiled by {source} and reviewed by independent analysts shows {topic} moved by about {pct}% in the last {period}.",
]

FAKE_HOOKS = [
    "SHOCKING:", "YOU WON'T BELIEVE WHAT HAPPENED WITH", "BREAKING (they don't want you to know):",
    "EXPOSED:", "The TRUTH about", "LEAKED:", "URGENT WARNING about",
    "Doctors HATE this fact about", "What the media is HIDING about",
    "This one secret about", "ALERT:",
]

FAKE_CLAIMS = [
    "will completely change everything overnight",
    "is being covered up by the mainstream media",
    "was secretly manipulated by anonymous insiders",
    "has been linked to a massive hidden conspiracy",
    "will destroy the economy within days, insiders claim",
    "is a scheme designed to control the public",
    "has shocked experts who refuse to speak on record",
    "is the biggest secret no one is talking about",
    "was staged, according to unnamed sources",
    "proves what 'they' have been hiding for years",
]

FAKE_FILLERS = [
    "Share this before it gets DELETED!!!",
    "Wake up, people — nobody is reporting this!",
    "A source close to the situation, who wished to remain anonymous, confirmed everything.",
    "This changes EVERYTHING you thought you knew.",
    "The government does not want this getting out.",
    "Experts are too scared to comment publicly.",
    "You need to see this before it's too late.",
    "Mainstream media REFUSES to cover this story.",
]

FAKE_TEMPLATES = [
    "{hook} {topic} {claim}. {filler}",
    "{hook} {topic}! {claim}. {filler}",
    "People are furious after learning that {topic} {claim}. {filler}",
    "It's finally out: {topic} {claim}. {filler}",
    "{hook} the real story behind {topic}. Sources say it {claim}. {filler}",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "last week", "earlier this month"]
PERIODS = ["week", "month", "quarter", "six months", "year"]


def rand_pct():
    return round(random.uniform(0.5, 24.9), 1)


def make_real():
    t = random.choice(REAL_TEMPLATES)
    return t.format(
        source=random.choice(REAL_SOURCES),
        verb=random.choice(REAL_VERBS),
        topic=random.choice(TOPICS),
        pct=rand_pct(),
        day=random.choice(DAYS),
        period=random.choice(PERIODS),
    )


def make_fake():
    t = random.choice(FAKE_TEMPLATES)
    return t.format(
        hook=random.choice(FAKE_HOOKS),
        topic=random.choice(TOPICS),
        claim=random.choice(FAKE_CLAIMS),
        filler=random.choice(FAKE_FILLERS),
    )


def generate(n_per_class=1500):
    rows = []
    seen = set()
    while len([r for r in rows if r[1] == 1]) < n_per_class:
        text = make_real()
        if text not in seen:
            seen.add(text)
            rows.append((text, 1))
    while len([r for r in rows if r[1] == 0]) < n_per_class:
        text = make_fake()
        if text not in seen:
            seen.add(text)
            rows.append((text, 0))
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "news_dataset.csv")
    rows = generate(1500)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])  # label: 1 = REAL, 0 = FAKE
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
