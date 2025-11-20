import csv
import re

INPUT_CSV = "data/raw/search_basic.csv"
OUTPUT_CSV = "data/raw/search_basic_clean.csv"

BAD_WORDS = [
    # Core NSFW
    "hentai","sex","sexual","porn","porno","erotic","ecchi",
    "nsfw","xxx","18\\+","adult","lewd","nude","nudity",
    "uncensored","uncensor","breeding",

    # Slang
    "fap","orgasm","boobs","breasts","busty","oppai","pantsu",
    "strip","stripping","milf","stepmom","stepsis",

    # Fetish
    "succubus","psycubus","harem","dominate","dominatrix",
    "bdsm","bondage","fetish","tentacle","mature only","leotard",

    # Problematic
    "loli","lolita","shota","underage",

    # More NSFW
    "pussy","lust","furry","futa","futanari","femboy","traphouse",
    "thigh","thicc","kinks",

    # Strong profanity
    "fuck","fucked","fuckgirl","fuckboy","slut","whore","hooker",

    # Romance-explicit
    "seduce","seduction","sensual","carnal",

    # Adult VN patterns
    "r18","nsfw edition","golden shower","boobjob","handjob",
    "maid cafe",
]

# NSFW word pattern
pattern = re.compile("|".join(BAD_WORDS), re.IGNORECASE)

# ❌ Remove only titles containing non-English SCRIPTS (CJK, Cyrillic, Arabic, etc.)
non_english_scripts = re.compile(
    "[" +
    "\u0400-\u04FF" +  # Cyrillic
    "\u0500-\u052F" +
    "\u0590-\u05FF" +  # Hebrew
    "\u0600-\u06FF" +  # Arabic
    "\u0900-\u097F" +  # Devanagari
    "\u0E00-\u0E7F" +  # Thai
    "\u3040-\u309F" +  # Hiragana
    "\u30A0-\u30FF" +  # Katakana
    "\u4E00-\u9FFF" +  # CJK Unified Ideographs
    "\uAC00-\uD7AF" +  # Hangul syllables
    "\u1100-\u11FF" +  # Hangul Jamo
    "]"
)

removed_bad = 0
removed_non_english = 0
kept = 0

with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as fin, \
     open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as fout:

    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        title = row.get("title", "") or ""

        # 1. Remove NSFW titles
        if pattern.search(title):
            removed_bad += 1
            continue

        # 2. Remove ONLY if title contains any non-Latin script character
        if non_english_scripts.search(title):
            removed_non_english += 1
            continue

        writer.writerow(row)
        kept += 1

print(f"Kept {kept} rows, removed {removed_bad} NSFW, removed {removed_non_english} non-English script titles.")
