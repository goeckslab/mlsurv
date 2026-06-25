"""Text helpers for report generation — pluralization and title-casing."""


def pluralize(word: str) -> str:
    """Pluralize an English noun with common irregular handling.

    Rules applied in order:
    1. All-caps acronyms (e.g. "OS", "PFS") → unchanged
    2. Ends in -sis → -ses (metastasis → metastases)
    3. Ends in -s/-x/-z/-sh/-ch → +es
    4. Consonant + y → -ies (study → studies)
    5. Default → +s
    """
    if not word:
        return word
    if word.isupper() and word.isalpha():
        return word
    lower = word.lower()
    if lower.endswith("sis"):
        return word[:-3] + ("SES" if word[-3:].isupper() else "ses")
    if lower.endswith(("s", "x", "z", "sh", "ch")):
        return word + "es"
    if lower.endswith("y") and len(word) > 1 and lower[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def title_if_lower(word: str) -> str:
    """Title-case a word only if it is entirely lowercase.

    Preserves acronyms ("OS", "PFS"), mixed-case ("mOS"), and already
    capitalized words ("Death") unchanged.
    """
    if not word:
        return word
    if word.islower():
        return word.title()
    return word
