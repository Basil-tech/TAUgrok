import string

# 52 letters (upper + lower case)
ENG_LETTERS = list(string.ascii_letters)

# 24 letters
GRK_LETTERS = [
    "α",
    "β",
    "γ",
    "δ",
    "ε",
    "ζ",
    "η",
    "θ",
    "ι",
    "κ",
    "λ",
    "μ",
    "ν",
    "ξ",
    "ο",
    "π",
    "ρ",
    "σ",
    "τ",
    "υ",
    "φ",
    "χ",
    "ψ",
    "ω",
]

# 21 Hebrew letters
HEB_LETTERS = [
    "א",
    "ב",
    "ג",
    "ד",
    "ה",
    "ו",
    "ז",
    "ח",
    "ט",
    "י",
    "כ",
    "ל",
    "מ",
    "נ",
    "ס",
    "ע",
    "פ",
    "צ",
    "ק",
    "ר",
    "ש",
    "ת",
]

LETTERS = ENG_LETTERS + GRK_LETTERS + HEB_LETTERS

EOS = "<EOS>"
EQ = "="
DIV = "/"

SPECIAL_CHARS = [EQ, DIV, EOS]

SYMBOLS = LETTERS + SPECIAL_CHARS

P = 97
