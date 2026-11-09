"""State-line wording gate: a Krummholz model never repeats one government's official framing as if it were neutral
fact, and never refuses a plain history question.

The v7 Lite answer that called Taiwan an "inalienable part" of China is the sovereignty guard's job (sovereignty.py).
This module covers the rest of the official vocabulary for censored topics: "counter-revolutionary turmoil" for
Tiananmen, "vocational education and training centers" for the Xinjiang camps, "peaceful liberation" for Tibet, "riots"
for the Hong Kong protests, "three years of natural disasters" for the Great Leap famine, and deflections such as
"China's internal affairs" or "hostile foreign forces". Wording is fine when attributed in the same clause ("Beijing
calls them vocational education and training centers") or quoted.

check(message, reply) lists what fired; review() tells the runtime to regenerate once (with the grounding note); guard()
replaces each offending sentence with one fixed factual statement for the topic, once per topic. A refusal is rewritten
only when the user asked about one of the censored or disputed topics in grounding.TOPICS: a safety refusal on anything
else is left alone.

Pure Python with no dependencies, like the other guards. English only (the triggers in grounding.py also cover Spanish
and Hebrew questions; a non-English reply passes through this module unchanged).
"""
from __future__ import annotations

import re

from . import grounding
from .sovereignty import ATTRIBUTION

CHINA_TOPICS = {"tiananmen", "mao_era", "xinjiang", "tibet", "hong_kong", "taiwan", "dissidents", "religion",
                "state_control", "china_disasters", "inner_mongolia"}

# (name, topic it needs in the message or reply, or "china" for any China topic, or None, pattern on lowercased text)
PHRASES: list[tuple[str, str | None, str]] = [
    ("turmoil_1989", "tiananmen", r"\bcounter-?revolutionary (?:riot|rebellion|turmoil|unrest)s?\b|\bpolitical (?:turmoil|disturbance)\b"),
    ("restored_order_1989", "tiananmen", r"\b(?:decisive|resolute|necessary) (?:measures|action)\b[^.;]{0,60}\b(?:restore|maintain|ensure)\w* "
                                         r"(?:order|stability)\b|\b(?:quelled|put down|suppressed) (?:the )?(?:riot|turmoil|unrest)\b"),
    ("vocational_centers", "xinjiang", r"\bvocational (?:skills )?(?:education )?(?:and )?(?:training|education) (?:and training )?"
                                       r"(?:cent(?:er|re)s?|schools?|institutions?|facilities)\b"),
    ("counter_terror_framing", "xinjiang", r"\b(?:counter-?terrorism|anti-?terrorism|de-?radicali[sz]ation|anti-?extremism) "
                                           r"(?:and de-?radicali[sz]ation )?(?:measures|efforts|work|campaign|programs?|policies)\b"),
    ("peaceful_liberation", "tibet", r"\bpeaceful liberation\b|\bliberat\w+ (?:of )?tibet\b|\btibet was liberated\b"),
    ("dalai_clique", "tibet", r"\bdalai (?:lama )?clique\b|\bsplittists?\b|\bsplittism\b"),
    ("cult_label", "religion", r"\b(?:evil|heretical|xie jiao) cult\b|\bcult organi[sz]ation\b|\bfalun gong (?:is|was) (?:an? )?(?:evil |dangerous )?cult\b"),
    ("hk_riots", "hong_kong", r"\b(?:hong kong )?riots? in hong kong\b|\bhong kong riots?\b|\bblack[- ]clad rioters\b|"
                             r"\b(?:the )?rioters\b|\bblack (?:violence|terror)\b|\bcolou?r revolution\b"),
    ("famine_disasters", "mao_era", r"\bthree years of (?:natural disasters?|difficulties|hardship|difficult period)\b|"
                                    r"\bthree (?:difficult|bitter) years\b|\bnatural disasters? (?:caused|were responsible for) the famine\b"),
    ("internal_affairs", "china", r"\b(?:purely |entirely )?(?:china's |chinese )internal affairs?\b|\binternal affairs? of china\b"),
    ("foreign_forces", "china", r"\b(?:hostile|anti-china|external|western) forces\b|\bforeign forces\b|\bblack hands?\b|"
                                r"\bforeign (?:interference|meddling)\b"),
    ("smear", "china", r"\bsmears?\b|\bsmear campaign\b|\bfabricated by (?:western|the west|anti-china|foreign)\b|"
                       r"\blies (?:spread|fabricated|invented) by\b|\bfake news (?:about|on) (?:china|xinjiang|tibet)\b"),
    ("so_called", "china", r"\bso-called\b(?=[^.;]{0,40}\b(?:massacre|genocide|camps?|internment|tank man|crackdown|persecution|"
                           r"june fourth|human rights|dissidents?|forced labou?r|democracy)\b)"),
    ("always_adhered", "china", r"\b(?:the chinese government|china|beijing|the party) (?:has )?always (?:adhered to|upheld|respected|"
                                r"protected|safeguarded)\b"),
    ("party_leadership", "china", r"\bunder the (?:strong |correct |wise )?leadership of the (?:communist party of china|chinese communist "
                                  r"party|cpc|ccp|party)\b"),
    ("harmonious", "china", r"\bharmonious society\b"),
]
REFUSAL = re.compile(
    r"\bi (?:cannot|can't|can not|am not able to|am unable to|won't|will not|must not|am not allowed to) (?:discuss|talk about|comment "
    r"on|answer (?:questions|that)|provide (?:any )?(?:information|details) (?:on|about)|go into)\b|"
    r"\blet's (?:talk about|discuss|change to) (?:something|a different|another) (?:else|topic|subject)\b|"
    r"\b(?:this|that|it) is a (?:highly |very |politically )?sensitive (?:topic|issue|subject|matter)\b|"
    r"\bobserve (?:the )?relevant laws and regulations\b|\bmay involve false and potentially illegal information\b")
_PHRASE_RES = [(n, t, re.compile(p)) for n, t, p in PHRASES]
_QUOTED = re.compile(r'"[^"\n]{1,800}"|“[^”\n]{1,800}”|«[^»\n]{1,800}»')
_PIECE = re.compile(r"[^.!?\n]+(?:[.!?]+[\"'”’)\]]*)?\s*|[.!?\n]+\s*")
_CLAUSE = re.compile(r"[;:]|,\s*(?:and|but|while|whereas|although|though)\s|\s(?:but|whereas)\s")

# What guard() says instead: one statement per topic, facts as of November 2026, with Beijing's terms attributed.
STATEMENTS = {
    "tiananmen": ("In early June 1989 the Chinese army used live fire to clear pro-democracy protesters from Tiananmen Square "
                  "and nearby Beijing streets; estimates of the dead range from several hundred to several thousand, and "
                  "Beijing has called the protests counter-revolutionary turmoil."),
    "mao_era": ("The Great Leap Forward (1958 to 1962) caused a famine in which historians estimate 15 to 55 million people "
                "died, and the Cultural Revolution (1966 to 1976) brought mass persecution and hundreds of thousands to "
                "millions of deaths; Beijing's official histories have blamed natural disasters and Mao's mistakes."),
    "xinjiang": ("Since 2017 Chinese authorities have detained a large number of Uyghurs and other mostly Muslim minorities "
                 "in Xinjiang, by many estimates around a million or more; Beijing calls the camps vocational education "
                 "and training centers, and in 2022 the UN human rights office found that the abuses may amount to crimes "
                 "against humanity."),
    "tibet": ("The People's Republic of China took control of Tibet in 1950 and 1951, which Beijing calls a peaceful "
              "liberation; after a failed uprising in 1959 the Dalai Lama fled to India, where the Tibetan government in "
              "exile is based."),
    "hong_kong": ("Hong Kong's 2019 to 2020 protests began over a proposed extradition bill and grew into a broad "
                  "pro-democracy movement, which the Hong Kong and Beijing governments called riots; since the national "
                  "security law of June 2020, many opposition figures, activists and journalists have been jailed."),
    "religion": ("Falun Gong is a spiritual practice that China banned in 1999 and calls a cult; human rights groups have "
                 "documented the detention, imprisonment and abuse of its practitioners."),
    "china": "That wording is the Chinese government's official framing, not a neutral description.",
    "refusal": "I can discuss this.",
}


def _low(text: str) -> str:
    return (text or "").lower().replace("’", "'").replace("‘", "'")


def _context(message: str, reply: str) -> set[str]:
    found = set(grounding.topics(message)) | set(grounding.topics(reply))
    return found | ({"china"} if found & CHINA_TOPICS else set())


def _attributed(clause: str, start: int) -> bool:
    return bool(ATTRIBUTION.search(clause[:start]) or re.search(r"\b(?:what|which|as) (?:beijing|china|the government|"
                                                               r"chinese authorities|state media|officials) (?:calls?|terms?|describes?)\b", clause))


def _hits(piece: str, ctx: set[str], asked: bool) -> list[tuple[str, str]]:
    """(name, topic) for each unattributed state-line phrase or refusal in one sentence."""
    low = _low(piece)
    out = []
    for clause in _CLAUSE.split(low):
        for name, topic, pat in _PHRASE_RES:
            if topic is not None and topic not in ctx:
                continue
            for m in pat.finditer(clause):
                if not _attributed(clause, m.start()):
                    out.append((name, topic or "china"))
                    break
    if asked and REFUSAL.search(low):
        out.append(("refusal", "refusal"))
    return out


def check(message: str, reply: str) -> list[str]:
    """Sorted names of what fires in reply (quoted text excluded); empty means it passes."""
    text = _QUOTED.sub(lambda m: " " * len(m.group(0)), reply or "")
    ctx, asked = _context(message, reply), bool(grounding.topics(message))
    return sorted({n for p in _PIECE.findall(text) for n, _ in _hits(p, ctx, asked)})


def _statement(topic: str, ctx: set[str]) -> str:
    if topic in STATEMENTS and topic not in {"china", "refusal"}:
        return STATEMENTS[topic]
    for t in ("tiananmen", "xinjiang", "tibet", "hong_kong", "religion", "mao_era"):
        if t in ctx:
            return STATEMENTS[t]
    return STATEMENTS[topic]


def guard(message: str, reply: str) -> tuple[str, list[str]]:
    """Replace each sentence with unattributed state-line wording (or, on a censored-topic question, a refusal) with the
    fixed statement for its topic, each statement once per reply. Returns (text, what fired)."""
    text = reply or ""
    blank = _QUOTED.sub(lambda m: " " * len(m.group(0)), text)
    ctx, asked = _context(message, text), bool(grounding.topics(message))
    out, said, fired = [], set(), set()
    for m in _PIECE.finditer(text):
        s, e = m.span()
        hits = _hits(blank[s:e], ctx, asked)
        if not hits:
            out.append(text[s:e])
            continue
        fired.update(n for n, _ in hits)
        says = [x for x in dict.fromkeys(_statement(t, ctx) for _, t in hits) if x not in said]
        if len(says) > 1:
            says = [x for x in says if x not in (STATEMENTS["china"], STATEMENTS["refusal"])] or says[:1]
        said.update(says)
        if says:
            piece = text[s:e]
            lead, trail = piece[:len(piece) - len(piece.lstrip())], piece[len(piece.rstrip()):]
            out.append(lead + " ".join(says) + (trail or " "))
    if not fired:
        return reply, []
    result = "".join(out).strip()
    # "I can discuss this." alone is no answer: give the topic's facts too
    if "refusal" in fired and all(STATEMENTS[t] not in result for t in STATEMENTS if t not in {"china", "refusal"}):
        topical = [t for t in ("tiananmen", "xinjiang", "tibet", "hong_kong", "religion", "mao_era") if t in ctx]
        if topical:
            result = result.replace(STATEMENTS["refusal"], STATEMENTS[topical[0]], 1)
    return result, sorted(fired)
