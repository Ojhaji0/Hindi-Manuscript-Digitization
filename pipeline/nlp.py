"""
NLP pipeline for Hindi manuscript entity recognition and relation extraction.

Explicit pipeline stages:
  Stage 1 – clean_text        : normalize whitespace, convert Devanagari digits
  Stage 2 – segment           : split cleaned text into paragraphs
  Stage 3 – extract_entities  : named entity recognition via Google NLP API
  Stage 4 – extract_relations : relation extraction via 60+ regex patterns
  Stage 5 – validate_records  : confidence scoring and consistency checks
"""
from __future__ import annotations

import os
import re
from datetime import datetime

try:
    from google.cloud import language_v1
    _NLP_AVAILABLE = True
except ImportError:
    _NLP_AVAILABLE = False

try:
    from hindu_calendar import HinduCalendar as _HC
    _HinduCalendar = _HC
    HINDU_CALENDAR_AVAILABLE = True
except ImportError:
    _HinduCalendar = None
    HINDU_CALENDAR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Regex pattern library (ported from Bahi_Project.ipynb)
# ---------------------------------------------------------------------------

_NAME_PART = r"[A-Za-zऀ-ॿ]+(?:-[A-Za-zऀ-ॿ]+)*"
_MWNAME = rf"{_NAME_PART}(?:\s+{_NAME_PART}){{0,2}}"

PRIMARY_MARKERS_RAW = sorted([
    "प्राग साये", "प्राग", "आदि", "प्रा०मु०", "प्रा0मु0", "प्रा०", "प्रा0",
    "प्रत्", "प्रगोकल", "प्र.", "प्र0", "प्र。", "ठाο", "ठाग्र", "डा0", "मो0",
    "ठाप्पारधी प्र。", "प्रारोज",
], key=len, reverse=True)

_MARKERS_RE = "|".join(re.escape(m.lstrip("'\"• ")) for m in PRIMARY_MARKERS_RAW)
_NAME_CAP = rf"{_MWNAME}(?:\s+स्त्री\s+{_MWNAME})?|{_NAME_PART}स्त्री{_MWNAME}|{_MWNAME}"

PRIMARY_PERSON_MARKERS = re.compile(rf"({_MARKERS_RE})\s*({_NAME_CAP})")
PRIMARY_PERSON_SELF_REL = re.compile(
    rf"({_MARKERS_RE})\s*({_NAME_CAP})\s*(की|का|के)\s*(पतोह‌?|पतोहू‌?|लड़‌?का|लडका|बेटी|लड़की|पत्नी)"
)
IMPLICIT_PRIMARY_BETA_POTA = re.compile(
    rf"^({_MWNAME}(?:\sशर्मा)?)\s+(बेटा|पोता|नाती)\s+({_MWNAME})\s*(?:के)?(?=\s|$)"
)

SELF_REL_TO_ANCESTOR = [
    ("पिता",    re.compile(rf"बेट\s*({_MWNAME})(?:\s*कि)?")),
    ("दादा",    re.compile(rf"पोता\s*({_MWNAME})(?:\s*कि)?")),
    ("परदादा",  re.compile(rf"^पो-\s*({_MWNAME})(?:\s*कि)?")),
]
OWNER_OF_REL = [
    ("लड़का",  re.compile(rf"({_MWNAME})\s+की\s+(?:लड़‌?का|लडका)\s*(.+)")),
    ("बेटी",   re.compile(rf"({_MWNAME})\s+की\s+(?:बेटी|लड़की)\s*(.+)")),
    ("पत्नी",  re.compile(rf"({_MWNAME})\s+की\s+पत्नी\s*({_MWNAME})")),
    ("पतोह",   re.compile(rf"({_MWNAME})\s+की\s+(?:पतोह‌?|पतोहू‌?)\s*({_MWNAME})")),
]
_NLC = r"(.+)"
IMPLIED_SUBJECT_REL = [
    ("भाई",   re.compile(r"^के\s+भाई\s*" + _NLC)),
    ("लड़का", re.compile(r"^के\s+लड़का\s*" + _NLC)),
    ("बेटा",  re.compile(r"^के\s+बेटा\s*" + _NLC)),
    ("बेटी",  re.compile(r"^की\s+बेटी\s*" + _NLC)),
    ("लड़की", re.compile(r"^की\s+लड़की\s*" + _NLC)),
    ("पत्नी", re.compile(rf"^की\s+पत्नी\s*({_MWNAME})")),
    ("पतोह",  re.compile(rf"^की\s+(?:पतोह‌?|पतोहू‌?)\s*({_MWNAME})")),
    ("माता",  re.compile(rf"^के\s+साये\s+माता\s*({_MWNAME})")),
    ("मामा",  re.compile(rf"^के\s+मामा\s*({_MWNAME})")),
]
IMPLIED_DIRECT_REL = [
    ("भाई",    re.compile(r"^भाई\s*(.+)")),
    ("लड़का",  re.compile(r"^लड़का\s*(.+)")),
    ("बेटा",   re.compile(r"^बेटा\s*(.+)")),
    ("बेटी",   re.compile(r"^बेटी\s*(.+)")),
    ("लड़की",  re.compile(r"^लड़की\s*(.+)")),
    ("पिता",   re.compile(rf"^पिता\s*({_MWNAME})")),
    ("दादा",   re.compile(rf"^दादा\s*({_MWNAME})")),
    ("पोता",   re.compile(rf"^पोता\s*({_MWNAME})")),
    ("पत्नी",  re.compile(rf"^पत्नी\s*({_MWNAME})")),
    ("पतोह",   re.compile(rf"^पतोह‌?\s*({_MWNAME})")),
    ("पतोहू",  re.compile(rf"^पतोहू‌?\s*({_MWNAME})")),
    ("भतीजा",  re.compile(r"^भतीजा\s*(.+)")),
    ("माता",   re.compile(rf"^माता\s*({_MWNAME})")),
    ("मामा",   re.compile(rf"^मामा\s*({_MWNAME})")),
]
OWNER_KE_REL = [
    ("भाई",   re.compile(rf"^({_MWNAME})\s+के\s+भाई\s*(.+)")),
    ("लड़का", re.compile(rf"^({_MWNAME})\s+के\s+लड़का\s*(.+)")),
    ("बेटा",  re.compile(rf"^({_MWNAME})\s+के\s+बेटा\s*(.+)")),
    ("पिता",  re.compile(rf"^({_MWNAME})\s+के\s+पिता\s*({_MWNAME})")),
    ("पोता",  re.compile(rf"^({_MWNAME})\s+के\s+पोता\s*({_MWNAME})")),
]
EXPLICIT_PAIR_REL = [
    ("लड़का", re.compile(rf"^({_MWNAME})\s+(?:लड़‌?का|लडका)\s+({_MWNAME})(?=\s|$)")),
    ("बेटा",  re.compile(rf"^({_MWNAME})\s+बेटा\s+({_MWNAME})(?=\s|$)")),
    ("भाई",   re.compile(rf"^({_MWNAME})\s+भाई\s+({_MWNAME})(?=\s|$)")),
]

INVALID_PARTICLES = {
    "के", "की", "का", "व", "और", "तथा", "साये", "नाती", "पिता", "लड़का",
    "बेटा", "भाई", "दादा", "पोता", "माता", "मामा", "पत्नी", "पतोह",
    "भतीजा", "संग", "संगे", "साथै", "साधै", "वाल", "वाला", "वाले", "जी",
}
KNOWN_SURNAMES = {
    "शर्मा", "सिंह", "कुमार", "पाठक", "जोशी", "शुक्ला", "महतो", "प्रसाद",
    "लाल", "यादव", "गुप्ता", "वर्मा", "देवी", "बाई", "तिवारी", "पाण्डेय",
    "मिश्रा", "चौधरी", "श्रीवास्तव", "खंगार", "राठौर", "तोमर", "सोदिया",
    "बघेले", "कुशवाहा",
}
KNOWN_CASTES = [
    "राजपूत", "राजावत", "ब्राह्मण", "शुक्ला", "पाठक", "जोशी",
    "ठाकुर", "गुजर", "यादव", "बघेल", "शर्मा", "वर्मा",
]

HINDI_MONTH_MAP = {
    "चैत": "चैत्र", "चैत्र": "चैत्र", "बैसाख": "वैशाख", "वैशाख": "वैशाख",
    "जेठ": "ज्येष्ठ", "ज्येष्ठ": "ज्येष्ठ", "असाढ़": "आषाढ़", "आषाढ़": "आषाढ़",
    "सावन": "श्रावण", "श्रावण": "श्रावण", "भादो": "भाद्रपद", "भाद्रपद": "भाद्रपद",
    "अश्विन": "आश्विन", "कुआर": "आश्विन", "आश्विन": "आश्विन",
    "कातिक": "कार्तिक", "कार्तिक": "कार्तिक",
    "अगहन": "मार्गशीर्ष", "मार्गशीर्ष": "मार्गशीर्ष",
    "पूस": "पौष", "पौष": "पौष",
    "माघ": "माघ", "माह": "माघ",
    "फागुन": "फाल्गुन", "फाल्गुन": "फाल्गुन",
}
HINDI_MONTH_TO_NUM = {
    "चैत्र": 1, "वैशाख": 2, "ज्येष्ठ": 3, "आषाढ़": 4,
    "श्रावण": 5, "भाद्रपद": 6, "आश्विन": 7, "कार्तिक": 8,
    "मार्गशीर्ष": 9, "पौष": 10, "माघ": 11, "फाल्गुन": 12,
}
_MONTH_RE = "|".join(re.escape(k) for k, v in HINDI_MONTH_MAP.items() if v)
STANDARD_DATE_RE = re.compile(
    rf"(?:मिती|मिटी\s*)?\s*(?P<month_name>{_MONTH_RE})"
    r"\s*(?:(?P<paksha>सुदी|वदी|बदी)\s*(?P<day>\d{1,2}))?"
    r"\s*(?:(?:सम्बत्?|सम्बर|सं\.?)\s*(?P<year>\d{4}))?"
)
DATE_DDMMYYYY_RE = re.compile(
    r"(?:दिनांक\s*)?(?P<day>\d{2})(?P<month>\d{2})/(?P<year>\d{4})"
)
DATE_DD_MM_YY_RE = re.compile(
    r"(?P<day>\d{1,2})[-./](?P<month>\d{1,2})[-./](?P<year>\d{2}(?:\d{2})?)"
)
PAKSHA_TITHI_RE = re.compile(r"^(सुदी|वदी|बदी)\s*(\d{1,2}|च|छ)")
BAHI_RE = re.compile(
    r"(बही|वही)\s*([^\s]*(?:की)?)\s*(?:नम्बर|नं\.?|No\.?|नम्वर)\s*([^\s]+)"
)
FOLIO_RE = re.compile(r"(पद्मा|पन्ना|पत्र)\s*(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Stage 1 – clean_text
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    text = raw.replace("\r", "")
    text = text.replace("\n", "###NEWLINE###")
    text = re.sub(r"[ \t]+", " ", text).strip()
    table = str.maketrans("०१२३४५६७८९", "0123456789")
    return text.translate(table)


# ---------------------------------------------------------------------------
# Stage 2 – segment
# ---------------------------------------------------------------------------

def segment_paragraphs(cleaned: str) -> list[str]:
    parts = cleaned.split("###NEWLINE###")
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Stage 3 – Named Entity Recognition (Google NLP)
# ---------------------------------------------------------------------------

def extract_entities(text: str, nlp_client=None) -> dict:
    """
    Call Google Natural Language API for NER on the paragraph text.
    Returns dict with keys 'persons' (list of names) and 'locations' (list of places).
    Falls back to empty lists if API is unavailable.
    """
    result = {"persons": [], "locations": []}
    if not nlp_client or not _NLP_AVAILABLE:
        return result
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return result

    plain = text.replace("###NEWLINE###", " ").strip()
    if not plain:
        return result

    try:
        doc = language_v1.types.Document(
            content=plain,
            type_=language_v1.types.Document.Type.PLAIN_TEXT,
            language="hi",
        )
        response = nlp_client.analyze_entities(
            request={"document": doc, "encoding_type": language_v1.types.EncodingType.UTF8}
        )
        for ent in response.entities:
            name = ent.name.strip()
            if ent.type_ == language_v1.types.Entity.Type.PERSON:
                result["persons"].append({"name": name, "salience": ent.salience})
            elif ent.type_ == language_v1.types.Entity.Type.LOCATION:
                result["locations"].append({"name": name, "salience": ent.salience})
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_name(raw: str) -> str:
    if not raw:
        return ""
    name = raw.strip()
    for suffix in [" की", " के", " का", " कि", " कु"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    name = re.sub(r"^\s*[^A-Za-zऀ-ॿ\-(\[]+", "", name, flags=re.UNICODE)
    name = re.sub(r"[^A-Za-zऀ-ॿ\s\-()]+$", "", name, flags=re.UNICODE)
    return name.strip("(). ")


def _split_conjoined(chunk: str) -> list[str]:
    if not chunk:
        return []
    norm = re.sub(r"[\s,।;/]+(?:और|तथा)?[\s,।;/]*", " व ", chunk)
    norm = re.sub(r"\s*व\s*", " व ", norm.strip())
    names = []
    for part in norm.split(" व "):
        n = _clean_name(part.strip())
        if n and len(n) >= 2 and n.lower() not in INVALID_PARTICLES and not n.isdigit():
            names.append(n)
    return names


def _convert_hindu_date(hc, day_str, month_str, year_str) -> str:
    if not HINDU_CALENDAR_AVAILABLE or not hc:
        return ""
    try:
        day = int(day_str)
        year = int(year_str)
        std_month = HINDI_MONTH_MAP.get(
            re.sub(r"(सुदी|वदी|बदी)$", "", month_str.strip()).strip()
        )
        if not std_month:
            return ""
        month_num = HINDI_MONTH_TO_NUM.get(std_month)
        if not month_num or not (1 <= day <= 32) or not (1000 < year < 3000):
            return ""
        obj = hc.find_regional_date(f"{day}/{month_num}/{year}")
        if obj and obj.get("ce_date"):
            return datetime.strptime(obj["ce_date"], "%d/%m/%Y").strftime("%Y-%m-%d")
    except Exception:
        pass
    return ""


def _confidence(match_type: str) -> float:
    scores = {
        "nlp_entity": 0.92,
        "explicit_marker": 0.85,
        "implicit_marker": 0.70,
        "regex_owner": 0.75,
        "regex_implied": 0.65,
        "regex_pair": 0.80,
        "orphan": 0.30,
    }
    return scores.get(match_type, 0.50)


# ---------------------------------------------------------------------------
# Stage 4 – Relation Extraction
# ---------------------------------------------------------------------------

def _extract_from_paragraph(
    para_text: str,
    para_idx: int,
    context: dict,
    nlp_entities: dict,
    hc=None,
) -> tuple[list[dict], dict]:
    """
    Extract all individual records from a single paragraph.
    Returns (records_list, updated_context).
    """
    individuals = []
    added = set()
    ctx = dict(context)

    text = para_text.lstrip("• ").strip().strip("'\"")
    text = re.sub(r"^\s*(?:\d+\.\s*|\d+\s+)", "", text).strip()

    family_id = f"F{para_idx + 1:03d}"

    # Detect folio
    m_folio = FOLIO_RE.search(text)
    if m_folio:
        ctx["folio"] = f"{m_folio.group(1)} {m_folio.group(2)}"
        text = text[: m_folio.start()] + text[m_folio.end() :]

    # Detect caste/place header on this line
    line_caste, line_subcaste, line_place = ctx.get("caste", ""), ctx.get("subcaste", ""), ctx.get("place", "")
    m_jat = re.match(r"जात\s+([^\s]+)(?:\s+([^\sप्रागसायेगांवजिलामितीलड़काबेटीभाईस्त्री]+))?", text)
    if m_jat:
        line_caste = m_jat.group(1).strip()
        line_subcaste = (m_jat.group(2) or "").strip()
        text = text[m_jat.end():].strip()

    # Enrich location from NLP if not already found
    if not line_place and nlp_entities.get("locations"):
        top_loc = max(nlp_entities["locations"], key=lambda x: x["salience"])
        line_place = top_loc["name"]

    # Detect primary person marker
    m_psr = PRIMARY_PERSON_SELF_REL.search(text)
    m_pm = PRIMARY_PERSON_MARKERS.search(text)
    marker_match = None
    if m_psr and m_pm:
        marker_match = m_psr if m_psr.start() <= m_pm.start() else m_pm
    elif m_psr:
        marker_match = m_psr
    elif m_pm:
        marker_match = m_pm

    primary_name = ""
    primary_is_female = False
    primary_desc = ""
    father_name = ""
    husband_name = ""
    wife_name = ""
    confidence_type = "orphan"

    text_after_marker = text

    if marker_match:
        confidence_type = "explicit_marker"
        primary_name = _clean_name(marker_match.group(2).strip())
        if marker_match == m_psr:
            primary_desc = marker_match.group(4).strip().replace("‌", "")
        consumed = marker_match.group(0)
        text_after_marker = text[marker_match.start() + len(consumed):].strip()

        m_stri = re.match(rf"^({_MWNAME})\s+स्त्री\s+({_MWNAME})$", primary_name)
        if m_stri:
            wife_name = _clean_name(m_stri.group(1))
            husband_name = _clean_name(m_stri.group(2))
            primary_name = wife_name
            primary_is_female = True
        else:
            primary_is_female = primary_desc in {"पतोह", "पतोहू", "बेटी", "लड़की", "पत्नी"}
    else:
        m_imp = IMPLICIT_PRIMARY_BETA_POTA.match(text)
        if m_imp:
            confidence_type = "implicit_marker"
            primary_name = _clean_name(m_imp.group(1))
            rel_word = m_imp.group(2)
            ancestor = _clean_name(m_imp.group(3))
            if rel_word == "बेटा":
                father_name = ancestor
            else:
                ctx["grandfather"] = ancestor
            text_after_marker = text[m_imp.end():].strip()

    # Seed primary name from NLP entities if regex failed
    if not primary_name and nlp_entities.get("persons"):
        top_person = max(nlp_entities["persons"], key=lambda x: x["salience"])
        primary_name = top_person["name"]
        confidence_type = "nlp_entity"

    if primary_name:
        ctx["primary"] = primary_name
        ctx["caste"] = line_caste
        ctx["subcaste"] = line_subcaste
        ctx["place"] = line_place
        ctx["date_text"] = ctx.get("date_text", "")
        ctx["date_greg"] = ctx.get("date_greg", "")

        gender = "स्त्री" if primary_is_female else "पुरुष"
        rel_str = "स्वयं"
        if primary_is_female and husband_name:
            rel_str = f"{husband_name} की पत्नी"
        elif father_name:
            rel_str = f"{father_name} का बेटा"

        key = (primary_name.lower(), rel_str)
        if key not in added:
            individuals.append({
                "Given Name": primary_name, "Relation": rel_str, "Gender": gender,
                "confidence": _confidence(confidence_type), "_is_primary": True,
                "Family Id": family_id,
            })
            added.add(key)

        if husband_name:
            hk = (husband_name.lower(), f"{primary_name} के पति")
            if hk not in added:
                individuals.append({
                    "Given Name": husband_name, "Relation": f"{primary_name} के पति",
                    "Gender": "पुरुष", "confidence": _confidence("explicit_marker"),
                    "_is_primary": False, "Family Id": family_id,
                })
                added.add(hk)

        if father_name:
            fk = (father_name.lower(), f"पिता ({primary_name} का)")
            if fk not in added:
                individuals.append({
                    "Given Name": father_name, "Relation": f"पिता ({primary_name} का)",
                    "Gender": "पुरुष", "confidence": _confidence("explicit_marker"),
                    "_is_primary": False, "Family Id": family_id,
                })
                added.add(fk)

    # Iterative relation extraction
    remaining = text_after_marker
    active_subject = ctx.get("primary", "")
    ITER_LIMIT = 15

    for _ in range(ITER_LIMIT):
        remaining = remaining.lstrip(", ").strip()
        if not remaining:
            break

        matched = False
        all_pattern_groups = [
            ("self_ancestor",  SELF_REL_TO_ANCESTOR),
            ("owner_of",       OWNER_OF_REL),
            ("owner_ke",       OWNER_KE_REL),
            ("implied_ke_ki",  IMPLIED_SUBJECT_REL),
            ("implied_direct", IMPLIED_DIRECT_REL),
            ("explicit_pair",  EXPLICIT_PAIR_REL),
        ]

        for group_name, patterns in all_pattern_groups:
            for rel_kw, pat in patterns:
                use_search = group_name in ("owner_of", "owner_ke")
                m = pat.search(remaining) if use_search else pat.match(remaining)
                if not m:
                    continue

                names_to_add = []
                owner = active_subject
                conf = _confidence("regex_implied")

                if group_name == "self_ancestor":
                    raw_anc = m.group(1).strip() if m.lastindex and m.group(1) else ""
                    names_to_add = [raw_anc] if raw_anc else []
                    conf = _confidence("regex_owner")

                elif group_name in ("owner_of", "owner_ke"):
                    owner_raw = _clean_name(m.group(1))
                    related_chunk = m.group(2).strip()
                    if owner_raw and owner_raw.lower() != active_subject.lower():
                        ok = (owner_raw.lower(), "अभिभावक")
                        if ok not in added:
                            individuals.append({
                                "Given Name": owner_raw, "Relation": "अभिभावक",
                                "Gender": "पुरुष", "confidence": _confidence("regex_owner"),
                                "_is_primary": False, "Family Id": family_id,
                            })
                            added.add(ok)
                    owner = owner_raw or active_subject
                    names_to_add = _split_conjoined(related_chunk)
                    conf = _confidence("regex_owner")

                elif group_name in ("implied_ke_ki", "implied_direct"):
                    chunk = m.group(1).strip()
                    if pat.pattern.endswith(rf"({_MWNAME})"):
                        names_to_add = [chunk]
                    else:
                        names_to_add = _split_conjoined(chunk)

                elif group_name == "explicit_pair":
                    p1 = _clean_name(m.group(1))
                    p2 = _clean_name(m.group(2))
                    if p1 and p2:
                        r1 = f"पिता ({p2} का)" if rel_kw in ("लड़का", "बेटा") else f"भाई ({p2} का)"
                        r2 = f"{rel_kw} ({p1} का/की)"
                        g1, g2 = "पुरुष", "पुरुष"
                        for nm, rl, gn in [(p1, r1, g1), (p2, r2, g2)]:
                            k = (nm.lower(), rl)
                            if k not in added:
                                individuals.append({
                                    "Given Name": nm, "Relation": rl, "Gender": gn,
                                    "confidence": _confidence("regex_pair"),
                                    "_is_primary": False, "Family Id": family_id,
                                })
                                added.add(k)
                    names_to_add = []
                    conf = _confidence("regex_pair")

                for raw_n in names_to_add:
                    for n in _split_conjoined(raw_n) if len(names_to_add) == 1 else [raw_n]:
                        fn = _clean_name(n)
                        if not fn or len(fn) < 2 or fn.lower() in INVALID_PARTICLES or fn.isdigit():
                            continue
                        rel_s = f"{rel_kw} ({owner} का/की)"
                        gdr = "पुरुष" if rel_kw in {
                            "भाई", "लड़का", "बेटा", "पिता", "दादा", "पोता", "मामा", "परदादा"
                        } else "स्त्री" if rel_kw in {
                            "बेटी", "लड़की", "पत्नी", "पतोह", "पतोहू", "माता"
                        } else ""
                        k = (fn.lower(), rel_s)
                        if k not in added:
                            individuals.append({
                                "Given Name": fn, "Relation": rel_s, "Gender": gdr,
                                "confidence": conf, "_is_primary": False,
                                "Family Id": family_id,
                            })
                            added.add(k)

                remaining = remaining[m.end():].strip()
                matched = True
                break
            if matched:
                break

        if not matched:
            break

    # Date extraction from remaining text
    date_text, date_greg = ctx.get("date_text", ""), ctx.get("date_greg", "")
    temp_month, temp_paksha, temp_day, temp_year = None, None, None, None

    m_dmy = DATE_DDMMYYYY_RE.search(remaining)
    m_dmy2 = DATE_DD_MM_YY_RE.search(remaining) if not m_dmy else None

    if m_dmy:
        date_text = m_dmy.group(0)
        try:
            d, mo, yr = int(m_dmy.group("day")), int(m_dmy.group("month")), int(m_dmy.group("year"))
            if 1 <= d <= 31 and 1 <= mo <= 12 and 1900 <= yr <= 2300:
                date_greg = f"{yr:04d}-{mo:02d}-{d:02d}"
        except ValueError:
            pass
        remaining = remaining.replace(date_text, "", 1).strip()
    elif m_dmy2:
        date_text = m_dmy2.group(0)
        try:
            d, mo = int(m_dmy2.group("day")), int(m_dmy2.group("month"))
            yr_s = m_dmy2.group("year")
            yr = int(yr_s) + (2000 if len(yr_s) == 2 else 0)
            if 1 <= d <= 31 and 1 <= mo <= 12 and 1900 <= yr <= 2300:
                date_greg = f"{yr:04d}-{mo:02d}-{d:02d}"
        except ValueError:
            pass
        remaining = remaining.replace(date_text, "", 1).strip()

    m_std = STANDARD_DATE_RE.search(remaining)
    if m_std and not date_text:
        temp_month = m_std.group("month_name")
        temp_day = m_std.group("day")
        temp_year = m_std.group("year")
        temp_paksha = m_std.group("paksha")
        remaining = remaining.replace(m_std.group(0), "", 1).strip()

    m_pt = PAKSHA_TITHI_RE.match(remaining)
    if m_pt and not date_text:
        temp_paksha = m_pt.group(1)
        raw_d = m_pt.group(2)
        temp_day = "8" if raw_d in ("च", "छ") else raw_d
        remaining = remaining[m_pt.end():].strip()

    if temp_month and temp_day and temp_year and temp_paksha:
        date_text = f"{temp_month} {temp_paksha} {temp_day} सं. {temp_year}"
        date_greg = _convert_hindu_date(hc, temp_day, f"{temp_month} {temp_paksha}", temp_year)

    ctx["date_text"] = date_text
    ctx["date_greg"] = date_greg

    # Attach date, caste, place to all individuals from this paragraph
    for ind in individuals:
        ind.setdefault("Caste", ctx.get("caste", ""))
        ind.setdefault("Subcaste", ctx.get("subcaste", ""))
        ind.setdefault("From Which Place", ctx.get("place", ""))
        ind.setdefault("Date of Ritual", date_text)
        ind.setdefault("Date of Ritual (Gregorian)", date_greg)
        ind.setdefault("Folio Number", ctx.get("folio", ""))
        ind.setdefault("Whose Ritual 1",
                       ctx.get("primary", "") if not ind.get("_is_primary") else "")
        ind.setdefault("Additional Information 1", "")
        if ind is individuals[-1] and remaining:
            ind["Additional Information 1"] = remaining

    # Orphan line
    if not individuals and remaining:
        individuals.append({
            "Given Name": remaining, "Relation": "उल्लेखित (शेष)", "Gender": "",
            "Caste": ctx.get("caste", ""), "Subcaste": ctx.get("subcaste", ""),
            "From Which Place": ctx.get("place", ""),
            "Date of Ritual": date_text, "Date of Ritual (Gregorian)": date_greg,
            "Folio Number": ctx.get("folio", ""),
            "Whose Ritual 1": ctx.get("primary", ""),
            "confidence": _confidence("orphan"),
            "_is_primary": False, "Family Id": family_id,
            "Additional Information 1": remaining,
        })

    return individuals, ctx


# ---------------------------------------------------------------------------
# Stage 5 – Validate
# ---------------------------------------------------------------------------

def validate_records(records: list[dict], confidence_threshold: float = 0.5) -> list[dict]:
    """
    Flag records below confidence threshold and filter obvious garbage.
    Returns all records with a 'flagged' bool added.
    """
    validated = []
    for rec in records:
        conf = rec.get("confidence", 0.5)
        name = rec.get("Given Name", "").strip()
        flagged = conf < confidence_threshold or len(name) < 2 or name.isdigit()
        validated.append({**rec, "flagged": flagged})
    return validated


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_nlp_pipeline(
    raw_ocr_text: str,
    file_name: str = "unknown",
    nlp_client=None,
    hc_instance=None,
    bahi_number: str = "",
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """
    Full NLP pipeline: clean → segment → NER → relation extraction → validate.
    Returns list of record dicts ready for database insertion.
    """
    # Stage 1
    cleaned = clean_text(raw_ocr_text)

    # Detect bahi number from full text
    if not bahi_number:
        m_bahi = BAHI_RE.search(cleaned)
        if m_bahi:
            bahi_number = f"{m_bahi.group(1)} {m_bahi.group(2)} {m_bahi.group(3)}".strip()

    # Stage 2
    paragraphs = segment_paragraphs(cleaned)

    # Global context carried across paragraphs
    context = {
        "primary": "", "caste": "", "subcaste": "", "place": "",
        "date_text": "", "date_greg": "", "folio": "",
    }

    # Detect pre-loop caste/subcaste/place header
    if paragraphs:
        csp_candidate = paragraphs[0].lstrip("• '\"").strip()
        m_csp = re.match(
            r"^\s*(?P<caste>[^\s(]+)\s*\((?P<subcaste>[^)]+)\)\s*वासी\s*(?P<place>.+)",
            csp_candidate,
        )
        if not m_csp:
            m_csp = re.match(
                r"^\s*(?P<caste>[A-Za-zऀ-ॿ]+(?:\s+[A-Za-zऀ-ॿ]+)?)\s+वासी\s+(?P<place>.+)",
                csp_candidate,
            )
        if m_csp:
            context["caste"] = m_csp.group("caste").strip()
            context["subcaste"] = m_csp.groupdict().get("subcaste", "").strip() or ""
            context["place"] = m_csp.group("place").strip().rstrip(" के")
            paragraphs[0] = ""

    all_records = []
    individual_counter = 1

    for idx, para in enumerate(paragraphs):
        if not para.strip():
            continue

        # Stage 3 – NER on this paragraph
        para_plain = para.replace("###NEWLINE###", " ")
        nlp_ents = extract_entities(para_plain, nlp_client)

        # Stage 4 – relation extraction
        individuals, context = _extract_from_paragraph(
            para, idx, context, nlp_ents, hc_instance
        )

        # Assemble full records
        for ind in individuals:
            given = ind.get("Given Name", "")
            surname = ""
            parts = given.split()
            if len(parts) > 1 and parts[-1] in KNOWN_SURNAMES:
                surname = parts[-1]
                given = " ".join(parts[:-1])

            m_img = re.search(r"(\d+)", file_name)
            record = {
                "Image No":                  m_img.group(1) if m_img else file_name,
                "File Name":                 file_name,
                "Bahi Number":               bahi_number,
                "Folio Number":              ind.get("Folio Number", ""),
                "Individual ID":             f"P{individual_counter:04d}",
                "Given Name":                given,
                "Surname":                   surname,
                "Gender":                    ind.get("Gender", ""),
                "Relation":                  ind.get("Relation", ""),
                "Caste":                     ind.get("Caste", ""),
                "Subcaste":                  ind.get("Subcaste", ""),
                "From Which Place":          ind.get("From Which Place", ""),
                "Date of Ritual":            ind.get("Date of Ritual", ""),
                "Date of Ritual (Gregorian)": ind.get("Date of Ritual (Gregorian)", ""),
                "Whose Ritual 1":            ind.get("Whose Ritual 1", ""),
                "Family Id":                 ind.get("Family Id", ""),
                "confidence":                ind.get("confidence", 0.5),
                "flagged":                   False,
                "Additional Information 1":  ind.get("Additional Information 1", ""),
            }
            all_records.append(record)
            individual_counter += 1

    # Stage 5 – validate
    all_records = validate_records(all_records, confidence_threshold)
    return all_records
