"""Unit tests for the pure building blocks (no network, no LLM)."""
import io

import numpy as np
import pytest
import soundfile as sf

from backend.agent.allocation import initial_allocation, reallocate, search_shape
from backend.audio.stitch import (
    concat_segments,
    decode_audio,
    encode_mp3,
    loudness_match,
    stitch_turns,
    trim_silence,
)
from backend.audio.wer import wer
from backend.jsonutil import JSONExtractError, extract_json
from backend.retrieval import BM25Index, overlap
from backend.schemas import Chunk
from backend.textnorm import normalize_for_tts, number_to_words


# ---------------------------------------------------------------- allocation (Snell et al.)
def test_single_and_no_critic_use_one_call_each():
    assert initial_allocation([1, 5, 3], 6, "single", 4) == [1, 1, 1]
    assert initial_allocation([1, 5, 3], 6, "no_critic", 4) == [1, 1, 1]


def test_uniform_spends_the_whole_budget_evenly():
    alloc = initial_allocation([1, 5, 3], 7, "uniform", 4)
    assert sum(alloc) == 7 and max(alloc) - min(alloc) <= 1


def test_adaptive_front_loads_hard_segments_and_holds_back_the_rest():
    alloc = initial_allocation([1, 5, 2, 4], 8, "adaptive", 4)
    assert alloc == [1, 2, 1, 2]
    assert sum(alloc) < 8  # remainder held for verifier-guided reallocation


def test_reallocate_ignores_segments_that_met_the_bar():
    extra = reallocate(3, best_scores=[4.5, 3.0, 3.9], difficulties=[2, 4, 2], spent=[1, 1, 1], accept=4.2, max_per=4)
    assert extra[0] == 0
    assert sum(extra) == 3
    assert extra[1] >= extra[2]  # bigger gap + harder => more budget


def test_reallocate_respects_caps_and_zero_budget():
    assert reallocate(0, [1.0], [5], [1], 4.2, 4) == [0]
    assert reallocate(10, [1.0, 1.0], [5, 5], [4, 3], 4.2, 4) == [0, 1]


@pytest.mark.parametrize("budget,diff,policy,expected", [
    (1, 5, "adaptive", (1, 0)),
    (3, 1, "adaptive", (1, 2)),   # easy: sequential revisions
    (4, 3, "adaptive", (2, 2)),
    (3, 5, "adaptive", (2, 1)),   # hard: mostly parallel
    (3, 1, "uniform", (3, 0)),    # classic best-of-N
])
def test_search_shape(budget, diff, policy, expected):
    assert search_shape(budget, diff, policy) == expected


# ---------------------------------------------------------------- WER
def test_wer_normalises_numbers_and_punctuation():
    assert wer("It has 5 apples!", "it has five apples") == 0.0
    assert wer("a b c d", "a x c") == 0.5
    assert wer("", "") == 0.0


# ---------------------------------------------------------------- text normalisation
@pytest.mark.parametrize("src,expected", [
    ("Water is H2O.", "Water is H two O."),
    ("It moves at 5 m/s.", "It moves at five metres per second."),
    ("Speed: 1 km/h", "Speed: one kilometre per hour"),
    ("About 45% of it", "About forty five percent of it"),
    ("In 1998 it rained", "In nineteen ninety eight it rained"),
    ("Heat to 100°C", "Heat to one hundred degrees Celsius"),
    ("x^2 + y = 3", "x squared plus y equals three"),
    ("CaCO3 → CaO + CO2", "C A C O three gives C A O plus C O two"),
    ("Read 5-10 pages", "Read five to ten pages"),
    ("SO what? NO way.", "SO what? NO way."),
])
def test_normalize_for_tts(src, expected):
    assert normalize_for_tts(src) == expected


def test_number_to_words():
    assert number_to_words("1,500") == "one thousand five hundred"
    assert number_to_words("3.14") == "three point one four"
    assert number_to_words("2005") == "two thousand and five"


# ---------------------------------------------------------------- JSON repair
def test_extract_json_from_fenced_prose_with_trailing_commas():
    assert extract_json('Sure!\n```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}


def test_extract_json_truncated_output_is_closed():
    assert extract_json('{"turns": [{"speaker": "host", "text": "hi"}, {"speaker": "exp') ["turns"][0]["text"] == "hi"


def test_extract_json_prefers_expected_type_and_smart_quotes():
    assert extract_json('noise [1,2] then {“k”: 1}', expect=dict) == {"k": 1}


def test_extract_json_raises_on_garbage():
    with pytest.raises(JSONExtractError):
        extract_json("no json here at all")


# ---------------------------------------------------------------- retrieval
def test_bm25_ranks_relevant_chunk_first():
    chunks = [Chunk(id="c1", section="Rust", text="Iron rusts when exposed to moist air; this is corrosion."),
              Chunk(id="c2", section="Fats", text="Oxidation of fats and oils makes food rancid."),
              Chunk(id="c3", section="Light", text="Light reflects from a mirror.")]
    idx = BM25Index(chunks)
    assert idx.search("corrosion of iron", k=1)[0][0].id == "c1"
    assert idx.search("rancidity of oils", k=1)[0][0].id == "c2"
    assert idx.search("zebra", k=3) == []


def test_overlap():
    assert overlap("iron rusts in moist air", "Iron rusts when exposed to moist air.") == 1.0
    assert overlap("photosynthesis needs sunlight", "Iron rusts.") == 0.0


# ---------------------------------------------------------------- audio
SR = 24000


def _tone(sec: float, amp: float = 0.3, f: float = 220.0) -> np.ndarray:
    t = np.arange(int(SR * sec)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _wav(a: np.ndarray, sr: int = SR) -> bytes:
    b = io.BytesIO()
    sf.write(b, a, sr, format="WAV")
    return b.getvalue()


def test_decode_wav_resamples_and_downmixes():
    stereo = np.stack([_tone(1.0), _tone(1.0)], axis=1)
    b = io.BytesIO()
    sf.write(b, stereo, 16000, format="WAV")
    out = decode_audio(b.getvalue(), SR)
    # 24000 samples written at 16 kHz = 1.5 s, resampled to 24 kHz
    assert out.ndim == 1 and abs(len(out) - 1.5 * SR) < 10


def test_decode_mp3_via_ffmpeg():
    mp3 = encode_mp3(_tone(1.0), SR)
    out = decode_audio(mp3, SR)
    assert abs(len(out) / SR - 1.0) < 0.15


def test_trim_and_loudness_match():
    a = np.concatenate([np.zeros(SR), _tone(0.5, 0.02), np.zeros(SR)])
    trimmed = trim_silence(a, SR)
    assert len(trimmed) < 0.7 * SR
    loud = loudness_match(trimmed)
    rms = float(np.sqrt(np.mean(loud ** 2)))
    assert 0.07 < rms < 0.13  # ~ -20 dBFS


def test_stitch_timestamps_are_monotonic_and_include_gaps():
    clips = [_tone(1.0), _tone(0.5), _tone(2.0)]
    res = stitch_turns(clips, ["Hello there, how are you today?", "Right.", "Long explanation follows here."], SR)
    assert len(res.spans) == 3
    assert all(b[0] >= a[1] for a, b in zip(res.spans, res.spans[1:]))
    assert res.duration > 3.5


def test_concat_segments_offsets():
    audio, offs = concat_segments([_tone(1.0), _tone(1.0)], SR, gap_s=0.9, with_stings=False)
    assert offs == [0.0, pytest.approx(1.9, abs=0.01)]
    assert np.max(np.abs(audio)) <= 0.951


def test_extract_json_top_level_list_is_not_cut_to_its_first_item():
    """Regression (real Gemini output): a list of turns came back as only its first turn."""
    text = '[{"speaker": "host", "text": "Hi"}, {"speaker": "expert", "text": "Hello"}]'
    assert extract_json(text) == [{"speaker": "host", "text": "Hi"}, {"speaker": "expert", "text": "Hello"}]
    assert extract_json(text, expect=dict) == {"speaker": "host", "text": "Hi"}  # explicit dict still honoured


def test_extract_json_lines_of_objects():
    text = '{"speaker": "host", "text": "Hi"}\n{"speaker": "expert", "text": "Hello"}'
    assert extract_json(text) == [{"speaker": "host", "text": "Hi"}, {"speaker": "expert", "text": "Hello"}]
    assert extract_json('{"a": 1}') == {"a": 1}


def test_tts_reads_names_with_digits_digit_by_digit_and_keeps_commas():
    from backend.textnorm import normalize_for_tts
    assert normalize_for_tts("Hi Error606, I'm Gree.") == "Hi Error six oh six, I'm Gree."
    assert normalize_for_tts("It has 1,200 cells, see?") == "It has one thousand two hundred cells, see?"


def test_retrieval_reads_indian_scripts():
    """Hindi/Malayalam/Tamil chapters must be searchable: whole words, vowel signs kept, endings tolerated."""
    from backend.retrieval import terms
    assert terms("चुम्बकों के ध्रुव।") == ["चुम्बकों", "चुम्बक", "के", "ध्रुव"]  # danda dropped, vowel signs kept
    assert set(terms("चुम्बक")) & set(terms("चुम्बकों"))
    assert set(terms("ജലാവസ്ഥകളിലൂടെയുള്ള")) & set(terms("ജലാവസ്ഥ"))  # Malayalam endings still match
    assert terms("வேதிவினைகளின்")[0] == "வேதிவினைகளின்"
    assert terms("Magnets attract") == ["magnet", "attract"]


def test_pdf_lines_in_indian_scripts_are_kept_and_cleaned():
    """A Hindi NCERT PDF lost every line (letters were counted as A-Z only) and its text layer
    repeats vowel signs ("क्योंों")."""
    from backend.ingest import Line, _sections_from_lines, dedupe_signs
    assert dedupe_signs("ये सारे लोग यहाँ क्योंों आए होंोंगे?") == "ये सारे लोग यहाँ क्यों आए होंगे?"
    assert dedupe_signs("കിളികൾ പൂവ് and English") == "കിളികൾ പൂവ് and English"
    body = "नीमा दोपहर में दो बजे स्कूल से लौटती है। इस समय घर पर दादी उसका इंतज़ार करती हैं।"
    lines = [Line(text="नीमा की दादी", size=20, bold=True, page=1)] + [Line(text=body, size=11, bold=False, page=1)] * 12
    title, sections = _sections_from_lines(lines, set())
    assert title == "नीमा की दादी"
    assert sum(len(p.split()) for s in sections for p, _ in s.paras) >= 12 * len(body.split())


def test_ocr_cleanup_drops_print_codes_but_keeps_maths():
    from backend.ingest import clean_ocr
    assert clean_ocr("$$0222CHO1$$\nनीमा की दादी") == "नीमा की दादी"
    assert clean_ocr("$0222CH01$ text") == "text"
    assert clean_ocr("0678CH04\nMagnets") == "Magnets"
    assert clean_ocr("Water is $$H_2O$$.") == "Water is $$H_2O$$."
    assert clean_ocr("बच्चोंों") == "बच्चों"


def test_short_unpunctuated_lines_in_scripts_without_capitals_are_headings():
    from backend.ingest import ingest_text
    body = " ".join(["नीमा दोपहर में दो बजे स्कूल से लौटती है।"] * 12)
    src = ingest_text(f"चित्र और बातचीत\n\n{body}\n\nनीमा की दादी\n\n{body}", 9000)
    assert src.title == "चित्र और बातचीत" and [s.title for s in src.sections][-1] == "नीमा की दादी"
