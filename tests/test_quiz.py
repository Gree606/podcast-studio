"""The quiz only uses verified facts, and no wrong option can be a correct fill (criterion 10)."""
import re
from pathlib import Path

import pytest

from backend.ingest import ingest_text
from backend.quiz import build_quiz
from backend.schemas import Chunk, Episode, EpisodeSettings, KeyFact, Section, Segment, Source

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?।])\s+", text.strip())[0]


def episode_from(src: Source, facts_per_seg: int = 2, n_seg: int = 4, verified=True) -> Episode:
    """An episode whose key facts are verbatim first sentences of passages, like the planner's grounded facts."""
    chunks = [c for c in src.chunks if len(first_sentence(c.text).split()) >= 8]
    segs = []
    for i in range(n_seg):
        group = chunks[i * facts_per_seg:(i + 1) * facts_per_seg]
        segs.append(Segment(id=f"s{i}", index=i, title=f"Part {i + 1}", goal="g", chunk_ids=[c.id for c in group],
                            key_facts=[KeyFact(fact=first_sentence(c.text), chunk_id=c.id, verified=verified) for c in group]))
    return Episode(id="ep_quiz", source_id=src.id, title=src.title, settings=EpisodeSettings(), segments=segs)


def check_invariants(quiz: list[dict], ep: Episode, src: Source) -> None:
    by_id = {c.id: c for c in src.chunks}
    verified = {(f.fact, f.chunk_id) for s in ep.segments for f in s.key_facts if f.verified is True}
    assert quiz == [] or 3 <= len(quiz) <= 5
    for q in quiz:
        assert (q["fact"], q["chunk_id"]) in verified, "question must come from a verified key fact"
        assert q["answer"] in q["fact"], "the answer must occur in the fact"
        assert q["answer"] in q["options"] and len(q["options"]) == len(set(o.lower() for o in q["options"])) >= 3
        assert q["question"].count("_____") == 1
        assert not re.search(rf"\b{re.escape(q['answer'])}\b", q["question"], re.I), "the blank must not give the answer away"
        passage = by_id[q["chunk_id"]].text.lower()
        for wrong in (o for o in q["options"] if o != q["answer"]):
            assert wrong.lower() != q["answer"].lower()
            assert wrong.lower() not in passage, f"wrong option {wrong!r} occurs in the cited passage"


@pytest.mark.parametrize("name", sorted(p.name for p in SAMPLES.glob("*.md")))
def test_quiz_invariants_hold_on_every_sample_chapter(name):
    src = ingest_text((SAMPLES / name).read_text(encoding="utf-8"), 9000, origin="sample")
    ep = episode_from(src)
    quiz = build_quiz(ep, src)
    assert len(quiz) >= 3, f"expected a quiz for {name}"
    check_invariants(quiz, ep, src)
    assert len({q["answer"].lower() for q in quiz}) == len(quiz), "each question should test a different term"


def test_wrong_option_found_in_cited_passage_is_dropped():
    src = ingest_text((SAMPLES / "chem-reactions.md").read_text(encoding="utf-8"), 9000, origin="sample")
    ep = episode_from(src)
    q = build_quiz(ep, src)[0]
    wrong = next(o for o in q["options"] if o != q["answer"])
    # Make the cited passage mention that option: now it could be a correct fill, so it must go.
    chunk = next(c for c in src.chunks if c.id == q["chunk_id"])
    chunk.text += f" Some books also mention {wrong} here."
    rebuilt = build_quiz(ep, src)
    check_invariants(rebuilt, ep, src)
    again = next((x for x in rebuilt if x["fact"] == q["fact"]), None)
    assert again is None or wrong not in again["options"]


def test_unverified_facts_are_never_used():
    src = ingest_text((SAMPLES / "nutrition-plants.md").read_text(encoding="utf-8"), 9000, origin="sample")
    for flag in (False, None):
        assert build_quiz(episode_from(src, verified=flag), src) == []


def test_no_quiz_when_fewer_than_three_safe_questions():
    # Every term appears in every passage, so no wrong option is ever safe: the quiz must be empty.
    text = "Rusting of iron needs oxygen and moisture, and paint protects the iron surface from rusting."
    chunks = [Chunk(id=f"c{i}", section="Rust", text=text) for i in range(4)]
    src = Source(id="src_amb", title="Rust", origin="paste", word_count=60, chunks=chunks,
                 sections=[Section(id="sec1", title="Rust", word_count=60, chunk_ids=[c.id for c in chunks])])
    segs = [Segment(id=f"s{i}", index=i, title="Rust", goal="g",
                    key_facts=[KeyFact(fact=text + (" " + "x" * i if i else ""), chunk_id=f"c{i}", verified=True)]) for i in range(4)]
    ep = Episode(id="ep_amb", source_id=src.id, title="Rust", settings=EpisodeSettings(), segments=segs)
    assert build_quiz(ep, src) == []


def test_two_good_questions_are_not_enough():
    src = ingest_text((SAMPLES / "crop-production.md").read_text(encoding="utf-8"), 9000, origin="sample")
    full = build_quiz(episode_from(src), src)
    assert len(full) >= 3
    keep = {q["fact"] for q in full[:2]}
    ep = episode_from(src)
    for s in ep.segments:
        s.key_facts = [f for f in s.key_facts if f.fact in keep]
    assert build_quiz(ep, src) == []


def test_missing_source_gives_no_quiz():
    src = ingest_text((SAMPLES / "chem-reactions.md").read_text(encoding="utf-8"), 9000, origin="sample")
    assert build_quiz(episode_from(src), None) == []
