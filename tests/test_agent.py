"""Tests for ingestion, tools, critic rules and the planner loop (mock LLM)."""
import asyncio

import pytest

from backend.agent.context import AgentContext
from backend.agent.critic import deterministic_issues
from backend.agent.planner import fallback_outline, run_planner
from backend.agent.tools import rule_check_fact, tool_define, tool_read, tool_search
from backend.agent.writer import WriterError, parse_turns
from backend.ingest import IngestError, ingest_text, select_chunks
from backend.schemas import Episode, EpisodeSettings, KeyFact, Segment, Turn

CHAPTER = """# Chemical Reactions

## 1.1 Chemical Equations

A chemical equation is a short way of representing a chemical reaction. Magnesium burns in oxygen to form magnesium oxide. The substances that undergo chemical change are called reactants. The new substances formed are called products. A balanced chemical equation has an equal number of atoms of each element on both sides. """ + "Filler sentence about equations. " * 30 + """

## 1.2 Corrosion

Corrosion is the process in which metals are attacked by substances around them such as moisture and acids. Rusting of iron is a common example of corrosion. Painting and galvanising protect iron from rusting. """ + "Filler sentence about rust. " * 30 + """

## Exercises

1. Why should a magnesium ribbon be cleaned before burning?
"""


@pytest.fixture
def ctx():
    src = ingest_text(CHAPTER, 9000)
    ep = Episode(id="ep_test", source_id=src.id, title=src.title, settings=EpisodeSettings(length="short"))
    chunks, _ = select_chunks(src, [], 9000)
    return AgentContext(episode=ep, source=src, chunks=chunks)


def test_ingest_text_detects_sections_and_skips_exercises(ctx):
    titles = [s.title for s in ctx.source.sections]
    assert ctx.source.title == "Chemical Reactions"
    assert titles == ["1.1 Chemical Equations", "1.2 Corrosion"]
    assert any("Exercises" in n for n in ctx.source.notes)


def test_ingest_rejects_tiny_input():
    with pytest.raises(IngestError):
        ingest_text("too short", 9000)


def test_select_chunks_respects_section_choice(ctx):
    sec2 = ctx.source.sections[1]
    chunks, titles = select_chunks(ctx.source, [sec2.id], 9000)
    assert titles == ["1.2 Corrosion"] and all(c.section == "1.2 Corrosion" for c in chunks)


def test_tools_search_read_define(ctx):
    assert "Rusting of iron" in asyncio.run(tool_search(ctx, {"query": "rusting iron"}))
    assert tool_read(ctx, {"chunk_id": "nope"}).startswith("Error")
    d = tool_define(ctx, {"term": "reactants"})
    assert "are called reactants" in d
    assert "does not mention" in tool_define(ctx, {"term": "zebra"})


def test_rule_check_corrects_wrong_citation(ctx):
    corrosion_chunk = next(c for c in ctx.source.chunks if "Rusting of iron" in c.text)
    other = next(c for c in ctx.source.chunks if c.id != corrosion_chunk.id)
    kf = rule_check_fact(ctx, KeyFact(fact="Rusting of iron is a common example of corrosion", chunk_id=other.id))
    assert kf.chunk_id == corrosion_chunk.id and kf.verified is not False
    bad = rule_check_fact(ctx, KeyFact(fact="Jupiter has seventy nine moons", chunk_id=other.id))
    assert bad.verified is False


def test_planner_runs_a_real_tool_loop(ctx):
    segs = asyncio.run(run_planner(ctx))
    kinds = [e.kind for e in ctx.episode.trace]
    assert kinds.count("action") >= 3 and "observation" in kinds
    assert segs[0].kind == "intro" and segs[-1].kind == "recap"
    assert all(f.chunk_id for s in segs for f in s.key_facts)


def test_fallback_outline(ctx):
    segs = fallback_outline(ctx)
    assert 1 <= len(segs) <= ctx.n_segments and all(s.key_facts for s in segs)


def test_parse_turns_maps_names_merges_and_strips_directions():
    data = {"turns": [{"speaker": "Riya", "text": "Hi! [laughs]"}, {"speaker": "riya", "text": "Ready?"},
                      {"speaker": "Arjun", "text": "Yes."}, {"speaker": "host", "text": "Go."}, {"speaker": "expert", "text": "Ok."}]}
    turns = parse_turns(data, "Riya", "Arjun")
    assert [t.speaker for t in turns] == ["host", "expert", "host", "expert"]
    assert turns[0].text == "Hi! Ready?"
    with pytest.raises(WriterError):
        parse_turns({"turns": [{"speaker": "host", "text": "only one"}]}, "Riya", "Arjun")


def test_deterministic_critic_flags_monologues_and_missing_facts():
    seg = Segment(id="s", index=0, title="t", goal="g", key_facts=[KeyFact(fact="Galvanising protects iron from rusting")])
    turns = [Turn(speaker="expert", text="word " * 120), Turn(speaker="host", text="As an AI, in this segment we learn.")]
    issues, penalty = deterministic_issues(seg, turns, 250)
    text = " ".join(issues)
    assert "monologue" in text and "banned" in text and "not covered" in text
    assert penalty > 0.5


@pytest.mark.parametrize("data", [
    [{"speaker": "host", "text": "Hi there!"}, {"speaker": "expert", "text": "Hello."},
     {"speaker": "host", "text": "Ready?"}, {"speaker": "expert", "text": "Yes."}],                      # bare list
    {"dialogue": [{"role": "Riya", "line": "Hi there!"}, {"role": "Arjun", "line": "Hello."},
                  {"role": "Riya", "line": "Ready?"}, {"role": "Arjun", "line": "Yes."}]},              # other keys
    {"segment": {"conversation": ["Riya: Hi there!", "Arjun: Hello.", "Riya: Ready?", "Arjun: Yes."]}},  # nested, "Name: line"
    {"turns": [{"Riya": "Hi there!"}, {"Arjun": "Hello."}, {"Riya": "Ready?"}, {"Arjun": "Yes."}]},      # {"Name": line}
])
def test_parse_turns_accepts_real_model_variants(data):
    turns = parse_turns(data, "Riya", "Arjun")
    assert [t.speaker for t in turns] == ["host", "expert", "host", "expert"]
    assert turns[0].text == "Hi there!"


def test_read_tool_batches_passages(ctx):
    ids = [c.id for c in ctx.source.chunks[:2]]
    out = tool_read(ctx, {"chunk_ids": ids + ["c999"]})
    assert all(f"[{i}]" in out for i in ids) and "not found: c999" in out
    assert tool_read(ctx, {"chunk_id": ids[0]}).startswith(f"[{ids[0]}]")  # single-id form still works


def test_parse_turns_strips_numbered_name_prefixes():
    turns = parse_turns({"turns": ["1. Riya: Quick recap?", "2. Arjun: You got it!", "3. Riya: Bye!"]}, "Riya", "Arjun", min_turns=2)
    assert [(t.speaker, t.text) for t in turns] == [("host", "Quick recap?"), ("expert", "You got it!"), ("host", "Bye!")]
