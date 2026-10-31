"""Deep agent stays within its per-job LLM budget; Reflexion notes come from the critic's feedback."""
from backend.agent.context import AgentContext
from backend.agent.critic import lesson_from
from backend.config import get_settings
from backend.ingest import ingest_text
from backend.schemas import Episode, EpisodeSettings, Segment

TEXT = "# Light\n\n" + " ".join(["Light travels in straight lines and reflects from shiny surfaces."] * 30)


def ctx_with(n_parts: int, calls_so_far: int) -> AgentContext:
    src = ingest_text(TEXT, 9000, origin="paste")
    ep = Episode(id="ep_b", source_id=src.id, title="t", settings=EpisodeSettings(mode="agent"),
                 segments=[Segment(id=f"s{i}", index=i, title=f"P{i}", goal="g", status="writing") for i in range(n_parts)])
    ctx = AgentContext(episode=ep, source=src, chunks=src.chunks)
    ep.metrics.llm_calls = calls_so_far
    return ctx


def test_optional_calls_leave_room_for_every_fact_check(monkeypatch):
    monkeypatch.setattr(get_settings(), "deep_max_llm_calls", 24)
    ctx = ctx_with(4, 18)              # 18 used; 4 parts still need 1 fact-check call each -> 2 spare
    assert ctx.can_spend(2) and not ctx.can_spend(3)
    ctx.claims_checked.add("s0")       # one part checked frees its reserve
    assert ctx.can_spend(3)


def test_budget_counts_per_job_not_per_episode_lifetime(monkeypatch):
    monkeypatch.setattr(get_settings(), "deep_max_llm_calls", 24)
    ctx = ctx_with(1, 0)
    ctx.episode.metrics.llm_calls = 40  # e.g. a later regeneration job: earlier calls don't count
    ctx.llm_start = 40
    assert ctx.can_spend(10)


def test_no_cap_when_zero(monkeypatch):
    monkeypatch.setattr(get_settings(), "deep_max_llm_calls", 0)
    assert ctx_with(4, 500).can_spend(100)


def test_lesson_comes_from_the_critic_without_a_model_call():
    assert lesson_from("Too lecture-like. Add a question from the host.", []) == "Too lecture-like."
    assert lesson_from("whatever", ["Monologue over 80 words in turn 3"]) == "Monologue over 80 words in turn 3."
    assert lesson_from("", []) == "Keep it faithful and conversational."
