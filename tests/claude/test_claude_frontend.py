"""Static checks of the frontend: DOM/API contract and basic accessibility (criteria 8, 9)."""
import re
from html.parser import HTMLParser
from pathlib import Path

from fastapi.routing import APIRoute

from backend.main import app

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
HTML = (FRONTEND / "index.html").read_text(encoding="utf-8")
JS = (FRONTEND / "app.js").read_text(encoding="utf-8")
CSS = (FRONTEND / "app.css").read_text(encoding="utf-8")
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class Node:
    def __init__(self, tag, attrs, parent):
        self.tag, self.attrs, self.parent, self.children, self.text = tag, dict(attrs), parent, [], ""

    def ancestors(self):
        n = self.parent
        while n is not None:
            yield n
            n = n.parent

    def descendants(self):
        for c in self.children:
            yield c
            yield from c.descendants()

    def all_text(self):
        return self.text + "".join(c.all_text() for c in self.children)


class Tree(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", [], None)
        self.cur = self.root
        self.nodes = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        self.nodes.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        n = Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        self.nodes.append(n)

    def handle_endtag(self, tag):
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.text += data


TREE = Tree(HTML)
IDS = [n.attrs["id"] for n in TREE.nodes if n.attrs.get("id")]


def strip_template(s: str) -> str:
    return re.sub(r"\$\{[^}]*\}", "X", s)


# ---------------------------------------------------------------- criterion 8
def test_ids_are_unique():
    assert len(IDS) == len(set(IDS)), sorted(i for i in IDS if IDS.count(i) > 1)


def test_every_id_used_in_js_exists_in_html():
    selectors = [m.group(2) for m in re.finditer(r"(?:\$\$?|querySelector(?:All)?)\(\s*([\"'`])(.*?)\1", JS)]
    used = {i for sel in selectors for i in re.findall(r"#([A-Za-z][\w-]*)", strip_template(sel))}
    used |= set(re.findall(r"getElementById\(\s*[\"'`]([\w-]+)", JS))
    assert len(used) > 40, "the scan should find the app's id lookups"
    missing = used - set(IDS)
    assert not missing, f"ids used in app.js but missing from index.html: {sorted(missing)}"


def test_every_api_path_in_js_is_a_real_route():
    paths = {strip_template(m.group(1)).split("?")[0] for m in re.finditer(r"[\"'`](/api/[^\"'`]*)", JS)}
    assert len(paths) >= 10
    routes = [re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", r.path) + "$") for r in app.routes if isinstance(r, APIRoute)]
    unknown = [p for p in paths if not any(rx.match(p) for rx in routes)]
    assert not unknown, f"app.js calls endpoints that don't exist: {unknown}"


def test_teacher_view_panels_exist():
    for pid in ("teacherPanel", "teacherTrace", "teacherQuality", "teacherClaims", "teacherSources", "teacherVersions", "teacherToggle"):
        assert pid in IDS, pid
    tabs = {n.attrs.get("data-tab") for n in TREE.nodes if n.tag == "button"}
    assert {"trace", "quality", "claims", "sources", "versions"} <= tabs


def test_kid_flow_elements_exist():
    for pid in ("journey", "setup", "setupPath", "samples", "drop", "file", "lengthChoices", "toneChoices", "generate",
                "makeCard", "rocket", "thought", "epWait", "resumeBtn", "player", "stageHost", "stageExpert", "playBtn",
                "nowLine", "trustBadge", "segments", "quizCard", "quizBody", "toast", "confetti"):
        assert pid in IDS, pid
    assert 'id="av-host"' in HTML and 'id="av-expert"' in HTML


# ---------------------------------------------------------------- criterion 9
def test_every_control_has_an_accessible_name():
    labelled = {n.attrs.get("for") for n in TREE.nodes if n.tag == "label"}
    bad = []
    for n in TREE.nodes:
        if n.tag not in ("button", "input", "select", "textarea") or n.attrs.get("type") == "hidden":
            continue
        ok = bool((n.attrs.get("aria-label") or "").strip() or n.attrs.get("aria-labelledby"))
        ok = ok or (n.tag in ("button", "select") and n.all_text().strip())
        ok = ok or (n.attrs.get("id") in labelled) or any(a.tag == "label" for a in n.ancestors())
        if not ok:
            bad.append((n.tag, n.attrs))
    assert not bad, bad
    assert sum(1 for n in TREE.nodes if n.tag in ("button", "input", "textarea")) > 15


def test_every_svg_has_a_title_or_label():
    svgs = [n for n in TREE.nodes if n.tag == "svg"]
    assert len(svgs) >= 8
    bad = [n.attrs for n in svgs if not (n.attrs.get("aria-label") or any(d.tag == "title" and d.all_text().strip() for d in n.descendants()))]
    assert not bad, bad


def test_reduced_motion_and_language():
    assert re.search(r"@media\s*\(prefers-reduced-motion:\s*reduce\)", CSS)
    assert "reducedMotion()" in JS  # confetti and smooth scrolling honour it too
    assert re.search(r"<html[^>]*\blang=\"en\"", HTML)
