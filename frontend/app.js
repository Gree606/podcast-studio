"use strict";
// Chapter Chat frontend. Plain JS, no build step. Every element looked up by id here
// exists in index.html (checked by tests/claude/test_claude_frontend.py).

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmtT = s => s == null ? "" : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

const DEFAULT_PRESETS = {
  length: [{value: "short", label: "Quick Bite", emoji: "🍪", hint: ""}, {value: "medium", label: "Story Time", emoji: "📖", hint: ""}, {value: "long", label: "Deep Dive", emoji: "🤿", hint: ""}],
  tone: [{value: "casual", label: "Fun", emoji: "🎈", hint: ""}, {value: "balanced", label: "Balanced", emoji: "⚖️", hint: ""}, {value: "serious", label: "Exam Hero", emoji: "🏆", hint: ""}],
};
const FIXES = [
  {label: "🐣 Simpler", instruction: "Explain this more simply, as if for a younger student. Keep every fact."},
  {label: "🍎 Add an example", instruction: "Add an everyday example that makes the idea easy to picture."},
  {label: "🎈 Make it fun", instruction: "Make it more fun and playful, with a wow-moment, but keep every fact."},
  {label: "✂️ Shorter", instruction: "Make it shorter and punchier."},
];
const STAGE_TEXT = {
  queued: "⏳ Waiting for a free studio", start: "🛰️ Opening the channel", plan: "📖 Reading your chapter",
  write: "✍️ Writing the script", verify: "🔎 Checking every fact against the book", voice: "🎙️ Recording the voices",
  assemble: "🎧 Mixing the episode", done: "🎉 Ready!",
};
const STATIONS = ["plan", "write", "voice", "assemble"];
const STATION_OF = {queued: "plan", start: "plan", plan: "plan", write: "write", verify: "voice", voice: "voice", assemble: "assemble"};

let CONFIG = {lengths: {}, presets: DEFAULT_PRESETS};
let SOURCE = null, epState = null, pollTimer = null, traceLen = 0, traceEvents = [];
let sourcesCache = null, currentAudioFile = null, lineIndex = [], quiz = null, quizLoadedFor = null, viewId = 0;
let seenLines = new Set(), launched = false;
// Live mode: play part by part while later parts are still being recorded (k = part playing, waiting = part k+1 not ready yet)
let live = {on: false, k: 0, waiting: false};  // bubbles already popped in; ships already flown in

// ------------------------------------------------------------------ helpers
async function api(path, opts = {}) {
  let r;
  try { r = await fetch(path, opts); }
  catch { throw new Error("Can't reach the studio right now. Check your internet and try again."); }
  let data = null;
  try { data = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) throw new Error((data && data.detail) || `Something went wrong (${r.status}). Please try again.`);
  return data;
}
const postJSON = (path, body) => api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});

let toastTimer = null;
function toast(msg, kind = "") {
  const t = $("#toast");
  t.textContent = msg; t.className = `toast show ${kind}`;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.className = "toast"; }, kind === "err" ? 6000 : 3500);
}
function showErr(el, msg) { el.innerHTML = `<div class="notice err" role="alert">😕 ${esc(msg)}</div>`; }
function tabs(root) {
  $$(".tabs", root).forEach(list => {
    const host = list.parentElement, btns = $$(".tab", list);
    btns.forEach(t => {
      t.onclick = () => {
        btns.forEach(x => x.setAttribute("aria-selected", String(x === t)));
        $$(":scope > [data-pane]", host).forEach(p => p.classList.toggle("hidden", p.dataset.pane !== t.dataset.tab));
        if (t.dataset.tab === "sources") renderTeacherSources();
      };
      t.onkeydown = e => {
        const i = btns.indexOf(t), n = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (n) { e.preventDefault(); const b = btns[(i + n + btns.length) % btns.length]; b.focus(); b.click(); }
      };
    });
  });
}
// menu drawer (global) and the episode's slide-up sheet (script, quiz, agent, save)
function openDrawer(open) {
  const d = $("#drawer"); d.classList.toggle("open", open); d.setAttribute("aria-hidden", String(!open));
  $("#menuBtn").setAttribute("aria-expanded", String(open));
}
const SHEET_TITLES = {script: "Script", quiz: "Quiz", agent: "How the agent made it", save: "Save & share"};
const wide = () => matchMedia("(min-width: 1100px)").matches;
function openSheet(name) {
  const sh = $("#sheet"); if (!sh) return;
  if (!name && wide()) return;
  if (!name) { sh.classList.remove("open"); sh.setAttribute("aria-hidden", "true"); $$(".dock-btn").forEach(b => b.classList.remove("on")); return; }
  if (name === "agent" && !document.body.classList.contains("teacher")) setTeacher(true);
  $$("[data-pane-sheet]", sh).forEach(p => p.classList.toggle("on", p.dataset.paneSheet === name));
  $$(".dock-btn").forEach(b => b.classList.toggle("on", b.dataset.sheet === name));
  $("#sheetTitle").textContent = SHEET_TITLES[name] || "";
  sh.classList.add("open"); sh.setAttribute("aria-hidden", "false");
  if (name === "quiz") $("#quizBtn").classList.remove("nudge");
  if (name === "script") { const now = $("#segments .line.now"); if (now) now.scrollIntoView({block: "center"}); }
}

function journey(step) {
  const order = ["pick", "choose", "make", "listen"], idx = order.indexOf(step);
  $$("#journey li").forEach((li, i) => {
    li.classList.toggle("done", i < idx); li.classList.toggle("now", i === idx);
    if (i === idx) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
  });
}
function myEpisodes() { try { return JSON.parse(localStorage.getItem("myEpisodes") || "[]"); } catch { return []; } }
function rememberEpisode(id, title) {
  try {
    const l = myEpisodes().filter(x => x.id !== id); l.unshift({id, title, t: Date.now()});
    localStorage.setItem("myEpisodes", JSON.stringify(l.slice(0, 12)));
  } catch { /* private mode */ }
}
function chapterEmoji(title, subject, i) {
  const t = `${title} ${subject}`.toLowerCase();
  const map = [[/chem|reaction|acid|metal/, "🧪"], [/brain|control|nerv|coordination/, "🧠"], [/crop|farm|agri|soil/, "🌾"],
    [/plant|nutrition|leaf|photosynth/, "🌱"], [/nation|europe|revolution|war|empire|history/, "🏰"], [/light|electric|magnet|motion|force/, "💡"],
    [/water|river|ocean|climate|geograph/, "🌍"], [/math|number|algebra|geometry/, "➗"], [/body|health|cell|life/, "🫀"], [/space|star|planet/, "🪐"]];
  for (const [re, e] of map) if (re.test(t)) return e;
  return ["📘", "📗", "📙", "📕"][i % 4];
}

// ------------------------------------------------------------------ teacher view + setup card
function setTeacher(on) {
  document.body.classList.toggle("teacher", on);
  $("#teacherToggle").setAttribute("aria-pressed", String(on));
  try { localStorage.setItem("teacherView", on ? "1" : "0"); } catch { /* ignore */ }
  if (on && epState) renderTeacher(epState);
}
function renderSetup() {
  const c = CONFIG.configured, box = $("#setup");
  const items = [["llm", "LLM_API_KEY", "Google Gemini: writes and checks the script"], ["tts", "TTS_API_KEY", "Sarvam AI: the two voices"],
    ["asr", "ASR_API_KEY", "Bodhan: listens back to check the audio"], ["ocr", "OCR_API_KEY", "Bodhan Indic-OCR: reads scanned PDF pages"]];
  if (c && items.some(([k]) => k in c && !c[k])) {
    box.classList.remove("hidden");
    $("#setupTitle").textContent = items.some(([k]) => c[k]) ? "🔑 Almost there: some keys are still missing" : "🔑 Grown-up setup needed";
    $("#setupPath").textContent = CONFIG.secrets_path || "~/.config/podcast-studio/secrets.env";
    $("#setupList").innerHTML = items.filter(([k]) => k in c).map(([k, name, what]) => `<li>${c[k] ? "✅" : "⬜"} <code>${name}</code> ${esc(what)}${c[k] ? " · working" : " · demo placeholder until set"}</li>`).join("");
  } else box.classList.add("hidden");
  $("#demoNote").innerHTML = CONFIG.mock && !(c && items.some(([k]) => k in c && !c[k]))
    ? `<div class="notice info small">🧪 Demo mode: scripts and voices are placeholders.</div>` : "";
}

// ------------------------------------------------------------------ home: pick + choose
async function renderHome() {
  stopPoll(); journey("pick");
  const view = ++viewId;
  const app = $("#app"); app.className = "home-layout"; app.innerHTML = ""; app.append($("#tpl-home").content.cloneNode(true)); tabs(app);
  try {
    const samples = await api("/api/samples");
    if (view !== viewId) return; // the user already moved on
    $("#samples").innerHTML = samples.length ? samples.map((s, i) => `
      <button class="sample c${i % 6}" type="button" data-id="${esc(s.id)}">
        <span class="book b${i % 6}" aria-hidden="true"><span class="emoji">${chapterEmoji(s.title, s.subject, i)}</span></span>
        <span class="tag">${esc(s.subject)} · Class ${esc(s.class)}</span>
        <b>${esc(s.title)}</b><span class="blurb">${esc(s.blurb || "")}</span>
      </button>`).join("") : `<p class="muted">No ready-made chapters here. Upload a PDF or paste some text instead!</p>`;
    $$(".sample").forEach(b => b.onclick = () => { $$(".sample").forEach(x => x.classList.toggle("sel", x === b)); loadSource({sample_id: b.dataset.id}); });
  } catch (e) { if (view === viewId) showErr($("#samples"), e.message); }

  const drop = $("#drop"), file = $("#file");
  drop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } };
  ["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) loadSource({file: f}); });
  file.onchange = () => { if (file.files[0]) loadSource({file: file.files[0]}); };
  $("#pasteGo").onclick = () => loadSource({text: $("#pasteText").value, title: $("#pasteTitle").value});
  setupPhotos();

  const mine = myEpisodes();
  if (mine.length) {
    $("#recentCard").classList.remove("hidden");
    $("#recent").innerHTML = mine.map(x => `<a href="#/episode/${esc(x.id)}">${esc(x.title)}<span>${esc(new Date(x.t).toLocaleString())}</span></a>`).join("");
    $$("#recent a").forEach(a => { a.onclick = () => openDrawer(false); });
  }
}

// ------------------------------------------------------------------ photos of pages (OCR, one page per photo)
const MAX_PHOTOS = 6;
let PHOTOS = [];
function setupPhotos() {
  PHOTOS = [];
  const add = files => {
    for (const f of files) {
      if (!f.type.startsWith("image/")) continue;
      if (PHOTOS.length >= MAX_PHOTOS) { showErr($("#srcStatus"), `At most ${MAX_PHOTOS} pages at a time.`); break; }
      PHOTOS.push({file: f, url: URL.createObjectURL(f)});
    }
    drawPhotos();
  };
  ["#photoCamera", "#photoGallery"].forEach(id => { const inp = $(id); inp.onchange = () => { add([...inp.files]); inp.value = ""; }; });
  $("#photoGo").onclick = () => loadSource({images: PHOTOS.map(p => p.file), title: $("#photoTitle").value});
  drawPhotos();
}
function drawPhotos() {
  $("#photoList").innerHTML = PHOTOS.map((p, i) => `<li><img src="${p.url}" alt="Page ${i + 1}"><span>Page ${i + 1}</span>
    <button type="button" class="chip" data-rm="${i}" aria-label="Remove page ${i + 1}">✕</button></li>`).join("");
  $$("[data-rm]", $("#photoList")).forEach(b => b.onclick = () => { URL.revokeObjectURL(PHOTOS[+b.dataset.rm].url); PHOTOS.splice(+b.dataset.rm, 1); drawPhotos(); });
  const n = PHOTOS.length;
  $("#photoGo").disabled = n === 0;
  $("#photoGo").textContent = n ? `📖 Read ${n === 1 ? "this page" : `these ${n} pages`}` : "📖 Read these pages";
  $("#photoHint").textContent = n ? `${n} of ${MAX_PHOTOS} pages. Add more with the buttons above, or press Read.` : "Tip: lay the page flat, use good light, and fit the whole page in the frame.";
}

async function loadSource(input) {
  const st = $("#srcStatus");
  if (input.text != null && input.text.trim().split(/\s+/).length < 80) { showErr(st, "Please paste a bit more: at least a few paragraphs (80+ words)."); return; }
  const what = input.images ? `Reading ${input.images.length} page${input.images.length === 1 ? "" : "s"} from your photo${input.images.length === 1 ? "" : "s"}… about 5 seconds a page 📷`
    : input.file ? "Reading your PDF… scanned pages take a little longer 📄" : "Opening the chapter… 📖";
  st.innerHTML = `<p class="row small"><span class="spinner" role="img" aria-label="Loading"></span> ${what}</p>`;
  const fd = new FormData();
  if (input.sample_id) fd.append("sample_id", input.sample_id);
  if (input.file) fd.append("file", input.file);
  if (input.images) input.images.forEach(f => fd.append("images", f, f.name || "page.jpg"));
  if (input.text != null) fd.append("text", input.text);
  if (input.title) fd.append("title", input.title);
  const view = viewId;
  try {
    const src = await api("/api/sources", {method: "POST", body: fd});
    if (view !== viewId) return;
    SOURCE = src; st.innerHTML = ""; showConfigure();
  } catch (e) { if (view === viewId) showErr(st, e.message); }
}

function presetGroup(box, name, items, checked) {
  box.innerHTML = items.map(p => `<input type="radio" name="${name}" id="${name}-${esc(p.value)}" value="${esc(p.value)}" ${p.value === checked ? "checked" : ""}>
    <label for="${name}-${esc(p.value)}" title="${esc(p.hint)}"><span class="e" aria-hidden="true">${esc(p.emoji)}</span><b>${esc(p.label)}</b><span class="small" data-hint="${esc(p.value)}">${esc(p.hint)}</span></label>`).join("");
}

function showConfigure() {
  journey("choose");
  const c = $("#configure"); c.classList.remove("hidden");
  $("#srcTitle").textContent = SOURCE.title;
  $("#srcMeta").textContent = `${SOURCE.word_count.toLocaleString()} words · ${SOURCE.sections.length} part${SOURCE.sections.length === 1 ? "" : "s"}`;
  $("#srcNotes").innerHTML = SOURCE.notes.map(n => `<div class="notice info small">ℹ️ ${esc(n)}</div>`).join("");
  const pages = SOURCE.page_texts || [];
  $("#ocrPreview").classList.toggle("hidden", !pages.length);
  $("#ocrPreviewBody").innerHTML = pages.map((t, i) => `<h4>Page ${i + 1}</h4><pre>${esc(t)}</pre>`).join("");
  $("#sections").innerHTML = SOURCE.sections.map(s => `<label><input type="checkbox" value="${esc(s.id)}" checked> ${esc(s.title)}</label>`).join("");
  const P = CONFIG.presets || DEFAULT_PRESETS, L = CONFIG.lengths || {};
  presetGroup($("#lengthChoices"), "length", P.length, "medium");
  presetGroup($("#toneChoices"), "tone", P.tone, "balanced");
  P.length.forEach(p => { const el = $(`[data-hint="${p.value}"]`, $("#lengthChoices")); if (el && L[p.value]) el.textContent = `~${Math.max(1, Math.round(L[p.value].audio_minutes))} min`; });
  const upd = () => {
    const k = $("input[name=length]:checked").value, sel = $$("#sections input:checked").length, est = L[k];
    $("#estimate").innerHTML = sel === 0 ? `<span class="pill warn">👆 Pick at least one part of the chapter</span>`
      : est ? `≈ ${Math.max(1, Math.round(est.audio_minutes))} min episode · starts playing in about a minute` : "";
    $("#generate").disabled = sel === 0;
  };
  $$("#configure input").forEach(i => i.addEventListener("change", upd)); upd();
  $("#selAll").onclick = () => { $$("#sections input").forEach(i => { i.checked = true; }); upd(); };
  $("#selNone").onclick = () => { $$("#sections input").forEach(i => { i.checked = false; }); upd(); };
  $("#generate").onclick = createEpisode;
  c.scrollIntoView({behavior: reducedMotion() ? "auto" : "smooth", block: "start"});
}

async function createEpisode() {
  if (!reducedMotion()) { document.body.classList.add("warp"); setTimeout(() => document.body.classList.remove("warp"), 1600); }
  const btn = $("#generate"); btn.disabled = true; btn.innerHTML = `<span class="spinner" role="img" aria-label="Starting"></span> Launching…`;
  const all = $$("#sections input"), sel = all.filter(i => i.checked).map(i => i.value);
  const settings = {length: $("input[name=length]:checked").value, tone: $("input[name=tone]:checked").value,
    host_name: $("#hostName").value, expert_name: $("#expertName").value, section_ids: sel.length === all.length ? [] : sel};
  try {
    const r = await postJSON("/api/episodes", {source_id: SOURCE.id, settings});
    rememberEpisode(r.id, SOURCE.title); location.hash = `#/episode/${r.id}`;
  } catch (e) { btn.disabled = false; btn.textContent = "🚀 Launch"; showErr($("#estimate"), e.message); }
}

// ------------------------------------------------------------------ episode
function stopPoll() { if (pollTimer) clearTimeout(pollTimer); pollTimer = null; }
const names = ep => ({host: ep.settings.host_name || "Error606", expert: ep.settings.expert_name || "Gree"});
const currentVersion = s => s.versions.find(v => v.version === s.active_version);

async function renderEpisode(id) {
  meter = null; drawWave.peaks = null;
  stopPoll(); viewId += 1;
  epState = null; traceLen = 0; traceEvents = []; sourcesCache = null; currentAudioFile = null; lineIndex = []; quiz = null; quizLoadedFor = null;
  seenLines = new Set(); launched = false; onTime.last = null; onTime.lastKey = null; live = {on: false, k: 0, waiting: false};
  const app = $("#app"); app.className = "ep-layout"; app.innerHTML = ""; app.append($("#tpl-episode").content.cloneNode(true)); tabs(app);
  if (wide()) openSheet("script");  // on monitors the panel is a permanent sidebar
  journey("make");
  const a = $("#audio");
  $("#playBtn").classList.add("waiting");
  $$(".dock-btn").forEach(b => { b.onclick = () => openSheet(b.dataset.sheet); });
  $$("[data-close-sheet]").forEach(b => { b.onclick = () => openSheet(null); });
  const seek = $("#seek");
  seek.oninput = () => { if (a.duration) a.currentTime = seek.value / 1000 * a.duration; };
  a.addEventListener("timeupdate", onTime);
  a.addEventListener("loadedmetadata", () => { $("#tDur").textContent = fmtT(a.duration); $("#playBtn").classList.remove("waiting"); drawWave.peaks = null; loadWave(a.getAttribute("src")); });
  addEventListener("resize", () => drawWave());
  a.addEventListener("play", startMeter);
  a.addEventListener("play", () => { if (a.currentTime < 1) launch(); $("#stage").classList.add("playing"); $("#playBtn").classList.remove("ready"); $("#playBtn").classList.add("is-playing"); $("#playBtn").setAttribute("aria-label", "Pause"); onTime(); });
  a.addEventListener("pause", () => { $("#stage").classList.remove("playing"); $("#playBtn").classList.remove("is-playing"); $("#playBtn").setAttribute("aria-label", "Play"); setSpeaking(null); });
  a.addEventListener("ended", () => { if (live.on) { nextLivePart(); return; } if (quiz && quiz.questions.length) { toast("🧠 Nice listening! Try the quiz."); $("#quizBtn").classList.add("nudge"); } });
  $("#playBtn").onclick = () => { if (live.on && !a.src) { playLivePart(0); return; } if (!a.src) return; if (a.paused) a.play().catch(() => toast("Tap play again to start the audio.")); else a.pause(); };
  $("#speedBtn").onclick = () => {
    const speeds = [1, 1.25, 1.5, 0.75], next = speeds[(speeds.indexOf(a.playbackRate) + 1) % speeds.length] || 1;
    a.playbackRate = next; $("#speedBtn").textContent = `${next}×`;
  };
  $("#share").onclick = async () => {
    try { await navigator.clipboard.writeText(location.href); toast("🔗 Link copied! Share it with a friend."); }
    catch { prompt("Copy this link:", location.href); }
  };
  $("#resumeBtn").onclick = async () => {
    $("#resumeBtn").disabled = true;
    try { await api(`/api/episodes/${epState.id}/resume`, {method: "POST"}); toast("🔁 Trying again…"); stopPoll(); poll(epState.id); }
    catch (e) { toast(e.message, "err"); }
    finally { $("#resumeBtn").disabled = false; }
  };
  await poll(id, viewId);
}

async function poll(id, view = viewId) {
  if (view !== viewId) return;
  let ep;
  try { ep = await api(`/api/episodes/${id}?since=${traceLen}`); }
  catch (e) {
    if (/not found/i.test(e.message)) {
      $("#app").innerHTML = `<div class="card"><div class="notice err">😕 ${esc(e.message)}</div><a class="btn primary" href="#/">✨ Make a new episode</a></div>`;
      return;
    }
    if (view !== viewId) return;
    if ($("#epMsg")) $("#epMsg").innerHTML = `<span class="pill warn">📡 Connection hiccup, trying again…</span>`;
    pollTimer = setTimeout(() => poll(id, view), 4000); return;
  }
  if (view !== viewId || !$("#makeCard")) return; // navigated away while waiting
  traceEvents.push(...ep.trace); traceLen = ep.trace_len;
  const prev = epState; epState = ep;
  rememberEpisode(ep.id, ep.title);
  renderMake(ep);
  const sig = e => JSON.stringify(e.segments.map(s => [s.status, s.active_version, s.versions.length, s.error])) + e.busy + e.status;
  if (!prev || sig(prev) !== sig(ep)) renderSegments(ep);
  if (ep.audio_file && ep.audio_file !== currentAudioFile) setAudio(ep);
  else if (!ep.audio_file && !live.on && partReady(ep, 0)) startLive(ep);
  if (live.on && live.waiting && partReady(ep, live.k + 1)) { live.waiting = false; playLivePart(live.k + 1); }
  if (live.on) liveTitle(ep);
  renderTrust(ep);
  if (document.body.classList.contains("teacher")) renderTeacher(ep, ep.trace);
  if (ep.status === "done" && quizLoadedFor !== ep.id) loadQuiz(ep);
  if (prev && prev.status !== "done" && ep.status === "done") { toast("🎉 Your podcast is ready! Press play."); confetti(40); }
  const active = ["queued", "running"].includes(ep.status) || ep.busy;
  if (active) pollTimer = setTimeout(() => poll(id, view), ep.status === "queued" ? 2500 : 1500);
}

function friendly(e) {
  const t = e.text || "";
  if (e.kind === "thought") return t.startsWith("Reflection stored in memory: ") ? `📝 Note to self: ${t.slice(29)}` : `💭 “${t}”`;
  let m = t.match(/^search\((.*)\)$/);
  if (m) { try { return `🔍 Looking up “${JSON.parse(m[1]).query}” in the book`; } catch { return "🔍 Searching the book"; } }
  if (/^list_sections/.test(t)) return "🗂️ Flipping through the chapter's sections";
  if (/^read\(/.test(t)) return "📖 Reading a passage really carefully";
  m = t.match(/^define\((.*)\)$/);
  if (m) { try { return `📚 Checking what the book says “${JSON.parse(m[1]).term}” means`; } catch { return "📚 Looking up a key word"; } }
  if (/^finish\(/.test(t)) return "🗺️ The plan for the episode is ready!";
  if (/^Writer produced draft/.test(t)) return "✍️ Writing a draft of one part";
  if (/^Revised draft/.test(t)) return "✨ Polishing a draft with the critic's tips";
  m = t.match(/^Claim check: (\d+)\/(\d+)/);
  if (m) return `✅ ${m[1]} of ${m[2]} facts found in the book`;
  if (/self-check: WER/.test(t)) return "🎙️ A line sounded mumbled, so it's being re-recorded";
  if (/^Audio for/.test(t)) return "🎧 Finished recording one part";
  return null;
}

function renderMake(ep) {
  const n = names(ep), working = ["queued", "running"].includes(ep.status);
  $("#epTitle").textContent = ep.title;
  $("#crewHost").textContent = n.host; $("#crewExpert").textContent = n.expert;
  const badge = {queued: ["", "⏳ In line"], running: ["", "🛠️ Working"], done: ["ok", "✅ Ready"], failed: ["bad", "😕 Stopped"], interrupted: ["warn", "⏸️ Paused"]}[ep.status] || ["", ep.status];
  $("#epStatus").innerHTML = `<span class="pill ${badge[0]}">${ep.busy && ep.status === "done" ? `<span class="spinner" role="img" aria-label="Updating"></span> Updating` : badge[1]}</span>`;

  const idx = ep.status === "done" ? STATIONS.length : Math.max(0, STATIONS.indexOf(STATION_OF[ep.stage] || ep.stage));
  $$("#track .station").forEach((s, i) => { s.classList.toggle("done", i < idx); s.classList.toggle("active", i === idx && working); });
  const pct = Math.round((ep.status === "done" ? 1 : ep.progress) * 100);
  $("#bar").style.width = `${Math.max(2, pct)}%`; $("#rocket").style.left = `${Math.min(97, Math.max(2, pct))}%`;
  $("#progress").setAttribute("aria-valuenow", String(pct));

  const latest = [...traceEvents].reverse().map(friendly).find(Boolean);
  $("#thought").textContent = working && latest ? latest : "";
  $("#stage").classList.toggle("busy", working);
  if (!currentAudioFile && !live.on) {  // nothing to play yet: the speech bubble narrates the work
    const say = ep.status === "failed" ? "😕 Lost the signal. Press Try again below."
      : ep.status === "interrupted" ? "⏸️ The station restarted. Press Try again below."
      : (STAGE_TEXT[ep.stage] || "🛠️ Working…");
    const cap = $("#nowLine");
    if (cap.dataset.say !== say) {
      cap.dataset.say = say; cap.classList.remove("from-host", "from-expert", "pop-in"); void cap.offsetWidth; cap.classList.add("pop-in");
      cap.innerHTML = `<span>${esc(say)}${working ? '<span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>' : ""}</span>`;
    }
    $("#plTitle").textContent = ep.title;
  }

  let msg;
  if (ep.queue_position) msg = `⏳ You're number ${ep.queue_position} in line. Other people are making podcasts right now; yours starts automatically.`;
  else if (ep.status === "failed") msg = `😕 Oops! ${ep.error || "Something went wrong."} Don't worry: press Try again and the hosts pick up where they stopped.`;
  else if (ep.status === "interrupted") msg = "⏸️ The studio restarted while making this. Press Try again to continue where it stopped.";
  else if (working) msg = `${STAGE_TEXT[ep.stage] || "🛠️ Working…"} <span class="muted small">${esc(ep.message)}</span>`;
  else msg = STAGE_TEXT.done;
  $("#epMsg").innerHTML = ep.status === "failed" || ep.status === "interrupted" ? esc(msg) : msg.startsWith("⏳") ? esc(msg) : msg;
  $("#epMsg").setAttribute("role", ep.status === "failed" ? "alert" : "status");

  const est = CONFIG.lengths && CONFIG.lengths[ep.settings.length];
  const elapsed = Math.max(0, Date.now() / 1000 - ep.created_at);
  $("#epWait").textContent = working ? `${fmtT(elapsed)} elapsed${est ? ` · usually about ${Math.ceil(est.wait_minutes)} min` : ""}` : "";
  // the speech bubble already says what the crew is doing; repeat it here only when something needs attention
  $("#makeCard").classList.toggle("quiet", working && !ep.queue_position);
  $("#resumeBtn").classList.toggle("hidden", !(["failed", "interrupted"].includes(ep.status) && !ep.busy));
  $("#makeCard").classList.toggle("hidden", ep.status === "done");
  journey(ep.status === "done" ? "listen" : "make");
}

function setAudio(ep) {
  currentAudioFile = ep.audio_file;
  const a = $("#audio"), rate = a.playbackRate;
  let t = a.currentTime, wasPlaying = !a.paused;
  if (live.on) {  // same spot in the full mix: where part k starts in the episode, plus how far into it we are
    const seg = ep.segments[live.k], v = seg && currentVersion(seg), first = v && v.turns.find(x => x.start != null && x.seg_start != null);
    const off = first ? first.start - first.seg_start : 0;
    if (live.waiting) { const nx = ep.segments[live.k + 1], nv = nx && currentVersion(nx), f2 = nv && nv.turns.find(x => x.start != null); t = f2 ? f2.start : off + (a.duration || 0); wasPlaying = true; }
    else t = off + t;
    live = {on: false, k: 0, waiting: false};
    if (wasPlaying) toast("🎧 The whole episode is ready, carrying on from the same spot.");
  }
  a.src = `/api/episodes/${ep.id}/audio?f=${encodeURIComponent(ep.audio_file)}`;
  a.addEventListener("loadedmetadata", () => {
    a.playbackRate = rate;
    if (t) a.currentTime = Math.min(t, a.duration || t);
    if (wasPlaying) a.play().catch(() => {});
  }, {once: true});
  $("#player").classList.remove("hidden");
  if (!launched) { launched = true; launch(); }
  if (!wasPlaying) {
    $("#playBtn").classList.add("ready"); const cap = $("#nowLine"); cap.dataset.say = "";
    cap.classList.remove("from-host", "from-expert"); cap.innerHTML = "<span>🎧 Ready. Press play!</span>";
  }
  const n = names(ep);
  $("#nameHost").textContent = n.host; $("#nameExpert").textContent = n.expert;
  $("#plTitle").textContent = ep.title;
  $("#dlMp3").href = `/api/episodes/${ep.id}/audio?download=true`;
  $("#dlTxt").href = `/api/episodes/${ep.id}/transcript`;
}

const ICON_CHECK = '<svg class="ico-check" viewBox="0 0 20 20" aria-hidden="true" focusable="false"><circle cx="10" cy="10" r="9" fill="currentColor" opacity=".18"/><path d="M6 10.4l2.6 2.6L14.2 7" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
function renderTrust(ep) {
  const done = ep.segments.filter(s => currentVersion(s));
  if (!done.length) { $("#trustBadge").innerHTML = ""; return; }
  const claims = done.flatMap(s => currentVersion(s).claims || []), bad = claims.filter(c => !c.supported).length;
  const failed = ep.segments.filter(s => s.status === "failed").length;
  const parts = [];
  if (claims.length) parts.push(bad ? `<span class="badge warn">${ICON_CHECK} ${claims.length - bad}/${claims.length} facts checked · ${bad} corrected</span>`
    : `<span class="badge">${ICON_CHECK} ${claims.length}/${claims.length} facts checked against your chapter</span>`);
  parts.push(`<span class="badge plain">🧩 ${ep.segments.length} parts</span>`);
  if (ep.duration) parts.push(`<span class="badge plain">⏱️ ${fmtT(ep.duration)}</span>`);
  if (failed) parts.push(`<span class="badge warn">⚠️ ${failed} part${failed > 1 ? "s" : ""} didn't finish: tap a “Make this part…” button under ${failed > 1 ? "them" : "it"} to redo</span>`);
  $("#trustBadge").innerHTML = parts.join("");
}

const sentences = text => (text.match(/[^.!?]+[.!?]+["')]*|[^.!?]+$/g) || [text]).map(x => x.trim()).filter(Boolean);
// a long explanation reads better as points than as a paragraph
function fmtTurn(text) {
  const ss = sentences(text);
  return ss.length >= 3 && text.split(/\s+/).length > 26 ? `<ul class="pts">${ss.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : esc(text);
}
function renderSegments(ep) {
  const box = $("#segments"), n = names(ep);
  const drafts = Object.fromEntries($$("input[data-own]", box).map(i => [i.dataset.own, i.value]));
  const openSrc = new Set($$("details.src[open]", box).map(d => d.dataset.seg));
  lineIndex = [];
  if (!ep.segments.length) {
    box.innerHTML = `<p class="muted"><span class="spinner" role="img" aria-label="Working"></span> ${esc(n.host)} and ${esc(n.expert)} are still reading the chapter…</p>`;
    return;
  }
  $("#scriptHint").textContent = ep.status === "done" ? "Tap any line to hear it. Want a part different? Use the buttons under it." : "Parts appear here as they are written. Tap any line to hear it.";
  let freshN = 0;  // new lines pop in one after another, like comic panels
  box.innerHTML = ep.segments.map((s, si) => {
    const v = currentVersion(s);
    const busy = ["writing", "synthesizing", "checking"].includes(s.status) && (ep.busy || ep.status === "running");
    const label = {writing: "✍️ Writing…", synthesizing: "🎙️ Recording…", checking: "🔎 Fact-checking…"}[s.status] || "";
    const status = s.status === "failed" ? `<span class="pill bad">needs a redo</span>` : busy ? `<span class="pill"><span class="spinner" role="img" aria-label="Working"></span> ${label}</span>` : s.status === "pending" ? `<span class="pill">⏳ waiting</span>` : "";
    const bad = v ? v.claims.filter(c => !c.supported) : [];
    const score = v && v.scores ? `<span class="pill teacher-only ${v.scores.overall >= 4 ? "ok" : v.scores.overall >= 3.3 ? "" : "warn"}" title="Critic score (1-5)">★ ${v.scores.overall.toFixed(1)}</span>` : "";
    const facts = v && v.claims.length ? `<span class="pill teacher-only ${bad.length ? "warn" : "ok"}">✓ ${v.claims.length - bad.length}/${v.claims.length} claims</span>` : "";
    const diff = `<span class="pill teacher-only" title="Difficulty predicted by the planner">difficulty ${s.difficulty}/5</span>`;
    const lines = v ? v.turns.map((t, ti) => {
      lineIndex.push({start: t.start, end: t.end, ss: t.seg_start, se: t.seg_end, speaker: t.speaker, text: t.text, seg: s.id, si});
      const key = `${s.id}:${v.version}:${ti}`, fresh = !seenLines.has(key);
      if (fresh) seenLines.add(key);
      return `<div class="line ${t.speaker}${fresh ? " fresh" : ""}" style="--i:${Math.min(fresh ? freshN++ : 0, 14)}" role="button" tabindex="0" data-start="${t.start ?? ""}" data-si="${si}" data-ss="${t.seg_start ?? ""}" aria-label="${esc(n[t.speaker])}: ${esc(t.text)}">
        <svg class="mini" viewBox="0 0 120 120" aria-hidden="true"><use href="#av-${t.speaker}"/></svg>
        <div class="bubble"><span class="who">${esc(n[t.speaker])}<span class="ts">${fmtT(t.start)}</span></span>${fmtTurn(t.text)}</div></div>`;
    }).join("") : s.error ? `<div class="notice err">😕 ${esc(s.error)}</div>` : `<p class="muted small">${s.status === "pending" ? "⏳ Waiting for its turn…" : "✍️ Being written…"}</p>`;
    const versions = s.versions.length > 1 ? `<div class="versions small"><b>Versions:</b>${s.versions.map(x => `<button type="button" class="chip ${x.version === s.active_version ? "on" : ""}" data-act="ver" data-seg="${s.id}" data-v="${x.version}" aria-pressed="${x.version === s.active_version}" title="${esc(x.instruction || "first version")}" ${ep.busy ? "disabled" : ""}>v${x.version}${x.instruction && x.instruction !== "regenerate" ? ` · ${esc(FIXES.find(f => f.instruction === x.instruction)?.label || "✏️ yours")}` : ""}</button>`).join("")}
      <button type="button" class="chip" data-act="preview" data-seg="${s.id}">▶ Hear another version</button><audio class="hidden" data-prev="${s.id}" controls></audio></div>` : "";
    const fix = ep.status === "done" ? `<div class="fixit">
        ${versions}
        <div class="chips"><b class="small">Make this part…</b>${FIXES.map((f, k) => `<button type="button" class="chip" data-act="fix" data-k="${k}" data-seg="${s.id}" ${ep.busy ? "disabled" : ""}>${f.label}</button>`).join("")}</div>
        <div class="own"><label class="sr" for="own-${s.id}">Your own idea for part ${si + 1}</label>
          <input type="text" id="own-${s.id}" data-own="${s.id}" maxlength="300" placeholder="✏️ Or type your own idea, e.g. “compare it to cricket”" value="${esc(drafts[s.id] || "")}">
          <button type="button" class="btn sm primary" data-act="own" data-seg="${s.id}" ${ep.busy ? "disabled" : ""}>Rewrite it</button></div>
      </div>` : "";
    return `<article class="segment c${si % 6}" data-segment="${s.id}">
      <div class="seg-head"><span class="seg-num">${si + 1}</span><h3>${esc(s.title)}</h3>${status}${score}${facts}${diff}</div>
      ${s.error && v ? `<div class="notice err seg-msg">😕 ${esc(s.error)}</div>` : ""}
      <div class="turns">${lines}</div>
      ${bad.map(c => `<div class="claim-bad">⚠️ We couldn't find this in the book, so double-check it: “${esc(c.claim)}”</div>`).join("")}
      <details class="src" data-seg="${s.id}" ${openSrc.has(s.id) ? "open" : ""}><summary>📚 Where this came from in the book</summary><div data-srcbox="${s.id}"><span class="spinner" role="img" aria-label="Loading"></span></div></details>
      ${fix}
    </article>`;
  }).join("");

  $$(".line", box).forEach(el => {
    const go = () => {
      const a = $("#audio");
      if (live.on) {
        if (el.dataset.ss === "" || !partReady(epState, +el.dataset.si)) { toast("🎙️ This part is still being recorded. It'll play in a moment!"); return; }
        playLivePart(+el.dataset.si, +el.dataset.ss); return;
      }
      if (el.dataset.start === "" || !a.src) { toast("🎙️ This line isn't recorded yet."); return; }
      a.currentTime = +el.dataset.start; a.play().catch(() => {});
    };
    el.onclick = go;
    el.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } };
  });
  $$("[data-act]", box).forEach(b => { b.onclick = () => segAction(b); });
  $$("input[data-own]", box).forEach(i => { i.onkeydown = e => { if (e.key === "Enter") { e.preventDefault(); $(`[data-act="own"][data-seg="${i.dataset.own}"]`, box).click(); } }; });
  $$("details.src", box).forEach(d => { d.ontoggle = () => { if (d.open) fillSources(d.dataset.seg); }; });
  openSrc.forEach(fillSources);
  onTime();
}

async function regenerate(segId, instruction) {
  const ep = epState, s = ep.segments.find(x => x.id === segId);
  await postJSON(`/api/episodes/${ep.id}/segments/${segId}/regenerate`, {instruction});
  toast(`✍️ Rewriting part ${s.index + 1}… the rest stays the same.`);
  stopPoll(); poll(ep.id);
}

async function segAction(b) {
  const ep = epState, seg = b.dataset.seg, act = b.dataset.act;
  if (act === "preview") {
    const a = $(`audio[data-prev="${seg}"]`), s = ep.segments.find(x => x.id === seg);
    const other = s.versions.filter(x => x.version !== s.active_version).map(x => x.version);
    const v = other.length ? Math.max(...other) : s.active_version;
    a.src = `/api/episodes/${ep.id}/segments/${seg}/audio?version=${v}`; a.classList.remove("hidden"); a.title = `Version ${v}`;
    a.play().catch(() => {}); return;
  }
  b.disabled = true;
  try {
    if (act === "fix") await regenerate(seg, FIXES[+b.dataset.k].instruction);
    else if (act === "own") {
      const input = $(`input[data-own="${seg}"]`), text = input.value.trim();
      if (!text) { toast("✏️ Type your idea first, then press Rewrite it."); b.disabled = false; input.focus(); return; }
      await regenerate(seg, text); input.value = "";
    } else if (act === "ver") {
      await postJSON(`/api/episodes/${ep.id}/segments/${seg}/activate`, {version: +b.dataset.v});
      toast(`🔀 Switched to version ${b.dataset.v}.`); stopPoll(); poll(ep.id);
    }
  } catch (e) { b.disabled = false; toast(e.message, "err"); }
}

async function getSources() {
  if (!sourcesCache) sourcesCache = await api(`/api/episodes/${epState.id}/sources`);
  return sourcesCache;
}
const passageHTML = p => `<div class="passage"><b>${esc(p.section)}</b>${p.page ? ` · page ${p.page}` : ""}<br>${esc(p.text)}</div>`;
async function fillSources(segId) {
  const box = $(`[data-srcbox="${segId}"]`); if (!box) return;
  try { box.innerHTML = ((await getSources())[segId] || []).map(passageHTML).join("") || `<span class="small muted">No passages recorded.</span>`; }
  catch (e) { box.textContent = e.message; }
}

// ------------------------------------------------------------------ live listening (part by part)
const partReady = (ep, k) => { const s = ep && ep.segments[k], v = s && currentVersion(s); return !!(v && v.audio_file && s.status === "done"); };
function startLive(ep) {
  live = {on: true, k: 0, waiting: false};
  $("#player").classList.remove("hidden");
  if (!launched) { launched = true; launch(); }
  const n = names(ep);
  $("#nameHost").textContent = n.host; $("#nameExpert").textContent = n.expert;
  $("#nowLine").classList.remove("from-host", "from-expert");
  $("#nowLine").innerHTML = "<span>📡 Part 1 has landed. Press play!</span>"; $("#nowLine").dataset.say = "";
  $("#playBtn").classList.remove("waiting"); $("#playBtn").classList.add("ready");
  liveTitle(ep);
  toast("🎧 Part 1 is ready. Start listening now!");
}
function liveTitle(ep) {
  const total = ep.segments.length, ready = ep.segments.filter((_, k) => partReady(ep, k)).length;
  $("#plTitle").textContent = `Part ${live.k + 1} of ${total} · ${ready < total ? `${ready}/${total} recorded` : "all recorded"}`;
}
function playLivePart(k, at = 0) {
  const ep = epState, s = ep.segments[k], v = s && currentVersion(s), a = $("#audio");
  if (!v) return;
  live.k = k; live.waiting = false; onTime.lastKey = null;
  a.src = `/api/episodes/${ep.id}/segments/${s.id}/audio?version=${v.version}`;
  a.addEventListener("loadedmetadata", () => { if (at) a.currentTime = at; a.play().catch(() => toast("Tap play to start the audio.")); }, {once: true});
  liveTitle(ep);
}
function nextLivePart() {
  const ep = epState, k = live.k + 1;
  if (k >= ep.segments.length) {  // heard every part before the final mix landed: start the full episode from the top
    live = {on: false, k: 0, waiting: false}; const a = $("#audio"); a.removeAttribute("src"); a.load();
    $("#nowLine").innerHTML = "<span>🎉 That's the whole transmission! The full episode is being mixed for downloading.</span>"; return;
  }
  if (partReady(ep, k)) { playLivePart(k); return; }
  live.waiting = true; setSpeaking(null);
  const cap = $("#nowLine"); cap.classList.remove("from-host", "from-expert");
  cap.innerHTML = `<span>📡 Receiving part ${k + 1}… it plays by itself the moment it lands.</span>`;
}

// "Incoming transmission": the two ships fly in and the banner flashes
function launch() {
  const st = $("#stage"); if (!st || reducedMotion()) return;
  st.classList.remove("launch"); void st.offsetWidth; st.classList.add("launch");
  clearTimeout(launch.t); launch.t = setTimeout(() => st.classList.remove("launch"), 3200);
}

// karaoke: highlight the line being spoken and make that host "talk"
function setSpeaking(speaker) {
  $("#stageHost").classList.toggle("speaking", speaker === "host");
  $("#stageExpert").classList.toggle("speaking", speaker === "expert");
  const st = $("#stage");  // signal dots fly from the speaker's ship to the listener's
  if (st) { st.classList.toggle("talk-host", speaker === "host"); st.classList.toggle("talk-expert", speaker === "expert"); }
}
// ------------------------------------------------------------------ waveform seek bar + live level
const peaksCache = new Map();
async function loadWave(src) {
  if (!src) return;
  if (!peaksCache.has(src)) {
    peaksCache.set(src, (async () => {
      const buf = await (await fetch(src)).arrayBuffer();
      const Ctx = window.AudioContext || window.webkitAudioContext, ctx = new Ctx();
      const audio = await ctx.decodeAudioData(buf); ctx.close();
      const ch = audio.getChannelData(0), n = 96, step = Math.floor(ch.length / n), out = [];
      for (let i = 0; i < n; i++) { let m = 0; for (let j = i * step; j < (i + 1) * step; j += 32) m = Math.max(m, Math.abs(ch[j])); out.push(m); }
      const top = Math.max(...out, .01); return out.map(v => Math.max(.08, Math.pow(v / top, .7)));
    })().catch(() => null));
  }
  const peaks = await peaksCache.get(src);
  if ($("#audio") && $("#audio").currentSrc.endsWith(src.split("/api")[1] || src)) { drawWave.peaks = peaks; drawWave(); }
}
function drawWave() {
  const c = $("#wave"), a = $("#audio"); if (!c || !a) return;
  const w = c.clientWidth, h = c.clientHeight, dpr = devicePixelRatio || 1;
  if (!w) return;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const peaks = drawWave.peaks || Array.from({length: 96}, (_, i) => .15 + .1 * Math.abs(Math.sin(i * .7)));
  const n = peaks.length, gap = 2, bw = Math.max(1.5, (w - gap * (n - 1)) / n), p = a.duration ? a.currentTime / a.duration : 0;
  const lvl = drawWave.level || 0;
  for (let i = 0; i < n; i++) {
    const x = i * (bw + gap), played = (i + .5) / n <= p, near = Math.abs((i + .5) / n - p) < 2.5 / n;
    const bh = Math.max(3, peaks[i] * (h - 6) * (near && !a.paused ? 1 + lvl * .6 : 1));
    g.fillStyle = played ? (near ? "#b9b0f2" : "#8c82d8") : "rgba(255,255,255,.18)";
    g.beginPath(); g.roundRect ? g.roundRect(x, (h - bh) / 2, bw, bh, bw / 2) : g.rect(x, (h - bh) / 2, bw, bh); g.fill();
  }
}
let meter = null;
function startMeter() {  // live audio level: the play button and the bars near the playhead pulse with the voice
  const a = $("#audio");
  try {
    if (!meter) {
      const Ctx = window.AudioContext || window.webkitAudioContext, ctx = new Ctx();
      const an = ctx.createAnalyser(); an.fftSize = 256;
      ctx.createMediaElementSource(a).connect(an); an.connect(ctx.destination);
      meter = {ctx, an, buf: new Uint8Array(an.frequencyBinCount), el: a};
    }
    if (meter.el !== a) return;  // a new page made a new <audio>; keep the plain player there
    meter.ctx.resume();
    const tick = () => {
      if (!$("#audio") || $("#audio") !== meter.el) return;
      meter.an.getByteFrequencyData(meter.buf);
      let s = 0; for (let i = 2; i < 48; i++) s += meter.buf[i];
      const lvl = Math.min(1, s / (46 * 160));
      drawWave.level = lvl; $("#playBtn").style.setProperty("--lvl", lvl.toFixed(3));
      if (!a.paused) drawWave();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  } catch { /* the plain player still works */ }
}

function onTime() {
  const a = $("#audio"); if (!a) return;
  const t = a.currentTime;
  drawWave();
  if (a.duration) {
    const p = t / a.duration;
    $("#seek").value = String(Math.round(p * 1000)); $("#seek").style.setProperty("--p", `${(p * 100).toFixed(1)}%`);
    $("#tCur").textContent = fmtT(t);
  }
  const cur = live.on
    ? lineIndex.find(l => l.si === live.k && l.ss != null && t >= l.ss && t < (l.se ?? l.ss) + 0.25)
    : lineIndex.find(l => l.start != null && t >= l.start && t < (l.end ?? l.start) + 0.25);
  const els = $$("#segments .line");
  let curEl = null;
  els.forEach(el => {
    const on = !!cur && (live.on ? +el.dataset.si === cur.si && el.dataset.ss !== "" && +el.dataset.ss === cur.ss
                                 : el.dataset.start !== "" && +el.dataset.start === cur.start);
    el.classList.toggle("now", on); if (on) curEl = el;
  });
  setSpeaking(cur && !a.paused ? cur.speaker : null);
  const key = cur ? `${cur.seg}:${cur.ss}` : null;
  if (cur && epState) {  // the bubble shows just the sentence being spoken (less text on screen)
    const ss = sentences(cur.text), t0 = live.on ? cur.ss : cur.start, t1 = live.on ? (cur.se ?? cur.ss) : (cur.end ?? cur.start);
    const frac = t1 > t0 ? Math.min(.999, Math.max(0, (t - t0) / (t1 - t0))) : 0, total = ss.reduce((n, x) => n + x.length, 0);
    let acc = 0, si = 0;
    for (; si < ss.length - 1; si++) { acc += ss[si].length; if (acc / total > frac) break; }
    const n = names(epState), cap = $("#nowLine");
    if (key !== onTime.lastKey) {  // a new line: pop a fresh comic bubble pointing at the speaker, and that ship hops
      onTime.lastKey = key; onTime.lastSi = -1;
      cap.classList.remove("from-host", "from-expert", "pop-in"); void cap.offsetWidth;
      cap.classList.add(`from-${cur.speaker}`, "pop-in");
      const ship = $(cur.speaker === "host" ? "#stageHost" : "#stageExpert");
      ship.classList.remove("hop"); void ship.offsetWidth; ship.classList.add("hop");
    }
    if (si !== onTime.lastSi) {
      onTime.lastSi = si;
      cap.innerHTML = `<span class="say"><b>${esc(n[cur.speaker])}</b>${esc(ss[si])}</span>`;
    }
  }
  if (curEl && curEl !== onTime.last && !a.paused) {  // keep the spoken line in view inside the scrollable log, not the page
    onTime.last = curEl;
    const box = $("#segments"), r = curEl.getBoundingClientRect(), b = box.getBoundingClientRect();
    if (r.top < b.top + 10 || r.bottom > b.bottom - 10) {
      box.scrollTo({top: box.scrollTop + (r.top - b.top) - (b.height - r.height) / 2, behavior: reducedMotion() ? "auto" : "smooth"});
    }
  }
}

// ------------------------------------------------------------------ quiz
async function loadQuiz(ep) {
  quizLoadedFor = ep.id;
  const view = viewId;
  try {
    const qs = await api(`/api/episodes/${ep.id}/quiz`);
    if (view !== viewId) return;
    if (!qs.length) { $("#quizCard").classList.add("hidden"); $("#quizBtn").classList.add("hidden"); return; }
    quiz = {questions: qs, i: 0, results: [], answered: false};
    $("#quizCard").classList.remove("hidden"); $("#quizBtn").classList.remove("hidden");
    renderQuiz();
  } catch { quizLoadedFor = null; }
}
function quizProgress() {
  return `<div class="q-progress" aria-hidden="true">${quiz.questions.map((_, k) => `<span class="${quiz.results[k] === true ? "right" : quiz.results[k] === false ? "wrong" : k === quiz.i ? "cur" : ""}"></span>`).join("")}</div>`;
}
function renderQuiz() {
  const body = $("#quizBody"), q = quiz.questions[quiz.i];
  if (!q) return renderQuizEnd();
  const text = esc(q.question).replace("_____", `<span class="blank" aria-label="blank">?</span>`);
  body.innerHTML = `${quizProgress()}
    <p class="small muted">Question ${quiz.i + 1} of ${quiz.questions.length} · from “${esc(q.segment_title)}”</p>
    <p class="q-text">${text}</p>
    <div class="q-options">${q.options.map((o, k) => `<button type="button" class="q-opt" data-k="${k}">${esc(o)}</button>`).join("")}</div>`;
  $$(".q-opt", body).forEach(b => { b.onclick = () => answer(b); });
}
function answer(b) {
  if (quiz.answered) return;
  quiz.answered = true;
  const q = quiz.questions[quiz.i], pick = q.options[+b.dataset.k], right = pick === q.answer;
  quiz.results[quiz.i] = right;
  $$(".q-opt", $("#quizBody")).forEach(x => { x.disabled = true; if (q.options[+x.dataset.k] === q.answer) x.classList.add("right"); });
  if (!right) b.classList.add("wrong");
  const where = `${esc(q.section)}${q.page ? `, page ${q.page}` : ""}`;
  const fb = document.createElement("div");
  fb.className = "q-feedback";
  fb.innerHTML = `<p><b>${right ? ["🎉 Yes! Spot on!", "🌟 Brilliant!", "🙌 You got it!"][quiz.i % 3] : `🤔 Not quite. The answer is “${esc(q.answer)}”.`}</b></p>
    <p class="small">📖 The book says (${where}): “${esc(q.fact)}”</p>
    <button type="button" class="btn primary" data-next="1">${quiz.i + 1 < quiz.questions.length ? "Next question ➡️" : "See my score 🏁"}</button>`;
  $(".q-options", $("#quizBody")).after(fb);
  $(".q-progress", $("#quizBody")).outerHTML = quizProgress();
  if (right) confetti(18);
  const next = $("[data-next]", fb);
  next.onclick = () => { quiz.i += 1; quiz.answered = false; renderQuiz(); $("#quizCard").scrollIntoView({block: "nearest"}); };
  next.focus();
}
function renderQuizEnd() {
  const right = quiz.results.filter(Boolean).length, total = quiz.questions.length, stars = Math.round(5 * right / total);
  const key = `quizBest:${epState.id}`;
  let best = 0; try { best = Math.max(+localStorage.getItem(key) || 0, right); localStorage.setItem(key, String(best)); } catch { best = right; }
  const msg = right === total ? "🏆 Perfect score! You're a chapter champion!" : right >= total * 0.6 ? "🌟 Great job! You really listened." : "💪 Good try! Listen once more and try again.";
  $("#quizBody").innerHTML = `<div class="q-end">${quizProgress()}
    <div class="stars" role="img" aria-label="${stars} out of 5 stars">${[0, 1, 2, 3, 4].map(k => `<span class="${k < stars ? "" : "off"}">⭐</span>`).join("")}</div>
    <h3>You got ${right} out of ${total}!</h3><p>${msg}</p>
    <p class="small muted">Your best so far: ${best}/${total}</p>
    <div class="row" style="justify-content:center"><button type="button" class="btn primary" data-again="1">🔁 Play again</button>
    <a class="btn" href="#/">✨ Make another episode</a></div></div>`;
  $("[data-again]").onclick = () => { quiz.i = 0; quiz.results = []; quiz.answered = false; renderQuiz(); };
  if (right >= total * 0.6) confetti(90);
}
function confetti(n) {
  if (reducedMotion()) return;
  const box = $("#confetti"), colors = ["#ffb703", "#ff5d7a", "#6c4cf5", "#1aa3d9", "#10a58f", "#80ed99"];
  for (let k = 0; k < n; k++) {
    const i = document.createElement("i");
    i.style.left = `${Math.random() * 100}%`; i.style.background = colors[k % colors.length];
    i.style.animationDelay = `${Math.random() * 0.6}s`; i.style.animationDuration = `${2 + Math.random() * 1.5}s`;
    box.append(i); setTimeout(() => i.remove(), 4200);
  }
}

// ------------------------------------------------------------------ teacher view panels
function renderTeacher(ep, newEvents) {
  if (!$("#teacherPanel")) return;
  const box = $("#teacherTrace");
  const events = newEvents && box.childElementCount ? newEvents : traceEvents;
  if (!newEvents || !box.childElementCount) box.innerHTML = "";
  const segName = id => { const s = ep.segments.find(x => x.id === id); return s ? `part ${s.index + 1}` : ""; };
  box.insertAdjacentHTML("beforeend", events.map(e => `<div class="ev"><span class="k ${esc(e.kind)}">${esc(e.kind)}<br><span class="seg">${esc(e.stage)} ${esc(segName(e.segment_id))}</span></span><span>${esc(e.text)}</span></div>`).join(""));
  if ($("#followTrace").checked) box.scrollTop = box.scrollHeight;
  renderQuality(ep); renderClaims(ep); renderVersions(ep);
}

function renderQuality(ep) {
  const m = ep.metrics, segs = ep.segments, done = segs.filter(s => currentVersion(s));
  const claims = done.flatMap(s => currentVersion(s).claims), sup = claims.filter(c => c.supported).length;
  const turns = done.flatMap(s => currentVersion(s).turns);
  const wers = turns.filter(t => t.wer != null).map(t => t.wer), retried = turns.filter(t => t.tts_attempts > 1).length;
  const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
  const sc = avg(done.map(s => currentVersion(s).scores?.overall).filter(x => x != null));
  const k = (v, l) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`;
  $("#teacherQuality").innerHTML = `<div class="kpis">
    ${k(sc != null ? `${sc.toFixed(2)}/5` : "–", "average critic score")}
    ${k(claims.length ? `${Math.round(100 * sup / claims.length)}%` : "–", `claims supported by the chapter (${sup}/${claims.length})`)}
    ${k(wers.length ? `${(100 * avg(wers)).toFixed(1)}%` : "–", `audio self-check WER · ${retried} line(s) re-recorded`)}
    ${k(m.llm_calls, "LLM calls")}${k(`${((m.prompt_tokens + m.completion_tokens) / 1000).toFixed(1)}k`, "tokens")}
    ${k(m.tts_calls, "TTS calls")}${k(`${Object.values(m.stage_seconds).reduce((a, b) => a + b, 0).toFixed(0)}s`, "agent time")}</div>
    <h3>Adaptive test-time compute (policy: ${esc(ep.settings.policy)})</h3>
    <p class="small muted">Each part's writer budget comes from its predicted difficulty, then leftover budget goes to parts the critic still scores below the bar (after Snell et al. 2024). Easy parts stop early; hard ones get more drafts.</p>
    <div class="scroll-x"><table><thead><tr><th>Part</th><th>Difficulty</th><th>Writer calls</th><th>Draft scores (★ chosen)</th><th>Lessons in memory (Reflexion)</th></tr></thead><tbody>
    ${segs.map(s => {
      const best = s.candidates.reduce((b, c) => (c.scores && (!b || c.scores.overall > b.scores.overall)) ? c : b, null);
      return `<tr><td>${s.index + 1}. ${esc(s.title)}</td><td>${s.difficulty}/5</td><td>${s.budget || s.candidates.length}</td>
        <td>${s.candidates.map(c => `<span class="${best && c.index === best.index ? "pill ok" : ""}" title="${esc(c.feedback || "")}">${c.round ? "rev" : "draft"} ${c.scores ? c.scores.overall.toFixed(2) : "–"}${best && c.index === best.index ? " ★" : ""}</span>`).join(" · ") || "–"}</td>
        <td class="small">${s.reflections.slice(-2).map(esc).join("<br>") || "–"}</td></tr>`;
    }).join("")}</tbody></table></div>
    <h3>Time per stage</h3><p class="small">${Object.entries(m.stage_seconds).map(([k2, v]) => `${esc(k2)}: ${v}s`).join(" · ") || "–"}</p>`;
}

function renderClaims(ep) {
  const rows = ep.segments.map(s => {
    const v = currentVersion(s);
    const facts = s.key_facts.map(f => `<div class="claim-row">${f.verified === true ? "✅" : f.verified === false ? "❌" : "◻️"} <b>key fact</b> ${esc(f.fact)} <span class="muted small">[${esc(f.chunk_id || "no citation")}]${f.verify_note ? ` · ${esc(f.verify_note)}` : ""}</span></div>`).join("");
    const claims = v ? v.claims.map(c => `<div class="claim-row">${c.supported ? "✅" : "⚠️"} ${esc(c.claim)} <span class="muted small">${c.evidence_chunk_id ? `[${esc(c.evidence_chunk_id)}]` : ""}${c.note ? ` · ${esc(c.note)}` : ""}</span></div>`).join("") : "";
    return `<h3>${s.index + 1}. ${esc(s.title)}</h3>${facts || `<p class="small muted">No key facts.</p>`}${claims || `<p class="small muted">Claims are checked after writing.</p>`}`;
  }).join("");
  $("#teacherClaims").innerHTML = `<p class="muted small">Planner key facts pass a two-layer check (rule-based lexical support, then a model judge, after ToolACE). After writing, every claim in the script is matched to BM25 evidence and judged; unsupported claims get one targeted rewrite and are flagged if they remain.</p>${rows || "–"}`;
}

async function renderTeacherSources() {
  if (!epState || !$("#teacherSources")) return;
  const box = $("#teacherSources");
  try {
    const src = await getSources();
    box.innerHTML = epState.segments.map(s => `<h3>${s.index + 1}. ${esc(s.title)}</h3>${(src[s.id] || []).map(passageHTML).join("") || `<p class="small muted">No passages yet.</p>`}`).join("") || "–";
  } catch (e) { box.textContent = e.message; }
}

function renderVersions(ep) {
  $("#teacherVersions").innerHTML = `<p class="muted small">Every regeneration is kept. Only the chosen part is rewritten and re-recorded, then the episode is re-stitched.</p>
    <div class="scroll-x"><table><thead><tr><th>Part</th><th>Version</th><th>Instruction</th><th>Score</th><th>Claims</th><th>Length</th><th></th></tr></thead><tbody>
    ${ep.segments.flatMap(s => s.versions.map(v => `<tr><td>${s.index + 1}. ${esc(s.title)}</td><td>v${v.version}${v.version === s.active_version ? " (in use)" : ""}</td>
      <td class="small">${esc(v.instruction || "first version")}</td><td>${v.scores ? v.scores.overall.toFixed(2) : "–"}</td>
      <td>${v.claims.filter(c => c.supported).length}/${v.claims.length}</td><td>${v.duration ? fmtT(v.duration) : "–"}</td>
      <td>${v.version === s.active_version ? "" : `<button type="button" class="btn sm" data-tver="${v.version}" data-tseg="${s.id}" ${ep.busy ? "disabled" : ""}>Use this</button>`}</td></tr>`)).join("") || `<tr><td colspan="7">No versions yet.</td></tr>`}
    </tbody></table></div>`;
  $$("[data-tver]", $("#teacherVersions")).forEach(b => {
    b.onclick = async () => {
      b.disabled = true;
      try { await postJSON(`/api/episodes/${ep.id}/segments/${b.dataset.tseg}/activate`, {version: +b.dataset.tver}); toast(`🔀 Switched to version ${b.dataset.tver}.`); stopPoll(); poll(ep.id); }
      catch (e) { b.disabled = false; toast(e.message, "err"); }
    };
  });
}

// ------------------------------------------------------------------ router + boot
function route() {
  const m = location.hash.match(/^#\/episode\/([\w-]+)/);
  openDrawer(false);
  if (m) renderEpisode(m[1]); else renderHome();
  window.scrollTo(0, 0);
}
(async () => {
  $("#howBtn").onclick = () => { openDrawer(false); $("#how").showModal(); };
  $("#menuBtn").onclick = () => openDrawer(!$("#drawer").classList.contains("open"));
  $$("[data-close-drawer]").forEach(b => { b.onclick = () => openDrawer(false); });
  addEventListener("keydown", e => { if (e.key === "Escape") { openDrawer(false); openSheet(null); } });
  $("#teacherToggle").onclick = () => setTeacher(!document.body.classList.contains("teacher"));
  let teacher = false; try { teacher = localStorage.getItem("teacherView") === "1"; } catch { /* ignore */ }
  setTeacher(teacher);
  try {
    CONFIG = {...CONFIG, ...(await api("/api/config"))};
    renderSetup();
  } catch {
    $("#demoNote").innerHTML = `<div class="notice warn small">😴 The studio is waking up. Please refresh in a few seconds.</div>`;
  }
  addEventListener("hashchange", route);
  route();
})();
