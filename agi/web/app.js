const $ = (id) => document.getElementById(id);
const log = $("log");
const thoughts = $("thoughts");
const input = $("input");
const sendBtn = $("send");
const form = $("composer");

const settings = Object.assign(
  { speak: false, enter: true, chain: true, focus: false },
  JSON.parse(localStorage.getItem("cortex.settings") || "{}")
);

let ws, pingTimer;
let thinking = false;
let currentAgi = null;
let state = null;
let sessionId = localStorage.getItem("cortex.session") || "";
let sessions = [];
let prompts = [];
let lastUser = "";
let attachments = [];
let artifact = { lang: "", code: "" };
let palItems = [];
let palIdx = 0;

function proto() {
  return location.protocol === "https:" ? "wss" : "ws";
}
function saveSettings() {
  localStorage.setItem("cortex.settings", JSON.stringify(settings));
  document.body.classList.toggle("focus", settings.focus);
  $("opt-speak").checked = settings.speak;
  $("opt-enter").checked = settings.enter;
  $("opt-chain").checked = settings.chain;
  $("opt-focus").checked = settings.focus;
}
function setConn(on) {
  $("conn-pill").textContent = on ? "live" : "offline";
  $("conn-pill").className = "pill" + (on ? " live" : " dim");
}
function connect() {
  ws = new WebSocket(`${proto()}://${location.host}/ws`);
  ws.onopen = () => {
    setConn(true);
    setThinking(false);
    if (pingTimer) clearInterval(pingTimer);
    pingTimer = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "ping" }));
    }, 25000);
  };
  ws.onclose = () => {
    setConn(false);
    if (pingTimer) clearInterval(pingTimer);
    setTimeout(connect, 1200);
  };
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
}

function handle(msg) {
  if (msg.type === "hello" || msg.type === "state") {
    if (msg.state) renderState(msg.state);
    return;
  }
  if (msg.type === "pong") return;
  if (msg.type === "session") {
    loadSessions();
    return;
  }
  if (msg.type === "thought") {
    addThought(msg.kind, msg.text);
    setThinking(true);
    return;
  }
  if (msg.type === "token") {
    appendToken(msg.text);
    return;
  }
  if (msg.type === "done") {
    if (currentAgi) {
      currentAgi.classList.remove("streaming");
      currentAgi.dataset.done = "1";
      const body = currentAgi.querySelector(".body");
      let raw = body.textContent;
      const cut = raw.indexOf("**Thinking**");
      if (cut > 0) raw = raw.slice(0, cut).trim();
      body.innerHTML = md(raw);
      decorateCode(body);
      if (msg.chain) attachChain(currentAgi, msg.chain);
      harvestArtifact(raw);
      if (msg.intent === "research" || msg.intent === "compare" || msg.intent === "agent" || msg.doc_id) {
        artifact = { lang: "md", code: raw };
        $("art-label").textContent = msg.strategy || "document";
        $("art-code").textContent = raw;
        showTab("docs");
        loadDocs();
        loadTasks();
      }
      if (msg.intent === "note") loadDocs();
      if (msg.intent === "task") loadTasks();
      if (msg.intent === "spawn" || msg.intent === "agent") loadAgents();
      if (settings.speak) speak(raw);
    }
    currentAgi = null;
    setThinking(false);
    if (msg.latency_ms != null) $("lat-pill").textContent = `${Math.round(msg.latency_ms)} ms`;
    if (msg.strategy) {
      const conf = msg.confidence != null ? Math.round(msg.confidence * 100) : "—";
      $("chain-pill").textContent = `${msg.strategy} ${conf}%`;
      $("chain-pill").className = "pill";
    }
    return;
  }
  if (msg.type === "improve") {
    (msg.events || []).forEach((e) => addThought("improve", `${e.kind}: ${e.text || ""}`));
    return;
  }
  if (msg.type === "error") {
    addThought("error", msg.text || "error");
    setThinking(false);
  }
}

function addThought(kind, text) {
  const d = document.createElement("div");
  d.className = "t";
  d.dataset.kind = kind || "";
  d.innerHTML = `<span class="k">${esc(kind)}</span>${esc(text)}`;
  thoughts.prepend(d);
  while (thoughts.children.length > 48) thoughts.removeChild(thoughts.lastChild);
}

function attachChain(el, chain) {
  const steps = chain.steps || [];
  if (!steps.length) return;
  const det = document.createElement("details");
  det.className = "chain";
  det.open = settings.chain && chain.system === 2;
  const conf = Math.round((chain.confidence || 0) * 100);
  det.innerHTML = `<summary>chain · ${esc(chain.strategy || "?")} · ${steps.length} steps · ${conf}%</summary><ol></ol>`;
  const ol = det.querySelector("ol");
  for (const s of steps) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="k">${esc(s.kind)}</span> ${esc(s.text)}`;
    ol.appendChild(li);
  }
  const who = el.querySelector(".who");
  if (who) who.after(det);
}

function bubble(role, text, asHtml) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="who"><span>${role === "user" ? "you" : "cortex"}</span><span class="acts"></span></div><div class="body"></div>`;
  const body = el.querySelector(".body");
  if (asHtml) {
    body.innerHTML = md(text);
    decorateCode(body);
  } else body.textContent = text;
  const acts = el.querySelector(".acts");
  const copy = document.createElement("button");
  copy.textContent = "copy";
  copy.onclick = () => navigator.clipboard.writeText(text);
  acts.appendChild(copy);
  if (role === "agi") {
    const sp = document.createElement("button");
    sp.textContent = "speak";
    sp.onclick = () => speak(text);
    acts.appendChild(sp);
  } else {
    const ed = document.createElement("button");
    ed.textContent = "edit";
    ed.onclick = () => {
      input.value = text;
      input.focus();
    };
    acts.appendChild(ed);
  }
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function appendToken(text) {
  if (!currentAgi) {
    currentAgi = bubble("agi", "");
    currentAgi.classList.add("streaming");
  }
  currentAgi.querySelector(".body").textContent += text;
  log.scrollTop = log.scrollHeight;
}

function send(text, opts = {}) {
  let t = (text || input.value).trim();
  const who = $("agent-sel") && $("agent-sel").value;
  if (who && t && !t.startsWith("/") && !t.startsWith("@") && !/^run\s/i.test(t) && !/^do:/i.test(t) && !opts.noAgent) {
    t = `@${who} ${t}`;
  }
  if (!t && !attachments.length) return;
  if (t.startsWith("/") && handleSlash(t)) {
    input.value = "";
    return;
  }
  if (attachments.length) {
    const ctx = attachments.map((a) => `[Attached: ${a.name}]\n${a.excerpt}`).join("\n\n");
    t = ctx + (t ? `\n\n${t}` : "\n\nSummarize and remember the attached files.");
    attachments = [];
    renderAttach();
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  lastUser = t;
  if (!opts.silentUser) bubble("user", t.split("\n\n").pop() || t);
  thoughts.innerHTML = "";
  currentAgi = bubble("agi", "");
  currentAgi.classList.add("streaming");
  setThinking(true);
  ws.send(JSON.stringify({ type: "chat", message: t, session_id: sessionId }));
  input.value = "";
  $("chars").textContent = "0";
}

function handleSlash(t) {
  const [cmd, ...rest] = t.slice(1).split(/\s+/);
  const arg = rest.join(" ");
  if (cmd === "help") {
    bubble("agi", "Commands: /new /spawn Name mission /run Name goal /do q /research q /compare a vs b /note t /todo t /search q /teach /stop /focus /agents", true);
    return true;
  }
  if (cmd === "new") { newChat(); return true; }
  if (cmd === "clear") { clearStage(); return true; }
  if (cmd === "export") { exportThread(); return true; }
  if (cmd === "improve") {
    ws.send(JSON.stringify({ type: "improve", message: "improve", session_id: sessionId }));
    return true;
  }
  if (cmd === "stop") {
    ws.send(JSON.stringify({ type: "stop" }));
    setThinking(false);
    return true;
  }
  if (cmd === "focus") {
    settings.focus = !settings.focus;
    saveSettings();
    return true;
  }
  if (cmd === "search" && arg) {
    send(`search memory for ${arg}`);
    return true;
  }
  if (cmd === "teach" && arg) {
    send(`learn this: ${arg}`);
    return true;
  }
  if (cmd === "research" && arg) {
    send("Research this: " + arg);
    return true;
  }
  if ((cmd === "do" || cmd === "agent") && arg) {
    send("Do: " + arg, { noAgent: true });
    return true;
  }
  if ((cmd === "spawn" || cmd === "create") && arg) {
    send("create agent " + arg, { noAgent: true });
    return true;
  }
  if (cmd === "run" && arg) {
    send("run " + arg, { noAgent: true });
    return true;
  }
  if (cmd === "agents") {
    showTab("agents");
    loadAgents();
    return true;
  }
  if (cmd === "compare" && arg) {
    send("Compare " + arg);
    return true;
  }
  if (cmd === "note" && arg) {
    send("Note: " + arg);
    return true;
  }
  if ((cmd === "todo" || cmd === "task") && arg) {
    send("Todo: " + arg);
    return true;
  }
  if (cmd === "think" && arg) {
    send(arg);
    return true;
  }
  return false;
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  send();
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && settings.enter) {
    e.preventDefault();
    send();
  }
});
input.addEventListener("input", () => {
  $("chars").textContent = String(input.value.length);
  input.style.height = "auto";
  input.style.height = Math.min(180, input.scrollHeight) + "px";
});

document.addEventListener("keydown", (e) => {
  const meta = e.ctrlKey || e.metaKey;
  if (meta && e.key.toLowerCase() === "k") {
    e.preventDefault();
    openPalette();
  }
  if (meta && e.key.toLowerCase() === "l") {
    e.preventDefault();
    clearStage();
  }
  if (meta && e.key === ".") {
    e.preventDefault();
    settings.focus = !settings.focus;
    saveSettings();
  }
  if (e.key === "Escape") {
    $("palette").hidden = true;
    $("settings").hidden = true;
  }
});

$("new-chat").onclick = () => newChat();
$("clear-btn").onclick = clearStage;
$("stop-btn").onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
  setThinking(false);
};
$("regen-btn").onclick = () => {
  if (lastUser) send(lastUser, { silentUser: false });
};
$("export-btn").onclick = exportThread;
$("cycle-btn").onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    addThought("improve", "manual cycle requested");
    ws.send(JSON.stringify({ type: "improve", message: "improve" }));
  }
};
$("focus-btn").onclick = () => {
  settings.focus = !settings.focus;
  saveSettings();
};
$("settings-btn").onclick = () => ($("settings").hidden = false);
$("palette-btn").onclick = openPalette;
$("toggle-chats").onclick = () => document.body.classList.toggle("show-chats");
$("attach-btn").onclick = () => $("file").click();
$("file").onchange = () => ingestFiles($("file").files);
$("reload-model").onclick = () => fetch("/api/reload-model", { method: "POST" });
$("opt-speak").onchange = (e) => { settings.speak = e.target.checked; saveSettings(); };
$("opt-enter").onchange = (e) => { settings.enter = e.target.checked; saveSettings(); };
$("opt-chain").onchange = (e) => { settings.chain = e.target.checked; saveSettings(); };
$("opt-focus").onchange = (e) => { settings.focus = e.target.checked; saveSettings(); };
$("settings").addEventListener("click", (e) => {
  if (e.target.id === "settings") $("settings").hidden = true;
});
$("palette").addEventListener("click", (e) => {
  if (e.target.id === "palette") $("palette").hidden = true;
});

$("mic-btn").onclick = () => {
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Rec) {
    addThought("error", "no speech recognition in this browser");
    return;
  }
  const rec = new Rec();
  rec.lang = "en-IN";
  rec.onresult = (ev) => {
    input.value = (input.value + " " + ev.results[0][0].transcript).trim();
    $("chars").textContent = String(input.value.length);
  };
  rec.start();
  $("mic-btn").classList.add("on");
  rec.onend = () => $("mic-btn").classList.remove("on");
};

["opt-speak", "opt-enter", "opt-chain", "opt-focus"].forEach(() => {});
saveSettings();

function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach((x) => x.classList.toggle("on", x.id === "tab-" + name));
}
document.querySelectorAll("#tabs button").forEach((b) => {
  b.onclick = () => showTab(b.dataset.tab);
});

$("art-copy").onclick = () => navigator.clipboard.writeText(artifact.code || "");
$("art-run").onclick = () => {
  if (!artifact.code) return;
  if (artifact.lang === "html") {
    $("art-frame").hidden = false;
    $("art-frame").srcdoc = artifact.code;
  } else {
    $("art-frame").hidden = true;
  }
};

let searchTimer = 0;
$("mem-q").addEventListener("input", (e) => {
  const q = e.target.value.trim();
  clearTimeout(searchTimer);
  if (!q) {
    if (state) renderFacts(state.facts || []);
    return;
  }
  searchTimer = setTimeout(() => {
    fetch("/api/search?q=" + encodeURIComponent(q))
      .then((r) => r.json())
      .then((d) => {
        const rows = [];
        for (const a of d.knowledge || []) rows.push({ subject: "know", predicate: a.title, object: a.body.slice(0, 80) });
        for (const f of d.facts || []) rows.push(f);
        for (const ep of d.episodes || []) rows.push({ subject: ep.role, predicate: ep.score.toFixed(2), object: ep.content.slice(0, 80) });
        renderFacts(rows);
      })
      .catch(() => {});
  }, 180);
});
$("chat-q").addEventListener("input", () => renderChatList($("chat-q").value));

function clearStage() {
  log.innerHTML = "";
  thoughts.innerHTML = "";
}
function setThinking(v) {
  thinking = v;
  sendBtn.disabled = v;
}
function esc(s) {
  return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}
function md(raw) {
  let s = esc(raw);
  s = s.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
    return `<div class="pre-wrap"><button class="copy-code" type="button">copy</button><pre data-lang="${esc(lang || "")}">${code}</pre></div>`;
  });
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  s = s.replace(/^## (.+)$/gm, "<h3>$1</h3>");
  s = s.replace(/^# (.+)$/gm, "<h3>$1</h3>");
  s = s.replace(/^\- (.+)$/gm, "<li>$1</li>");
  s = s.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");
  s = s.replace(/\n\n/g, "</p><p>");
  return `<p>${s}</p>`;
}
function decorateCode(root) {
  root.querySelectorAll(".copy-code").forEach((b) => {
    b.onclick = () => {
      const pre = b.parentElement.querySelector("pre");
      navigator.clipboard.writeText(pre ? pre.textContent : "");
      b.textContent = "copied";
      setTimeout(() => (b.textContent = "copy"), 800);
    };
  });
}
function harvestArtifact(raw) {
  const m = raw.match(/```(\w+)?\n([\s\S]*?)```/);
  if (!m) return;
  artifact = { lang: (m[1] || "txt").toLowerCase(), code: m[2] };
  $("art-label").textContent = artifact.lang;
  $("art-code").textContent = artifact.code;
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x.dataset.tab === "artifact"));
  document.querySelectorAll(".tab-body").forEach((x) => x.classList.toggle("on", x.id === "tab-artifact"));
}
function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.slice(0, 1200));
  u.rate = 1.02;
  speechSynthesis.speak(u);
}

function renderFacts(fl) {
  const facts = $("facts");
  facts.innerHTML = "";
  if (!fl.length) facts.textContent = "No semantic facts yet.";
  for (const f of fl.slice(0, 12)) {
    const d = document.createElement("div");
    d.textContent = `${f.subject} ${f.predicate} ${f.object}`;
    facts.appendChild(d);
  }
}

function renderState(s) {
  state = s;
  const ident = s.identity || {};
  const engine = s.engine || {};
  const neural = s.neural || {};
  $("self-desc").textContent = ident.self_description || "";
  const core = s.core || {};
  const coreKv = $("core-kv");
  if (coreKv) {
    coreKv.innerHTML = "";
    const rows = [
      ["source", engine.source],
      ["backend", engine.backend],
      ["gguf", core.gguf ? "present" : "drop model/model.gguf"],
      ["safetensors", core.safetensors ? "present" : "will write as I train"],
      ["dir", core.model_dir],
    ];
    for (const [k, v] of rows) {
      const li = document.createElement("li");
      li.innerHTML = `<span>${esc(k)}</span><b>${esc(v)}</b>`;
      coreKv.appendChild(li);
    }
  }
  $("src-pill").textContent = `${engine.source || "?"} · ${engine.backend || "?"}`;
  $("src-pill").className = "pill" + (engine.warning ? " warn" : "");
  $("turn-pill").textContent = `turns ${ident.turns ?? 0}`;
  $("cycle-pill").textContent = `cycles ${ident.cycles ?? 0}`;
  const loss = neural.last_loss;
  $("loss-pill").textContent = loss == null ? "loss —" : `loss ${Number(loss).toFixed(3)}`;
  if (s.latency_ms) $("lat-pill").textContent = `${Math.round(s.latency_ms)} ms`;

  const kv = $("identity-kv");
  kv.innerHTML = "";
  const tr = neural.trainer || {};
  const rows = [
    ["name", ident.name],
    ["constitution", `v${ident.constitution_version}`],
    ["params", (engine.params || neural.params || "—").toLocaleString?.() || engine.params || neural.params],
    ["neural steps", neural.steps ?? 0],
    ["trainer", tr.busy ? `busy q=${tr.queue}` : `idle q=${tr.queue ?? 0}`],
    ["episodes", s.memory?.episodes ?? 0],
    ["facts", s.memory?.facts ?? 0],
    ["taught", (s.taught || []).length],
  ];
  for (const [k, v] of rows) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${esc(k)}</span><b>${esc(v)}</b>`;
    kv.appendChild(li);
  }
  const pr = $("principles");
  pr.innerHTML = "";
  for (const p of (ident.principles || []).slice(-8)) {
    const li = document.createElement("li");
    li.textContent = p;
    pr.appendChild(li);
  }
  const goals = $("goals");
  goals.innerHTML = "";
  for (const g of s.goals || []) {
    const d = document.createElement("div");
    d.className = "g";
    const pct = Math.round((g.progress || 0) * 100);
    d.innerHTML = `<div class="t">${esc(g.title)} · ${pct}%</div><div class="bar"><i style="width:${pct}%"></i></div>`;
    goals.appendChild(d);
  }
  renderFacts(s.facts || []);
  const skills = $("skills");
  skills.innerHTML = "";
  const sl = s.skills || [];
  if (!sl.length) skills.textContent = "No skills loaded.";
  for (const sk of sl) {
    const d = document.createElement("div");
    d.textContent = `${sk.name} — ${sk.description}`;
    skills.appendChild(d);
  }
  drawChart(neural.loss_history || (s.metrics?.loss || []).map((x) => x[1]));
}

function drawChart(hist) {
  const c = $("chart");
  if (!c) return;
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  if (!hist || hist.length < 2) {
    ctx.fillStyle = "#7d8aa3";
    ctx.font = "11px IBM Plex Mono";
    ctx.fillText("loss plots as the neural core trains", 8, h / 2);
    return;
  }
  const min = Math.min(...hist);
  const max = Math.max(...hist);
  const span = Math.max(1e-6, max - min);
  ctx.beginPath();
  hist.forEach((v, i) => {
    const x = (i / (hist.length - 1)) * (w - 8) + 4;
    const y = h - 8 - ((v - min) / span) * (h - 16);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#3dffc8";
  ctx.lineWidth = 1.6;
  ctx.stroke();
}

async function loadSessions() {
  const d = await fetch("/api/sessions").then((r) => r.json());
  sessions = d.sessions || [];
  if (!sessionId && sessions[0]) sessionId = sessions[0].id;
  if (!sessionId) {
    const created = await fetch("/api/sessions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    }).then((r) => r.json());
    sessionId = created.id;
    sessions = [created, ...sessions];
  }
  localStorage.setItem("cortex.session", sessionId);
  renderChatList($("chat-q").value);
  const cur = sessions.find((s) => s.id === sessionId);
  $("thread-title").textContent = cur ? cur.title : "new thread";
}

function renderChatList(q) {
  const box = $("chat-list");
  box.innerHTML = "";
  const n = (q || "").toLowerCase();
  for (const s of sessions) {
    if (n && !(`${s.title} ${s.preview}`.toLowerCase().includes(n))) continue;
    const d = document.createElement("div");
    d.className = "chat-item" + (s.id === sessionId ? " on" : "");
    d.innerHTML = `<div>${esc(s.pinned ? "★ " : "")}${esc(s.title)}</div><div class="meta">${s.count} msgs</div>`;
    d.onclick = () => openSession(s.id);
    d.oncontextmenu = (e) => {
      e.preventDefault();
      if (confirm("delete this thread?")) {
        fetch("/api/sessions/" + s.id, { method: "DELETE" }).then(() => {
          if (sessionId === s.id) sessionId = "";
          loadSessions().then(() => {
            if (sessionId) openSession(sessionId);
          });
        });
      }
    };
    box.appendChild(d);
  }
}

async function openSession(id) {
  sessionId = id;
  localStorage.setItem("cortex.session", id);
  const s = await fetch("/api/sessions/" + id).then((r) => r.json());
  $("thread-title").textContent = s.title || "thread";
  log.innerHTML = "";
  for (const m of s.messages || []) {
    const el = bubble(m.role === "user" ? "user" : "agi", m.content, true);
    if (m.meta && m.meta.chain) attachChain(el, m.meta.chain);
  }
  renderChatList($("chat-q").value);
}

async function newChat() {
  const s = await fetch("/api/sessions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ title: "new thread" }),
  }).then((r) => r.json());
  sessionId = s.id;
  localStorage.setItem("cortex.session", sessionId);
  log.innerHTML = "";
  thoughts.innerHTML = "";
  await loadSessions();
}

async function exportThread() {
  if (!sessionId) return;
  const s = await fetch("/api/sessions/" + sessionId).then((r) => r.json());
  const blob = new Blob([JSON.stringify(s, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${s.title || "thread"}.json`;
  a.click();
}

function renderAttach() {
  const strip = $("attach-strip");
  strip.innerHTML = "";
  for (const a of attachments) {
    const d = document.createElement("span");
    d.className = "attach-chip";
    d.textContent = a.name;
    strip.appendChild(d);
  }
}

async function ingestFiles(fileList) {
  for (const file of fileList || []) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/ingest", { method: "POST", body: fd }).then((r) => r.json());
    if (res.ok) {
      attachments.push({ name: res.name, excerpt: res.excerpt });
      addThought("act", `ingested ${res.name} (${res.chars} chars)`);
    } else addThought("error", res.error || "ingest failed");
  }
  renderAttach();
  loadVault();
}

async function loadDocs() {
  const d = await fetch("/api/docs").then((r) => r.json()).catch(() => ({ docs: [] }));
  const box = $("doc-list");
  if (!box) return;
  box.innerHTML = "";
  for (const doc of d.docs || []) {
    const el = document.createElement("div");
    el.textContent = `${doc.kind} · ${doc.title} · ${doc.chars}c`;
    el.onclick = async () => {
      const full = await fetch("/api/docs/" + doc.id).then((r) => r.json());
      artifact = { lang: "md", code: full.body || "" };
      $("art-label").textContent = full.title;
      $("art-code").textContent = artifact.code;
      showTab("artifact");
    };
    el.oncontextmenu = (e) => {
      e.preventDefault();
      if (confirm("delete this document?")) {
        fetch("/api/docs/" + doc.id, { method: "DELETE" }).then(loadDocs);
      }
    };
    box.appendChild(el);
  }
  const rail = $("rail-docs");
  const n = $("rail-doc-n");
  if (n) n.textContent = String((d.docs || []).length);
  if (rail) {
    rail.innerHTML = "";
    for (const doc of (d.docs || []).slice(0, 6)) {
      const el = document.createElement("div");
      el.className = "chat-item";
      el.innerHTML = `<div>${esc(doc.title)}</div><div class="meta">${esc(doc.kind)}</div>`;
      el.onclick = () => {
        fetch("/api/docs/" + doc.id).then((r) => r.json()).then((full) => {
          artifact = { lang: "md", code: full.body || "" };
          $("art-label").textContent = full.title;
          $("art-code").textContent = artifact.code;
          showTab("artifact");
        });
      };
      rail.appendChild(el);
    }
  }
}

async function loadTasks() {
  const d = await fetch("/api/tasks").then((r) => r.json()).catch(() => ({ tasks: [] }));
  const box = $("task-list");
  if (!box) return;
  box.innerHTML = "";
  for (const t of d.tasks || []) {
    const el = document.createElement("div");
    el.className = "task" + (t.done ? " done" : "");
    el.innerHTML = `<input type="checkbox" ${t.done ? "checked" : ""} /><span></span><button type="button" class="x">×</button>`;
    el.querySelector("span").textContent = t.title;
    el.querySelector("input").onchange = (e) => {
      fetch("/api/tasks/" + t.id, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ done: e.target.checked }),
      }).then(loadTasks);
    };
    el.querySelector(".x").onclick = () => {
      fetch("/api/tasks/" + t.id, { method: "DELETE" }).then(loadTasks);
    };
    box.appendChild(el);
  }
  const rail = $("rail-tasks");
  const n = $("rail-task-n");
  const open = (d.tasks || []).filter((t) => !t.done);
  if (n) n.textContent = String(open.length);
  if (rail) {
    rail.innerHTML = "";
    for (const t of open.slice(0, 6)) {
      const el = document.createElement("div");
      el.className = "chat-item";
      el.textContent = t.title;
      el.onclick = () => showTab("tasks");
      rail.appendChild(el);
    }
  }
}

async function loadAgents() {
  const d = await fetch("/api/agents").then((r) => r.json()).catch(() => ({ agents: [] }));
  const agents = d.agents || [];
  const box = $("agent-list");
  if (box) {
    box.innerHTML = "";
    for (const a of agents) {
      const el = document.createElement("div");
      el.textContent = `${a.name} · ${(a.tools || []).join(", ")} · ${a.runs} runs`;
      el.onclick = () => {
        if ($("agent-sel")) $("agent-sel").value = a.name;
        input.value = `run ${a.name}: `;
        input.focus();
      };
      el.oncontextmenu = (e) => {
        e.preventDefault();
        if (confirm("retire " + a.name + "?")) {
          fetch("/api/agents/" + a.id, { method: "DELETE" }).then(loadAgents);
        }
      };
      box.appendChild(el);
    }
  }
  const rail = $("rail-agents");
  const n = $("rail-agent-n");
  if (n) n.textContent = String(agents.length);
  if (rail) {
    rail.innerHTML = "";
    for (const a of agents.slice(0, 8)) {
      const el = document.createElement("div");
      el.className = "chat-item";
      el.innerHTML = `<div>${esc(a.name)}</div><div class="meta">${esc((a.tools || []).slice(0, 3).join(" "))}</div>`;
      el.onclick = () => {
        if ($("agent-sel")) $("agent-sel").value = a.name;
        showTab("agents");
      };
      rail.appendChild(el);
    }
  }
  const sel = $("agent-sel");
  if (sel) {
    const cur = sel.value;
    sel.innerHTML = `<option value="">CORTEX</option>`;
    for (const a of agents) {
      const o = document.createElement("option");
      o.value = a.name;
      o.textContent = a.name;
      sel.appendChild(o);
    }
    if (cur) sel.value = cur;
  }
}

async function loadVault() {
  const d = await fetch("/api/vault").then((r) => r.json());
  const box = $("vault-list");
  box.innerHTML = "";
  for (const f of d.files || []) {
    const el = document.createElement("div");
    el.textContent = `${f.name} · ${f.bytes}B`;
    el.onclick = async () => {
      const body = await fetch("/api/vault/" + encodeURIComponent(f.name)).then((r) => r.json());
      artifact = { lang: "txt", code: body.text || "" };
      $("art-label").textContent = f.name;
      $("art-code").textContent = artifact.code;
    };
    box.appendChild(el);
  }
}

function openPalette() {
  $("palette").hidden = false;
  const q = $("palette-q");
  q.value = "";
  q.focus();
  palItems = [
    { title: "New thread", sub: "/new", run: () => newChat() },
    { title: "Research", sub: "/research", run: () => { input.value = "Research this: "; input.focus(); } },
    { title: "Do (agent)", sub: "/do", run: () => { input.value = "Do: "; input.focus(); } },
    { title: "Spawn agent", sub: "/spawn", run: () => { input.value = "create agent "; input.focus(); } },
    { title: "Run agent", sub: "/run", run: () => { input.value = "run Operator: "; input.focus(); } },
    { title: "Crew", sub: "agents", run: () => { showTab("agents"); loadAgents(); } },
    { title: "Compare", sub: "/compare", run: () => { input.value = "Compare "; input.focus(); } },
    { title: "New note", sub: "/note", run: () => { input.value = "Note: "; input.focus(); } },
    { title: "Add task", sub: "/todo", run: () => { input.value = "Todo: "; input.focus(); } },
    { title: "Focus mode", sub: "Ctrl+.", run: () => { settings.focus = !settings.focus; saveSettings(); } },
    { title: "Improve CORTEX", sub: "/improve", run: () => handleSlash("/improve") },
    { title: "Export thread", sub: "JSON", run: exportThread },
    { title: "Reload model", sub: "model/", run: () => fetch("/api/reload-model", { method: "POST" }) },
    ...prompts.map((p) => ({
      title: p.title,
      sub: "prompt",
      run: () => {
        input.value = p.body;
        input.focus();
      },
    })),
    ...sessions.slice(0, 12).map((s) => ({
      title: s.title,
      sub: "thread",
      run: () => openSession(s.id),
    })),
  ];
  renderPalette("");
}
function renderPalette(q) {
  const n = q.toLowerCase();
  const list = $("palette-list");
  list.innerHTML = "";
  palIdx = 0;
  const shown = palItems.filter((p) => !n || `${p.title} ${p.sub}`.toLowerCase().includes(n)).slice(0, 16);
  palItems._shown = shown;
  shown.forEach((p, i) => {
    const d = document.createElement("div");
    d.className = "pal-item" + (i === 0 ? " on" : "");
    d.innerHTML = `${esc(p.title)}<small>${esc(p.sub || "")}</small>`;
    d.onclick = () => {
      $("palette").hidden = true;
      p.run();
    };
    list.appendChild(d);
  });
}
$("palette-q").addEventListener("input", (e) => renderPalette(e.target.value));
$("palette-q").addEventListener("keydown", (e) => {
  const shown = palItems._shown || [];
  if (e.key === "ArrowDown") {
    e.preventDefault();
    palIdx = Math.min(shown.length - 1, palIdx + 1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    palIdx = Math.max(0, palIdx - 1);
  } else if (e.key === "Enter") {
    e.preventDefault();
    $("palette").hidden = true;
    if (shown[palIdx]) shown[palIdx].run();
    return;
  } else return;
  [...$("palette-list").children].forEach((c, i) => c.classList.toggle("on", i === palIdx));
});

["dragenter", "dragover"].forEach((ev) =>
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    document.body.classList.add("drop");
  })
);
["dragleave", "drop"].forEach((ev) =>
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "drop" && e.dataTransfer.files.length) ingestFiles(e.dataTransfer.files);
    document.body.classList.remove("drop");
  })
);

(function orb() {
  const c = $("orb");
  const ctx = c.getContext("2d");
  const pts = Array.from({ length: 24 }, () => ({
    a: Math.random() * Math.PI * 2,
    b: Math.random() * Math.PI,
    r: 0.35 + Math.random() * 0.6,
    s: 0.004 + Math.random() * 0.01,
  }));
  function frame() {
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(61,255,200,0.25)";
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 13, 0, Math.PI * 2);
    ctx.stroke();
    const pulse = thinking ? 1.4 : 1;
    for (const p of pts) {
      p.a += p.s * pulse;
      const x = w / 2 + Math.cos(p.a) * Math.sin(p.b) * 13 * p.r;
      const y = h / 2 + Math.sin(p.a) * Math.sin(p.b) * 13 * p.r;
      ctx.fillStyle = thinking ? "#3dffc8" : "rgba(122,162,255,0.9)";
      ctx.beginPath();
      ctx.arc(x, y, thinking ? 1.5 : 1.0, 0, Math.PI * 2);
      ctx.fill();
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

connect();
fetch("/api/state").then((r) => r.json()).then(renderState).catch(() => {});
fetch("/api/prompts")
  .then((r) => r.json())
  .then((d) => {
    prompts = d.prompts || [];
    const chips = $("chips");
    chips.innerHTML = "";
    for (const p of prompts) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = p.title;
      b.onclick = () => {
        if (p.body.trim() === "Improve yourself") send(p.body);
        else {
          input.value = p.body;
          input.focus();
        }
      };
      chips.appendChild(b);
    }
  })
  .catch(() => {});
loadSessions().then(() => {
  if (sessionId) openSession(sessionId);
});
loadVault();
loadDocs();
loadTasks();
loadAgents();

$("doc-new").onclick = async () => {
  const title = $("doc-title").value.trim();
  if (!title) return;
  await fetch("/api/docs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ title, body: artifact.code || "", kind: "note" }),
  });
  $("doc-title").value = "";
  loadDocs();
};
$("agent-new").onclick = async () => {
  const name = $("agent-name").value.trim();
  const mission = $("agent-mission").value.trim();
  const tools = $("agent-tools").value.split(/[,/]/).map((s) => s.trim()).filter(Boolean);
  if (!name) return;
  await fetch("/api/agents", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name, mission, tools }),
  });
  $("agent-name").value = "";
  $("agent-mission").value = "";
  loadAgents();
};
$("task-new").onclick = async () => {
  const title = $("task-title").value.trim();
  if (!title) return;
  await fetch("/api/tasks", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ title }),
  });
  $("task-title").value = "";
  loadTasks();
};
$("task-title").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    $("task-new").click();
  }
});
$("doc-title").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    $("doc-new").click();
  }
});
