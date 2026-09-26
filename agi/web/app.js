const $ = (id) => document.getElementById(id);

const log = $("log");
const thoughts = $("thoughts");
const input = $("input");
const sendBtn = $("send");
const form = $("composer");

let ws;
let thinking = false;
let currentAgi = null;
let state = null;
let pingTimer = null;

function proto() {
  return location.protocol === "https:" ? "wss" : "ws";
}

function setConn(on) {
  const el = $("conn-pill");
  el.textContent = on ? "live" : "offline";
  el.className = "pill" + (on ? " live" : " dim");
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
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    handle(msg);
  };
}

function handle(msg) {
  if (msg.type === "hello" || msg.type === "state") {
    if (msg.state) renderState(msg.state);
    return;
  }
  if (msg.type === "pong") return;
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
      currentAgi.dataset.done = "1";
      const body = currentAgi.querySelector(".body");
      let raw = body.textContent;
      const cut = raw.indexOf("**Thinking**");
      if (cut > 0) raw = raw.slice(0, cut).trim();
      body.innerHTML = md(raw);
      if (msg.chain) attachChain(currentAgi, msg.chain);
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
    if (msg.events) {
      for (const e of msg.events) addThought("improve", `${e.kind}: ${e.text || ""}`);
    }
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
  det.open = chain.system === 2;
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
  else el.prepend(det);
}

function bubble(role, text, asHtml) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="who">${role === "user" ? "you" : "cortex"}</div><div class="body"></div>`;
  const body = el.querySelector(".body");
  if (asHtml) body.innerHTML = md(text);
  else body.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function appendToken(text) {
  if (!currentAgi) currentAgi = bubble("agi", "");
  const body = currentAgi.querySelector(".body");
  body.textContent += text;
  log.scrollTop = log.scrollHeight;
}

function send(text) {
  const t = (text || input.value).trim();
  if (!t || !ws || ws.readyState !== WebSocket.OPEN) return;
  bubble("user", t);
  thoughts.innerHTML = "";
  currentAgi = bubble("agi", "");
  setThinking(true);
  ws.send(JSON.stringify({ type: "chat", message: t }));
  input.value = "";
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  send();
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "l") {
    e.preventDefault();
    clearStage();
  }
});
document.querySelectorAll("#chips button").forEach((b) => {
  b.addEventListener("click", () => send(b.dataset.q));
});
$("cycle-btn").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    addThought("improve", "manual cycle requested");
    ws.send(JSON.stringify({ type: "improve", message: "improve" }));
  }
});
$("clear-btn").addEventListener("click", clearStage);
$("stop-btn").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
  setThinking(false);
});
$("export-btn").addEventListener("click", () => {
  window.open("/api/export", "_blank");
});

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

function clearStage() {
  log.innerHTML = "";
  thoughts.innerHTML = "";
}

function setThinking(v) {
  thinking = v;
  sendBtn.disabled = v;
}

function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function md(raw) {
  let s = esc(raw);
  s = s.replace(/```([\s\S]*?)```/g, (_, code) => `<pre>${code}</pre>`);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return s;
}

function renderFacts(fl) {
  const facts = $("facts");
  facts.innerHTML = "";
  if (!fl.length) facts.textContent = "No semantic facts yet.";
  for (const f of fl.slice(0, 10)) {
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
    ["lessons", s.memory?.lessons ?? 0],
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
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  if (!hist || hist.length < 2) {
    ctx.fillStyle = "#7d8aa3";
    ctx.font = "11px IBM Plex Mono";
    ctx.fillText("loss will plot as the neural core trains", 8, h / 2);
    return;
  }
  const min = Math.min(...hist);
  const max = Math.max(...hist);
  const span = Math.max(1e-6, max - min);
  ctx.beginPath();
  hist.forEach((v, i) => {
    const x = (i / (hist.length - 1)) * (w - 8) + 4;
    const y = h - 8 - ((v - min) / span) * (h - 16);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#3dffc8";
  ctx.lineWidth = 1.6;
  ctx.stroke();
}

(function orb() {
  const c = $("orb");
  const ctx = c.getContext("2d");
  const pts = Array.from({ length: 28 }, () => ({
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
    ctx.arc(w / 2, h / 2, 16, 0, Math.PI * 2);
    ctx.stroke();
    const pulse = thinking ? 1.4 : 1;
    for (const p of pts) {
      p.a += p.s * pulse;
      const x = w / 2 + Math.cos(p.a) * Math.sin(p.b) * 16 * p.r;
      const y = h / 2 + Math.sin(p.a) * Math.sin(p.b) * 16 * p.r;
      ctx.fillStyle = thinking ? "#3dffc8" : "rgba(122,162,255,0.9)";
      ctx.beginPath();
      ctx.arc(x, y, thinking ? 1.6 : 1.1, 0, Math.PI * 2);
      ctx.fill();
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

connect();

fetch("/api/state")
  .then((r) => r.json())
  .then(renderState)
  .catch(() => {});

fetch("/api/history?k=24")
  .then((r) => r.json())
  .then((d) => {
    const msgs = d.messages || [];
    if (msgs.length < 2) return;
    log.innerHTML = "";
    for (const m of msgs) {
      bubble(m.role === "user" ? "user" : "agi", m.content, true);
    }
  })
  .catch(() => {});
