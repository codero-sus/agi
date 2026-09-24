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

function proto() {
  return location.protocol === "https:" ? "wss" : "ws";
}

function connect() {
  ws = new WebSocket(`${proto()}://${location.host}/ws`);
  ws.onopen = () => setThinking(false);
  ws.onclose = () => setTimeout(connect, 1200);
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
    if (currentAgi) currentAgi.dataset.done = "1";
    currentAgi = null;
    setThinking(false);
    return;
  }
  if (msg.type === "improve") {
    if (msg.events) {
      for (const e of msg.events) addThought("improve", `${e.kind}: ${e.text || ""}`);
    }
    return;
  }
}

function addThought(kind, text) {
  const d = document.createElement("div");
  d.className = "t";
  d.innerHTML = `<span class="k">${esc(kind)}</span>${esc(text)}`;
  thoughts.prepend(d);
  while (thoughts.children.length > 40) thoughts.removeChild(thoughts.lastChild);
}

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="who">${role === "user" ? "you" : "cortex"}</div><div class="body"></div>`;
  el.querySelector(".body").textContent = text;
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
document.querySelectorAll("#chips button").forEach((b) => {
  b.addEventListener("click", () => send(b.dataset.q));
});
$("cycle-btn").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    addThought("improve", "manual cycle requested");
    ws.send(JSON.stringify({ type: "improve", message: "improve" }));
  }
});

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

  const kv = $("identity-kv");
  kv.innerHTML = "";
  const rows = [
    ["name", ident.name],
    ["constitution", `v${ident.constitution_version}`],
    ["params", (engine.params || neural.params || "—").toLocaleString?.() || engine.params || neural.params],
    ["neural steps", neural.steps ?? 0],
    ["episodes", s.memory?.episodes ?? 0],
    ["facts", s.memory?.facts ?? 0],
    ["lessons", s.memory?.lessons ?? 0],
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

  const facts = $("facts");
  facts.innerHTML = "";
  const fl = s.facts || [];
  if (!fl.length) facts.textContent = "No semantic facts yet.";
  for (const f of fl.slice(0, 10)) {
    const d = document.createElement("div");
    d.textContent = `${f.subject} ${f.predicate} ${f.object}`;
    facts.appendChild(d);
  }

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

/* living orb */
(function orb() {
  const c = $("orb");
  const ctx = c.getContext("2d");
  const pts = Array.from({ length: 28 }, () => ({
    a: Math.random() * Math.PI * 2,
    b: Math.random() * Math.PI,
    r: 0.35 + Math.random() * 0.6,
    s: 0.004 + Math.random() * 0.01,
  }));
  function frame(t) {
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
