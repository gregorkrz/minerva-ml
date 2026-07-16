#!/usr/bin/env python3
"""Build a self-contained interactive HTML event viewer for the demo datasets.

Reads the per-bin signal/background ``.pb`` datasets produced by
``src/scripts/extract_bin_demo_datasets.py`` (a ``manifest.json`` plus
``<task>/<bin>/<signal|background>/0.pb`` folders) and writes a single
standalone HTML file. The page lets you pick:

* the signal definition / binning task (CCN\u03c0\u00b1 vs W, CC1\u03c0\u00b1 / CC1\u03c0\u2070 vs pion E),
* the kinematic bin,
* signal or background,
* and step through individual events,

showing each event both as a **3D detector display** and in the **\u03b7-\u03c6 plane**.

In 3D, particles with reconstructed positions are drawn at their detector
coordinates (blobs / prongs); prongs additionally get a momentum-direction
arrow, while muons and photons (which carry no stored position) are drawn as
arrows from the estimated interaction vertex (mean of the positioned hits).

An optional "Show MC truth" toggle overlays the MC truth pion direction (from
``truth_labels`` columns 11-14) as a dashed magenta arrow / star marker. Note
the processed dataset only retains the truth pion four-vector, not the full
``mc_FSPart*`` list, so that is the only MC particle available here.

The data is embedded directly in the HTML (Plotly.js is loaded from a CDN), so
the output file is fully portable.

Per-particle feature layout (see ``DATASET.md``): columns
``[eta, phi, log(pT), log(E), pid, dE/dx, x/1e4, y/1e4, z/1e4, t/1e4]``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_INPUT_DIR = Path(
    "/global/cfs/cdirs/m3246/gregork/Minerva/20260326_NEW_DEMO_ONLY"
)
DEFAULT_OUTPUT = _REPO_ROOT / "plots" / "event_viewer.html"

PID_NAMES = {
    0: "Muon",
    1: "Photon",
    2: "Blob",
    3: "Prong (pion)",
    4: "Prong (EM)",
    5: "Prong (muon-like)",
    6: "Aggregated blob",
    7: "Aggregated prong",
}

PID_COLORS = {
    0: "#e6194b",  # muon - red
    1: "#ffe119",  # photon - yellow
    2: "#911eb4",  # blob - purple
    3: "#4363d8",  # prong pion - blue
    4: "#3cb44b",  # prong EM - green
    5: "#f58231",  # prong muon-like - orange
    6: "#a9a9a9",  # aggregated blob - grey
    7: "#42d4f4",  # aggregated prong - cyan
}

PID_LABEL_MEANING = {
    0: "CC 1 charged pion",
    1: "CC >1 charged pion",
    2: "CC 1 pi0, no charged pions",
    3: "CC other",
    4: "NC",
}


def _to_float(x) -> float:
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def event_to_particles(feat: np.ndarray) -> list[list]:
    """Convert one ``(n_particles, 10)`` feature array to a compact list.

    Each particle becomes ``[pid, eta, phi, pt, E, x, y, z, dedx]`` with x/y/z
    scaled back to detector coordinates (the stored values are divided by 1e4)
    and ``dedx`` inverted from its preprocessed form back to a physical value.

    The stored dE/dx is ``log(|raw| + 0.1)`` (see ``preprocessing.preprocess_dEdX``);
    we recover ``raw = max(exp(stored) - 0.1, 0)``. Muons and blobs carry a hard
    ``0`` (not applicable) which we report as ``None``.
    """
    particles: list[list] = []
    for row in feat:
        eta = _to_float(row[0])
        phi = _to_float(row[1])
        pt = math.exp(_to_float(row[2])) - 1e-6
        E = math.exp(_to_float(row[3])) - 1e-6
        pid = int(round(_to_float(row[4])))
        dedx_pre = _to_float(row[5])
        if dedx_pre == 0.0:  # muons / blobs: dE/dx not applicable
            dedx = None
        else:
            dedx = round(max(math.exp(dedx_pre) - 0.1, 0.0), 4)
        x = _to_float(row[6]) * 1e4
        y = _to_float(row[7]) * 1e4
        z = _to_float(row[8]) * 1e4
        particles.append(
            [
                pid,
                round(eta, 4),
                round(phi, 4),
                round(max(pt, 0.0), 3),
                round(max(E, 0.0), 3),
                round(x, 2),
                round(y, 2),
                round(z, 2),
                dedx,
            ]
        )
    return particles


def _label_from_run(run: str) -> str:
    """Best-effort human-readable label from a run name when scores.json lacks one.

    Mirrors ``eval_demo_datasets._fallback_model_name``/``parse_seed`` so the
    viewer table reads nicely even for older scores.json files.
    """
    r = run
    name = run
    m = re.search(r"HyperScale_(small|medium)(_rw)?", r)
    if m:
        name = f"HyperScale-{m.group(1)}" + ("-rw" if m.group(2) else "")
    elif "BERT_tiny_rw" in r:
        name = "BERT-tiny-rw"
    elif "BERT_tiny_energy_order" in r:
        name = "BERT-tiny-energy-order"
    elif "BERT_tiny" in r:
        name = "BERT-tiny"
    elif "OLS_RW" in r:
        name = "OmniLearned-small-rw"
    elif "OLS_int" in r:
        name = "OmniLearned-small-int"
    elif re.search(r"\bOLM\b", r) or "_OLM_" in r:
        name = "OmniLearned-medium"
    elif "OLS" in r:
        name = "OmniLearned-small"
    elif "Transformer2" in r:
        name = "Transformer2-DIS" if "DIS_only" in r else "Transformer2"
    elif "Transformer1" in r:
        name = "Transformer-xsmall"
    elif "cond_only" in r or "_MLP" in r or r.startswith("MLP"):
        name = "MLP"
    seed_m = re.search(r"seed_?(\d+)", r)
    return f"{name} (seed {seed_m.group(1)})" if seed_m else name


def mc_particles_from_truth(tl_row: np.ndarray) -> list[dict]:
    """MC-truth particle overlay from a truth-labels row.

    The processed dataset only retains the MC truth **pion** four-vector
    (``truth_labels`` columns 11-14, nonzero only for single-pion CC events;
    see ``DATASET.md``); full ``mc_FSPart*`` lists are not stored. Returns a
    direction unit vector plus (eta, phi, E) for the viewer.
    """
    px, py, pz, E = (
        _to_float(tl_row[11]),
        _to_float(tl_row[12]),
        _to_float(tl_row[13]),
        _to_float(tl_row[14]),
    )
    p = math.sqrt(px * px + py * py + pz * pz)
    if E <= 0.0 or p <= 0.0:
        return []
    eta = 0.5 * math.log(max((p + pz), 1e-9) / max((p - pz), 1e-9))
    eta = max(-10.0, min(10.0, eta))
    phi = math.atan2(py, px)
    return [
        {
            "label": "π (MC truth)",
            "eta": round(eta, 4),
            "phi": round(phi, 4),
            "E": round(max(E, 0.0), 3),
            "dir": [round(px / p, 5), round(py / p, 5), round(pz / p, 5)],
        }
    ]


def load_dataset_events(pb_path: Path, meta_path: Path | None) -> list[dict]:
    """Load one ``0.pb`` (+ optional ``meta.json``) into a list of event dicts."""
    blob = torch.load(pb_path, weights_only=False, map_location="cpu")
    per_event = list(blob["data"].unbind())
    truth = blob.get("truth_labels")
    if truth is not None and not torch.is_tensor(truth):
        truth = torch.as_tensor(truth)
    truth_np = truth.detach().cpu().numpy() if truth is not None else None

    meta = {}
    if meta_path is not None and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    pid_classes = meta.get("pid_classes", [])
    bin_values = meta.get("bin_values", [])
    global_idx = meta.get("playlist_global_index", [])

    events = []
    for i, feat in enumerate(per_event):
        arr = feat.detach().cpu().numpy()
        mc = (
            mc_particles_from_truth(truth_np[i])
            if truth_np is not None and i < len(truth_np)
            else []
        )
        events.append(
            {
                "pid_class": int(pid_classes[i]) if i < len(pid_classes) else -1,
                "bin_value": _to_float(bin_values[i]) if i < len(bin_values) else None,
                "global_index": int(global_idx[i]) if i < len(global_idx) else None,
                "particles": event_to_particles(arr),
                "mc": mc,
            }
        )
    return events


def build_payload(input_dir: Path) -> dict:
    """Walk the manifest (or glob) and assemble the embedded JSON payload."""
    manifest_path = input_dir / "manifest.json"
    tasks_out: list[dict] = []
    payload_meta: dict = {"source": str(input_dir)}

    # Optional model scores (written by eval_demo_datasets.py).
    scores_path = input_dir / "scores.json"
    scores_blob: dict = {}
    if scores_path.exists():
        with open(scores_path) as f:
            scores_blob = json.load(f)
        print(f"  loaded scores.json ({len(scores_blob.get('models', {}))} models)")
    scores_map: dict = scores_blob.get("scores", {})
    models_meta: dict = scores_blob.get("models", {})
    runs_order: list = scores_blob.get("runs", list(models_meta.keys()))

    def attach_scores(events_list: list[dict], rel_key: str) -> None:
        ds = scores_map.get(rel_key, {})
        if not ds:
            return
        for i, ev in enumerate(events_list):
            sc: dict = {}
            for run in runs_order:
                entry = ds.get(run)
                pred = entry.get("prediction") if entry is not None else None
                if pred is not None and i < len(pred):
                    sc[run] = pred[i]
            if sc:
                ev["scores"] = sc

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        payload_meta.update(
            {
                "playlist": manifest.get("playlist"),
                "split": manifest.get("split"),
                "n_events_per_bin_class": manifest.get("n_events_per_bin_class"),
            }
        )
        task_items = manifest.get("tasks", {}).items()
        for task_name, task in task_items:
            bins_out = []
            for bin_entry in task.get("bins", []):
                classes_out = {}
                for cls in ("signal", "background"):
                    if cls not in bin_entry:
                        continue
                    rel = bin_entry[cls]["path"]
                    ds_dir = input_dir / rel
                    pb = ds_dir / "0.pb"
                    if not pb.exists():
                        print(f"  WARNING: missing {pb}, skipping")
                        continue
                    events = load_dataset_events(pb, ds_dir / "meta.json")
                    attach_scores(events, rel)
                    classes_out[cls] = events
                    print(
                        f"  {task_name}/{bin_entry['label']}/{cls}: "
                        f"{len(events)} events"
                    )
                if classes_out:
                    bins_out.append(
                        {
                            "label": bin_entry["label"],
                            "lo": bin_entry.get("lo"),
                            "hi": bin_entry.get("hi"),
                            "classes": classes_out,
                        }
                    )
            if bins_out:
                tasks_out.append(
                    {
                        "name": task_name,
                        "title": task.get("title", task_name),
                        "bin_unit": task.get("bin_variable", ""),
                        "signal_pid_classes": task.get("signal_pid_classes", []),
                        "bins": bins_out,
                    }
                )
    else:
        print(f"No manifest.json in {input_dir}; globbing for 0.pb files…")
        for pb in sorted(input_dir.glob("*/*/*/0.pb")):
            cls = pb.parent.name
            bin_label = pb.parent.parent.name
            task_name = pb.parent.parent.parent.name
            task = next((t for t in tasks_out if t["name"] == task_name), None)
            if task is None:
                task = {
                    "name": task_name,
                    "title": task_name,
                    "bin_unit": "",
                    "signal_pid_classes": [],
                    "bins": [],
                }
                tasks_out.append(task)
            b = next((bb for bb in task["bins"] if bb["label"] == bin_label), None)
            if b is None:
                b = {"label": bin_label, "lo": None, "hi": None, "classes": {}}
                task["bins"].append(b)
            events = load_dataset_events(pb, pb.parent / "meta.json")
            attach_scores(events, str(pb.parent.relative_to(input_dir)))
            b["classes"][cls] = events
            print(f"  {task_name}/{bin_label}/{cls}: {len(events)} events")

    models_out = []
    for run in runs_order:
        m = models_meta.get(run, {})
        models_out.append(
            {
                "run": run,
                "label": m.get("label") or _label_from_run(run),
                "model": m.get("model"),
                "seed": m.get("seed"),
                "mode": m.get("mode", "classifier"),
                "num_classes": m.get("num_classes"),
                "class_idx": m.get("class_idx"),
                "binary_signal_pid_classes": m.get("binary_signal_pid_classes"),
            }
        )
    # Disambiguate models that resolve to the same human-readable label
    # (e.g. repeated runs of the same model+seed) by appending the run timestamp.
    label_counts: dict[str, int] = {}
    for mo in models_out:
        label_counts[mo["label"]] = label_counts.get(mo["label"], 0) + 1
    for mo in models_out:
        if label_counts[mo["label"]] > 1:
            ts = re.search(r"(\d{8}_\d{6})", mo["run"])
            mo["label"] = (
                f"{mo['label']} [{ts.group(1)}]"
                if ts
                else f"{mo['label']} [{mo['run']}]"
            )

    return {
        "meta": payload_meta,
        "pid_names": PID_NAMES,
        "pid_colors": PID_COLORS,
        "pid_label_meaning": PID_LABEL_MEANING,
        "models": models_out,
        "tasks": tasks_out,
    }


def render_html(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("/*__DATA__*/null", data_json)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MINERvA Event Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg:#0f1115; --panel:#1a1d24; --fg:#e6e6e6; --muted:#9aa0aa; --accent:#4363d8; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding:12px 18px; background:var(--panel); border-bottom:1px solid #2a2e38; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .sub { color:var(--muted); font-size:12px; margin-top:3px; }
  .controls { display:flex; flex-wrap:wrap; gap:14px; padding:12px 18px; background:var(--panel);
              border-bottom:1px solid #2a2e38; align-items:flex-end; }
  .ctrl { display:flex; flex-direction:column; gap:4px; }
  .ctrl label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  select, button { background:#0f1115; color:var(--fg); border:1px solid #333a47; border-radius:6px;
                   padding:7px 9px; font-size:13px; }
  select:focus, button:focus { outline:1px solid var(--accent); }
  button { cursor:pointer; }
  button:hover { border-color:var(--accent); }
  .nav { display:flex; align-items:center; gap:8px; }
  .nav .idx { min-width:70px; text-align:center; font-variant-numeric:tabular-nums; }
  .info { padding:8px 18px; font-size:12.5px; color:var(--muted); background:var(--panel);
          border-bottom:1px solid #2a2e38; }
  .info b { color:var(--fg); }
  .plots { display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:10px; }
  @media (max-width:1000px){ .plots { grid-template-columns:1fr; } }
  .plotbox { background:var(--panel); border:1px solid #2a2e38; border-radius:8px; }
  .plotbox h2 { font-size:13px; margin:0; padding:8px 12px; border-bottom:1px solid #2a2e38; color:var(--muted); font-weight:600; }
  .plot { width:100%; height:560px; }
  .legend { padding:8px 18px 16px; font-size:12px; color:var(--muted); display:flex; flex-wrap:wrap; gap:14px; }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .swatch { width:12px; height:12px; border-radius:3px; display:inline-block; border:1px solid #00000055; }
  .chk { display:inline-flex; align-items:center; gap:6px; font-size:13px; color:var(--text); font-weight:500; cursor:pointer; padding-top:4px; }
  .chk input { accent-color:#ff4dff; }
  .scorebar { display:flex; align-items:center; gap:12px; padding:8px 18px; background:var(--panel); border-bottom:1px solid #2a2e38; }
  .scorebar button { font-size:13px; }
  .muted { color:var(--muted); font-size:12px; }
  .scorepanel { padding:10px 18px 14px; background:var(--panel); border-bottom:1px solid #2a2e38; overflow-x:auto; }
  .scoreblk { margin-bottom:10px; }
  .scoreblk h3 { margin:0 0 6px; font-size:12px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
  table.scoretbl { border-collapse:collapse; font-size:12.5px; }
  .scoretbl th, .scoretbl td { border:1px solid #2a2e38; padding:4px 9px; text-align:center; font-variant-numeric:tabular-nums; }
  .scoretbl th { color:var(--muted); font-weight:600; background:#0f1115; white-space:nowrap; }
  .scoretbl td.mname { text-align:left; white-space:nowrap; color:var(--fg); }
  .scoretbl td.hi { font-weight:700; color:#fff; background:#26303f; }
  .scoretbl th.sig, .scoretbl td.sig { background:#13261a; }
  .scoretbl th.bkg, .scoretbl td.bkg { background:#26161a; }
  .scoretbl th.truth { outline:2px solid #ff4dff; }
</style>
</head>
<body>
<header>
  <h1>MINERvA Event Viewer</h1>
  <div class="sub" id="srcline"></div>
</header>

<div class="controls">
  <div class="ctrl">
    <label for="task">Signal definition / binning</label>
    <select id="task"></select>
  </div>
  <div class="ctrl">
    <label for="bin">Kinematic bin</label>
    <select id="bin"></select>
  </div>
  <div class="ctrl">
    <label for="cls">Class</label>
    <select id="cls"></select>
  </div>
  <div class="ctrl">
    <label>Event</label>
    <div class="nav">
      <button id="prev">&#9664; Prev</button>
      <span class="idx" id="idx">- / -</span>
      <button id="next">Next &#9654;</button>
    </div>
  </div>
  <div class="ctrl">
    <label>Overlay</label>
    <label class="chk"><input type="checkbox" id="showmc"> Show MC truth (&pi;)</label>
  </div>
</div>

<div class="info" id="info"></div>

<div class="scorebar" id="scorebar" style="display:none">
  <button id="togglescores">&#9656; Model scores</button>
  <span class="muted" id="scoreshint"></span>
</div>
<div class="scorepanel" id="scoresPanel" style="display:none"></div>

<div class="plots">
  <div class="plotbox">
    <h2>3D detector display (coordinates + momentum arrows)</h2>
    <div id="plot3d" class="plot"></div>
  </div>
  <div class="plotbox">
    <h2>&eta;&ndash;&phi; plane (marker size &prop; log E)</h2>
    <div id="plot2d" class="plot"></div>
  </div>
</div>

<div class="legend" id="legend"></div>

<script>
const DATA = /*__DATA__*/null;

const PID_NAMES = DATA.pid_names;
const PID_COLORS = DATA.pid_colors;
const BLOB_PIDS = new Set([2,6]);
const PRONG_PIDS = new Set([3,4,5,7]);
const VERTEX_ARROW_PIDS = new Set([0,1]); // muon, photon: arrow from estimated vertex
const MC_COLOR = '#ff4dff'; // MC truth overlay (magenta)
const CLASS_SHORT = {0:'CC 1\u03c0\u00b1', 1:'CC >1\u03c0\u00b1', 2:'CC 1\u03c0\u2070', 3:'CC other', 4:'NC'};
function shortClass(c){ return (CLASS_SHORT[c]!==undefined) ? CLASS_SHORT[c] : ('class '+c); }
function fmtProb(x){ return (x==null||isNaN(x)) ? '\u2014' : (100*x).toFixed(1)+'%'; }

// particle = [pid, eta, phi, pt, E, x, y, z, dedx]
const P_PID=0, P_ETA=1, P_PHI=2, P_PT=3, P_E=4, P_X=5, P_Y=6, P_Z=7, P_DEDX=8;

const taskSel = document.getElementById('task');
const binSel  = document.getElementById('bin');
const clsSel  = document.getElementById('cls');
const idxLbl  = document.getElementById('idx');
const infoDiv = document.getElementById('info');

let state = { task:0, bin:0, cls:'signal', evt:0, showMC:false, showScores:false };

function currentTask(){ return DATA.tasks[state.task]; }
function currentBin(){ return currentTask().bins[state.bin]; }
function currentEvents(){
  const b = currentBin();
  return (b.classes[state.cls]) || [];
}

function fillTasks(){
  taskSel.innerHTML='';
  DATA.tasks.forEach((t,i)=>{
    const o=document.createElement('option'); o.value=i; o.textContent=t.title; taskSel.appendChild(o);
  });
}
function fillBins(){
  binSel.innerHTML='';
  currentTask().bins.forEach((b,i)=>{
    const unit = currentTask().bin_unit || '';
    const o=document.createElement('option'); o.value=i;
    o.textContent = `${b.label}` + (b.lo!=null ? `  [${b.lo.toFixed(2)}, ${b.hi.toFixed(2)}] ${unit}` : '');
    binSel.appendChild(o);
  });
}
function fillClasses(){
  clsSel.innerHTML='';
  const b = currentBin();
  ['signal','background'].forEach(c=>{
    if(b.classes[c]){
      const o=document.createElement('option'); o.value=c;
      o.textContent = c + ` (${b.classes[c].length})`; clsSel.appendChild(o);
    }
  });
  if(!b.classes[state.cls]){
    state.cls = clsSel.options.length? clsSel.options[0].value : 'signal';
  }
  clsSel.value = state.cls;
}

function vecFromAngles(eta, phi){
  const ch = Math.cosh(eta);
  return [Math.cos(phi)/ch, Math.sin(phi)/ch, Math.tanh(eta)];
}

function build3D(events){
  const ev = events[state.evt];
  const parts = ev.particles;
  const traces = [];

  // collect coords to estimate vertex + scale
  const coords = parts.filter(p => p[P_X]||p[P_Y]||p[P_Z]);
  let vx=0,vy=0,vz=0;
  if(coords.length){
    coords.forEach(p=>{vx+=p[P_X];vy+=p[P_Y];vz+=p[P_Z];});
    vx/=coords.length; vy/=coords.length; vz/=coords.length;
  }
  // characteristic length for arrows
  let span=0;
  if(coords.length){
    let xs=coords.map(p=>p[P_X]), ys=coords.map(p=>p[P_Y]), zs=coords.map(p=>p[P_Z]);
    const rng=a=>Math.max(...a)-Math.min(...a);
    span=Math.max(rng(xs),rng(ys),rng(zs));
  }
  // scene scale: fall back to a sensible detector-ish length when there are
  // no positioned hits, so muon/photon arrows are still clearly drawn.
  const scale = (span>10 ? span : 1000);
  const Emax = Math.max(1e-6, ...parts.map(p=>p[P_E]));

  // group point markers by pid
  const byPid = {};
  parts.forEach(p=>{
    if(BLOB_PIDS.has(p[P_PID]) || PRONG_PIDS.has(p[P_PID])){
      (byPid[p[P_PID]] ||= []).push(p);
    }
  });
  for(const pid in byPid){
    const ps=byPid[pid];
    traces.push({
      type:'scatter3d', mode:'markers', name:PID_NAMES[pid],
      x:ps.map(p=>p[P_X]), y:ps.map(p=>p[P_Y]), z:ps.map(p=>p[P_Z]),
      marker:{ size:ps.map(p=>6+10*Math.sqrt(p[P_E]/Emax)), color:PID_COLORS[pid],
               line:{width:0.5,color:'#000'} },
      text:ps.map(p=>`${PID_NAMES[pid]}<br>E=${p[P_E]} MeV, pT=${p[P_PT]} MeV<br>dE/dx=${p[P_DEDX]==null?'n/a':p[P_DEDX]+' MeV/cm'}`),
      hoverinfo:'text', legendgroup:'pid'+pid,
    });
  }

  // arrows: shafts (lines) + cone heads, grouped by pid
  const arrowGroups = {};
  function addArrow(pid, start, d, len){
    const g = (arrowGroups[pid] ||= {sx:[],sy:[],sz:[],cx:[],cy:[],cz:[],u:[],v:[],w:[]});
    const end=[start[0]+d[0]*len, start[1]+d[1]*len, start[2]+d[2]*len];
    g.sx.push(start[0],end[0],null); g.sy.push(start[1],end[1],null); g.sz.push(start[2],end[2],null);
    g.cx.push(end[0]); g.cy.push(end[1]); g.cz.push(end[2]);
    g.u.push(d[0]*len); g.v.push(d[1]*len); g.w.push(d[2]*len);
  }
  let hasVertexArrow=false;
  parts.forEach(p=>{
    const pid=p[P_PID];
    const d=vecFromAngles(p[P_ETA],p[P_PHI]);
    // mild energy scaling, bounded so arrows are never invisibly short
    const ef=0.6+0.6*Math.sqrt(p[P_E]/Emax);
    if(PRONG_PIDS.has(pid)){
      addArrow(pid, [p[P_X],p[P_Y],p[P_Z]], d, 0.35*scale*ef);
    } else if(VERTEX_ARROW_PIDS.has(pid)){
      // muon/photon have no position: draw a long arrow from the vertex so
      // it clearly extends beyond the hit cluster.
      addArrow(pid, [vx,vy,vz], d, 0.6*scale*ef);
      hasVertexArrow=true;
    }
  });
  for(const pid in arrowGroups){
    const g=arrowGroups[pid];
    // muon/photon have no marker trace, so let their arrow carry the legend entry
    const arrowOnly = VERTEX_ARROW_PIDS.has(+pid);
    traces.push({ type:'scatter3d', mode:'lines', showlegend:arrowOnly, name:PID_NAMES[pid],
      legendgroup:'pid'+pid,
      x:g.sx,y:g.sy,z:g.sz, line:{color:PID_COLORS[pid],width:5}, hoverinfo:'skip' });
    traces.push({ type:'cone', showlegend:false, legendgroup:'pid'+pid,
      x:g.cx,y:g.cy,z:g.cz,u:g.u,v:g.v,w:g.w, anchor:'tip',
      sizemode:'absolute', sizeref:0.06*scale, showscale:false,
      colorscale:[[0,PID_COLORS[pid]],[1,PID_COLORS[pid]]], hoverinfo:'skip' });
  }

  // vertex marker (anchor for muon/photon arrows that have no stored position)
  if(hasVertexArrow || coords.length){
    traces.push({ type:'scatter3d', mode:'markers', name:'est. vertex',
      x:[vx],y:[vy],z:[vz], marker:{size:5,color:'#ffffff',symbol:'x'},
      hoverinfo:'text', text:['estimated vertex (μ/γ arrow origin)'] });
  }

  // MC-truth overlay (pion direction from truth four-vector)
  if(state.showMC && ev.mc && ev.mc.length){
    ev.mc.forEach(m=>{
      const len=0.7*scale, d=m.dir;
      const end=[vx+d[0]*len, vy+d[1]*len, vz+d[2]*len];
      traces.push({ type:'scatter3d', mode:'lines', name:m.label,
        x:[vx,end[0]], y:[vy,end[1]], z:[vz,end[2]],
        line:{color:MC_COLOR,width:6,dash:'dash'},
        hoverinfo:'text', text:[`${m.label}<br>E=${m.E} MeV`] });
      traces.push({ type:'cone', showlegend:false,
        x:[end[0]],y:[end[1]],z:[end[2]], u:[d[0]*len],v:[d[1]*len],w:[d[2]*len],
        anchor:'tip', sizemode:'absolute', sizeref:0.07*scale, showscale:false,
        colorscale:[[0,MC_COLOR],[1,MC_COLOR]], hoverinfo:'skip' });
    });
  }

  const layout={
    margin:{l:0,r:0,t:0,b:0}, paper_bgcolor:'#1a1d24',
    scene:{ xaxis:{title:'x [mm]',color:'#9aa0aa',gridcolor:'#2a2e38'},
            yaxis:{title:'y [mm]',color:'#9aa0aa',gridcolor:'#2a2e38'},
            zaxis:{title:'z [mm]',color:'#9aa0aa',gridcolor:'#2a2e38'},
            bgcolor:'#1a1d24' },
    legend:{font:{color:'#e6e6e6'}, bgcolor:'rgba(0,0,0,0)'},
  };
  Plotly.react('plot3d', traces, layout, {responsive:true, displaylogo:false});
}

function build2D(events){
  const ev = events[state.evt];
  const parts = ev.particles;
  const byPid={};
  parts.forEach(p=>{ (byPid[p[P_PID]] ||= []).push(p); });
  const Emax=Math.max(1e-6,...parts.map(p=>p[P_E]));
  const traces=[];
  for(const pid in byPid){
    const ps=byPid[pid];
    traces.push({
      type:'scatter', mode:'markers', name:PID_NAMES[pid],
      x:ps.map(p=>p[P_ETA]), y:ps.map(p=>p[P_PHI]),
      marker:{ size:ps.map(p=>8+22*Math.sqrt(p[P_E]/Emax)), color:PID_COLORS[pid],
               line:{width:0.5,color:'#000'}, opacity:0.8 },
      text:ps.map(p=>`${PID_NAMES[pid]}<br>E=${p[P_E]} MeV, pT=${p[P_PT]} MeV<br>&eta;=${p[P_ETA]}, &phi;=${p[P_PHI]}<br>dE/dx=${p[P_DEDX]==null?'n/a':p[P_DEDX]+' MeV/cm'}`),
      hoverinfo:'text',
    });
  }
  if(state.showMC && ev.mc && ev.mc.length){
    traces.push({
      type:'scatter', mode:'markers', name:'π (MC truth)',
      x:ev.mc.map(m=>m.eta), y:ev.mc.map(m=>m.phi),
      marker:{ size:18, color:MC_COLOR, symbol:'star', line:{width:1.5,color:'#fff'} },
      text:ev.mc.map(m=>`${m.label}<br>E=${m.E} MeV<br>&eta;=${m.eta}, &phi;=${m.phi}`),
      hoverinfo:'text',
    });
  }
  const layout={
    margin:{l:55,r:10,t:10,b:45}, paper_bgcolor:'#1a1d24', plot_bgcolor:'#1a1d24',
    xaxis:{title:'&eta;', color:'#9aa0aa', gridcolor:'#2a2e38', zerolinecolor:'#3a3f4b'},
    yaxis:{title:'&phi; [rad]', range:[-Math.PI,Math.PI], color:'#9aa0aa',
           gridcolor:'#2a2e38', zerolinecolor:'#3a3f4b'},
    legend:{font:{color:'#e6e6e6'}, bgcolor:'rgba(0,0,0,0)'},
  };
  Plotly.react('plot2d', traces, layout, {responsive:true, displaylogo:false});
}

function updateInfo(events){
  const t=currentTask(), b=currentBin(), ev=events[state.evt];
  const meaning = DATA.pid_label_meaning[ev.pid_class] || ('class '+ev.pid_class);
  const unit=t.bin_unit||'';
  let bv = (ev.bin_value!=null)? ev.bin_value.toFixed(3)+' '+unit : 'n/a';
  let gi = (ev.global_index!=null)? ev.global_index : 'n/a';
  const mcAvail = (ev.mc && ev.mc.length) ? 'yes' : 'none';
  infoDiv.innerHTML =
    `<b>${t.title}</b> &nbsp;|&nbsp; bin <b>${b.label}</b> &nbsp;|&nbsp; ` +
    `<b>${state.cls}</b> &nbsp;|&nbsp; event truth: <b>${meaning}</b> (pid ${ev.pid_class}) ` +
    `&nbsp;|&nbsp; bin value: <b>${bv}</b> &nbsp;|&nbsp; particles: <b>${ev.particles.length}</b> ` +
    `&nbsp;|&nbsp; MC truth &pi;: <b>${mcAvail}</b> &nbsp;|&nbsp; playlist global index: <b>${gi}</b>`;
}

function buildScores(events){
  const panel=document.getElementById('scoresPanel');
  const models=DATA.models||[];
  if(!models.length){ panel.innerHTML='<div class="muted">No scores.json found in the dataset directory.</div>'; return; }
  const ev=events[state.evt];
  const scores=ev.scores||{};
  const task=currentTask();
  const signalSet=new Set(task.signal_pid_classes||[]);
  const truthPid=ev.pid_class;
  let html='';

  // ---- classifiers: per-class probabilities + merged signal/background ----
  const clf=models.filter(m=>m.mode==='classifier');
  if(clf.length){
    // union of pid classes across classifier models (preserve numeric order)
    const pidCols=[];
    clf.forEach(m=>(m.class_idx||[]).forEach(c=>{ if(!pidCols.includes(c)) pidCols.push(c); }));
    pidCols.sort((a,b)=>a-b);
    html+='<div class="scoreblk"><h3>Classification &mdash; probability per class (signal = '
        + (task.signal_pid_classes||[]).map(shortClass).join(', ') + ')</h3>';
    html+='<table class="scoretbl"><thead><tr><th>Model</th>';
    pidCols.forEach(c=>{ const t=(c===truthPid)?' class="truth"':''; html+=`<th${t}>${shortClass(c)}</th>`; });
    html+='<th class="sig">Signal &Sigma;</th><th class="bkg">Bkg &Sigma;</th></tr></thead><tbody>';
    clf.forEach(m=>{
      const pred=scores[m.run];
      html+=`<tr><td class="mname">${m.label}</td>`;
      if(!pred){ pidCols.forEach(()=>html+='<td>&mdash;</td>'); html+='<td class="sig">&mdash;</td><td class="bkg">&mdash;</td></tr>'; return; }
      const ci=m.class_idx;
      let sig=0,bkg=0;
      // argmax for highlighting
      let maxk=-1,maxv=-Infinity; pred.forEach((v,k)=>{ if(v>maxv){maxv=v;maxk=k;} });
      if(ci){
        const p={}; ci.forEach((c,k)=>{ p[c]=pred[k]; });
        const argPid=ci[maxk];
        pidCols.forEach(c=>{ const hi=(c===argPid)?' class="hi"':''; html+=`<td${hi}>${(c in p)?fmtProb(p[c]):'&mdash;'}</td>`; });
        ci.forEach((c,k)=>{ if(signalSet.has(c)) sig+=pred[k]; else bkg+=pred[k]; });
      } else {
        // binary head: prediction = [background, signal]
        pidCols.forEach(()=>html+='<td>&mdash;</td>');
        bkg=pred[0]; sig=pred[1];
      }
      html+=`<td class="sig">${fmtProb(sig)}</td><td class="bkg">${fmtProb(bkg)}</td></tr>`;
    });
    html+='</tbody></table></div>';
  }

  // ---- regression ----
  const reg=models.filter(m=>m.mode==='regression');
  if(reg.length){
    const unit=task.bin_unit||'';
    const truth=ev.bin_value;
    // error = prediction - MC-truth target; sort models by |error| (best first).
    const rows=reg.map(m=>{
      const pred=scores[m.run];
      const p=(pred==null)?null:+pred;
      const err=(p==null||truth==null)?null:(p-truth);
      return {label:m.label, p, err};
    });
    rows.sort((a,b)=>{
      const aa=(a.err==null)?Infinity:Math.abs(a.err);
      const bb=(b.err==null)?Infinity:Math.abs(b.err);
      return aa-bb;
    });
    html+='<div class="scoreblk"><h3>Regression &mdash; predicted value (sorted by |error|)</h3>';
    html+='<table class="scoretbl"><thead><tr><th>Model</th><th>Prediction</th><th>Error</th></tr></thead><tbody>';
    let bestMarked=false;
    rows.forEach(r=>{
      const predStr=(r.p==null)?'&mdash;':(r.p.toFixed(3)+' '+unit);
      let errStr='&mdash;', errCls='';
      if(r.err!=null){
        errStr=(r.err>=0?'+':'')+r.err.toFixed(3)+' '+unit;
        if(!bestMarked){ errCls=' class="hi"'; bestMarked=true; }  // smallest |error|
      }
      html+=`<tr><td class="mname">${r.label}</td><td>${predStr}</td><td${errCls}>${errStr}</td></tr>`;
    });
    html+='</tbody></table>';
    if(truth!=null) html+=`<div class="muted">MC-truth target: <b>${truth.toFixed(3)} ${unit}</b></div>`;
    html+='</div>';
  }

  panel.innerHTML = html || '<div class="muted">No model scores for this event.</div>';
}

function render(){
  const events=currentEvents();
  if(!events.length){ idxLbl.textContent='0 / 0'; return; }
  if(state.evt>=events.length) state.evt=0;
  idxLbl.textContent=`${state.evt+1} / ${events.length}`;
  updateInfo(events);
  build3D(events);
  build2D(events);
  if(state.showScores) buildScores(events);
}

function buildLegend(){
  const el=document.getElementById('legend');
  el.innerHTML='Particle types: ';
  Object.keys(PID_NAMES).forEach(pid=>{
    const s=document.createElement('span');
    s.innerHTML=`<span class="swatch" style="background:${PID_COLORS[pid]}"></span>${PID_NAMES[pid]}`;
    el.appendChild(s);
  });
}

taskSel.addEventListener('change', e=>{ state.task=+e.target.value; state.bin=0; state.evt=0;
  fillBins(); fillClasses(); render(); });
binSel.addEventListener('change', e=>{ state.bin=+e.target.value; state.evt=0;
  fillClasses(); render(); });
clsSel.addEventListener('change', e=>{ state.cls=e.target.value; state.evt=0; render(); });
document.getElementById('showmc').addEventListener('change', e=>{ state.showMC=e.target.checked; render(); });
document.getElementById('prev').addEventListener('click', ()=>{ const n=currentEvents().length;
  if(n){ state.evt=(state.evt-1+n)%n; render(); }});
document.getElementById('next').addEventListener('click', ()=>{ const n=currentEvents().length;
  if(n){ state.evt=(state.evt+1)%n; render(); }});
document.getElementById('togglescores').addEventListener('click', ()=>{
  state.showScores=!state.showScores;
  document.getElementById('scoresPanel').style.display = state.showScores?'block':'none';
  document.getElementById('togglescores').innerHTML =
    (state.showScores?'\u25be':'\u25b8') + ' Model scores';
  if(state.showScores) buildScores(currentEvents());
});
document.addEventListener('keydown', e=>{
  if(e.key==='ArrowLeft') document.getElementById('prev').click();
  if(e.key==='ArrowRight') document.getElementById('next').click();
});

(function init(){
  const m=DATA.meta||{};
  document.getElementById('srcline').textContent =
    `source: ${m.source||'?'}  |  playlist ${m.playlist||'?'} / ${m.split||'?'} split` +
    (m.n_events_per_bin_class? `  |  up to ${m.n_events_per_bin_class} events per bin/class` : '');
  buildLegend();
  if((DATA.models||[]).length){
    document.getElementById('scorebar').style.display='flex';
    document.getElementById('scoreshint').textContent =
      `${DATA.models.length} model(s) from scores.json`;
  }
  if(!DATA.tasks.length){ infoDiv.textContent='No datasets found.'; return; }
  fillTasks(); fillBins(); fillClasses(); render();
})();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Demo datasets dir (default: {DEFAULT_INPUT_DIR})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HTML (default: {DEFAULT_OUTPUT})",
    )
    args = ap.parse_args(argv)

    if not args.input_dir.exists():
        sys.exit(f"Input dir not found: {args.input_dir}")
    out = args.output

    print(f"Reading demo datasets from {args.input_dir}")
    payload = build_payload(args.input_dir)
    n_tasks = len(payload["tasks"])
    n_ev = sum(
        len(evs)
        for t in payload["tasks"]
        for b in t["bins"]
        for evs in b["classes"].values()
    )
    if n_tasks == 0:
        sys.exit("No datasets found; nothing to write.")
    html = render_html(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    size_kb = out.stat().st_size / 1024
    print(f"\nWrote {out} ({size_kb:.1f} KB): {n_tasks} tasks, {n_ev} events embedded.")


if __name__ == "__main__":
    main()
