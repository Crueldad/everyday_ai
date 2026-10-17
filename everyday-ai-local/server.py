"""Everyday AI: household profiles for home robots.

A homeowner shows a robot how their home works (demos plus a few words), the app
turns that into rules, and a robot can load the result as a "household profile".

FastAPI + SQLite backend. A small embedding model powers "ask your home" search,
and a small local language model (Qwen) turns notes into rules and proposes fixes
when you correct the robot. Both load in the background; if a model is missing,
everything falls back to plain keyword and template logic, so the app always works.

Run:  python server.py          (this computer only)
      python server.py --lan    (reachable from a phone on the same wifi)

Environment variables:
  EVERYDAY_NO_MODELS=1   skip all model loading
  EVERYDAY_LLM=0         load the embedding model only
  EMBED_MODEL            default sentence-transformers/all-MiniLM-L6-v2
  LLM_MODEL              default Qwen/Qwen2.5-1.5B-Instruct
  PORT                   default 8000
"""
import io
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).parent
DB_PATH = BASE / "everyday_homes.db"
UPLOADS = BASE / "uploads"
STATIC = BASE / "static"
UPLOADS.mkdir(exist_ok=True)

TASKS = ["Laundry", "Dishes", "Doors", "Cleaning", "Tidying"]
LOCATIONS = ["Kitchen", "Bedroom", "Bathroom", "Living room", "Hallway"]
KINDS = ["how", "where", "avoid", "off_limits"]
MINE = "mine"
MAX_UPLOAD = 150 * 1024 * 1024

_lock = threading.Lock()


def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


SCHEMA = """
CREATE TABLE IF NOT EXISTS households (
  id TEXT PRIMARY KEY, name TEXT, description TEXT, sample INTEGER DEFAULT 0, created REAL
);
CREATE TABLE IF NOT EXISTS demos (
  id TEXT PRIMARY KEY, household_id TEXT, title TEXT, note TEXT DEFAULT '', task TEXT, location TEXT,
  kind TEXT, dur REAL DEFAULT 0, file TEXT, sample INTEGER DEFAULT 0, created REAL
);
CREATE TABLE IF NOT EXISTS rules (
  id TEXT PRIMARY KEY, household_id TEXT, demo_id TEXT, kind TEXT, task TEXT, location TEXT,
  scope TEXT, text TEXT, source TEXT DEFAULT 'manual', created REAL, emb BLOB
);
CREATE TABLE IF NOT EXISTS corrections (
  id TEXT PRIMARY KEY, household_id TEXT, rule_id TEXT, attempt TEXT, feedback TEXT,
  old_text TEXT, new_text TEXT, created REAL
);
"""

# ------------------------------------------------------------- sample homes
SEED_HOUSEHOLDS = [
    ("okafor", "The Okafor flat", "A family of four in a small flat. Quiet evenings, a sleeping baby, and a few things nobody touches.", 1),
    ("sam", "Sam's studio", "One person, one cat, one tidy studio. Everything has a spot and nothing is stacked high.", 1),
    (MINE, "My home", "Your own profile. Teach it something and it shows up here.", 0),
]
# id, household, title, note, task, location, kind, dur
SEED_DEMOS = [
    ("d_ok1", "okafor", "Folding the bath towels", "I fold bath towels in thirds with the folded edge facing out, then stack them on the second shelf of the bathroom cabinet, biggest at the bottom.", "Laundry", "Bathroom", "video", 22),
    ("d_ok2", "okafor", "Loading the dishwasher", "Bowls go on the top rack and plates on the bottom, facing the center. The wooden spoons and the cast-iron pan never go in the dishwasher.", "Dishes", "Kitchen", "video", 38),
    ("d_ok3", "okafor", "Hanging a work shirt", "Button-up shirts get hung with the top button done, facing left. Wool sweaters are never hung. Fold them flat in the drawer.", "Laundry", "Bedroom", "video", 19),
    ("d_ok4", "okafor", "Closing the front door softly", "The baby sleeps near the front door, so close it gently and never let it slam.", "Doors", "Hallway", "video", 9),
    ("d_sm1", "sam", "Rolling towels", "I roll towels tightly instead of folding them and stand them upright in the woven basket under the sink.", "Laundry", "Bathroom", "video", 17),
    ("d_sm2", "sam", "Shelving books by height", "Books go back on the shelf spine out, sorted by height with the tallest on the left. The remote lives in the basket on the coffee table.", "Tidying", "Living room", "video", 28),
    ("d_sm3", "sam", "Wiping the counter", "Wipe with the gray cloth from left to right, then rinse the cloth. No spray cleaners anywhere near the desk or the laptop.", "Cleaning", "Kitchen", "video", 14),
]
# household, demo, kind, task, location, scope, text
SEED_RULES = [
    ("okafor", "d_ok1", "how", "Laundry", "Bathroom", None, "Fold bath towels in thirds with the folded edge facing out."),
    ("okafor", "d_ok1", "where", "Laundry", "Bathroom", None, "Stack folded towels on the second shelf of the bathroom cabinet, largest at the bottom."),
    ("okafor", "d_ok2", "how", "Dishes", "Kitchen", None, "Load bowls on the top rack and plates on the bottom rack, facing the center."),
    ("okafor", "d_ok2", "avoid", "Dishes", "Kitchen", None, "Never put the wooden spoons or the cast-iron pan in the dishwasher."),
    ("okafor", "d_ok3", "how", "Laundry", "Bedroom", None, "Hang button-up shirts with the top button done, facing left."),
    ("okafor", "d_ok3", "avoid", "Laundry", "Bedroom", None, "Never hang wool sweaters. Fold them flat and put them in the drawer."),
    ("okafor", "d_ok4", "how", "Doors", "Hallway", None, "Close the front door gently and never let it slam."),
    ("okafor", None, "where", "Dishes", "Kitchen", None, "Mugs go on the open shelf above the kettle, handles pointing right."),
    ("okafor", None, "how", "Cleaning", "Kitchen", None, "Wipe the counter with the blue cloth, then dry it with the white one."),
    ("okafor", None, "avoid", "Cleaning", "Living room", None, "Never vacuum or run loud appliances after 8 pm."),
    ("okafor", None, "off_limits", None, "Living room", "item", "Do not touch or move the glass cabinet or the piano."),
    ("okafor", None, "off_limits", None, "Hallway", "item", "Leave the stroller and the baby gate where they are."),
    ("sam", "d_sm1", "how", "Laundry", "Bathroom", None, "Roll towels tightly instead of folding them."),
    ("sam", "d_sm1", "where", "Laundry", "Bathroom", None, "Stand rolled towels upright in the woven basket under the sink."),
    ("sam", "d_sm2", "how", "Tidying", "Living room", None, "Shelve books spine out, sorted by height, tallest on the left."),
    ("sam", "d_sm2", "where", "Tidying", "Living room", None, "Return the TV remote to the basket on the coffee table."),
    ("sam", "d_sm3", "how", "Cleaning", "Kitchen", None, "Wipe the counter with the gray cloth from left to right, then rinse the cloth."),
    ("sam", "d_sm3", "avoid", "Cleaning", "Living room", None, "Never use spray cleaners near the desk or the laptop."),
    ("sam", None, "how", "Laundry", "Bedroom", None, "Fold shirts into thirds and stack them by color, light colors on top."),
    ("sam", None, "how", "Dishes", "Kitchen", None, "Hand-wash pans and knives. Put everything else in the dishwasher on the eco cycle."),
    ("sam", None, "where", "Dishes", "Kitchen", None, "Plates and bowls go on the open shelf, one stack each, plates on the left."),
    ("sam", None, "avoid", "Dishes", "Kitchen", None, "Never stack more than four plates. The shelf is narrow."),
    ("sam", None, "how", "Doors", "Hallway", None, "Leave interior doors half open so the cat can pass."),
    ("sam", None, "off_limits", None, "Living room", "item", "Never move papers, cables or the laptop on the desk."),
]


def init_db():
    with _lock, db() as con:
        con.executescript(SCHEMA)
        if con.execute("SELECT COUNT(*) FROM households").fetchone()[0] == 0:
            now = time.time()
            for h in SEED_HOUSEHOLDS:
                con.execute("INSERT INTO households (id,name,description,sample,created) VALUES (?,?,?,?,?)", (*h, now))
            for i, d in enumerate(SEED_DEMOS):
                con.execute(
                    "INSERT INTO demos (id,household_id,title,note,task,location,kind,dur,sample,created) VALUES (?,?,?,?,?,?,?,?,1,?)",
                    (*d, now + i),
                )
            for i, r in enumerate(SEED_RULES):
                con.execute(
                    "INSERT INTO rules (id,household_id,demo_id,kind,task,location,scope,text,source,created)"
                    " VALUES (?,?,?,?,?,?,?,?,'sample',?)",
                    (f"r_{r[0]}_{i}", *r, now + i),
                )


def rule_json(r):
    return {"id": r["id"], "household_id": r["household_id"], "demo_id": r["demo_id"], "kind": r["kind"],
            "task": r["task"], "location": r["location"], "scope": r["scope"], "text": r["text"],
            "source": r["source"], "created": r["created"]}


def demo_json(r):
    return {"id": r["id"], "household_id": r["household_id"], "title": r["title"], "note": r["note"] or "",
            "task": r["task"], "location": r["location"], "kind": r["kind"], "dur": r["dur"] or 0,
            "url": ("/files/" + r["file"]) if r["file"] else None, "sample": bool(r["sample"]), "created": r["created"]}


def get_state():
    with db() as con:
        hs = con.execute("SELECT * FROM households ORDER BY sample DESC, created ASC").fetchall()
        out = []
        for h in hs:
            rules = con.execute("SELECT * FROM rules WHERE household_id=? ORDER BY created ASC", (h["id"],)).fetchall()
            demos = con.execute("SELECT * FROM demos WHERE household_id=? ORDER BY created ASC", (h["id"],)).fetchall()
            ncorr = con.execute("SELECT COUNT(*) FROM corrections WHERE household_id=?", (h["id"],)).fetchone()[0]
            out.append({"id": h["id"], "name": h["name"], "description": h["description"], "sample": bool(h["sample"]),
                        "rules": [rule_json(r) for r in rules], "demos": [demo_json(d) for d in demos], "corrections": ncorr})
    return out


def get_house(hid):
    for h in get_state():
        if h["id"] == hid:
            return h
    raise HTTPException(404, "No such home.")


def require_mine(hid):
    if hid != MINE:
        raise HTTPException(403, "The sample homes are read-only. Switch to My home to make changes.")


# ------------------------------------------------------------ text helpers
STOP = set("a an the in on of at to for with and but or my some video videos clip clips show find me i want from by is are it that this away up out into how do does like should".split())


def stem(w):
    w = w.lower()
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def tokens(text):
    return [stem(w) for w in re.findall(r"[a-zA-Z]+", text or "") if w.lower() not in STOP and len(w) > 1]


TASK_WORDS = {
    "Laundry": "laundry towel sheet shirt fold clothes hanger hang linen garment sock sweater",
    "Dishes": "dish dishes dishwasher plate bowl mug cup glass cutlery fork spoon rack cupboard pan",
    "Doors": "door handle knob latch lever slam",
    "Cleaning": "clean wipe wiping cloth sponge mop scrub dust spill sweep vacuum",
    "Tidying": "tidy organize organise shelf book toy bin declutter clutter remote",
}
LOC_WORDS = {
    "Kitchen": "kitchen sink counter stove fridge cabinet cupboard",
    "Bedroom": "bedroom bed nightstand bedside wardrobe closet drawer",
    "Bathroom": "bathroom bath shower toilet mirror vanity",
    "Living room": "living lounge sofa couch tv piano desk",
    "Hallway": "hallway hall entry entryway corridor stroller",
}
TASK_SETS = {k: {stem(w) for w in v.split()} for k, v in TASK_WORDS.items()}
LOC_SETS = {k: {stem(w) for w in v.split()} for k, v in LOC_WORDS.items()}


def best_of(sets, toks):
    best, score = None, 0
    for name, words in sets.items():
        n = sum(1 for t in toks if t in words)
        if n > score:
            best, score = name, n
    return best


def keyword_parse(q):
    toks = tokens(q)
    return {"task": best_of(TASK_SETS, toks), "location": best_of(LOC_SETS, toks), "by": "keywords"}


def sent(t):
    t = (t or "").strip()
    return t if t[-1:] in ".!?" else t + "."


# ------------------------------------------------------------------ models
STATE = {"embed": "off", "llm": "off", "device": "cpu", "embed_name": None, "llm_name": None,
         "embed_error": None, "llm_error": None}
EMBEDDER = None
LLM = None
LLM_LOCK = threading.Lock()


def embed(texts):
    v = EMBEDDER.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    return v.astype(np.float32)


def rule_text(r):
    return f'{r["text"]} ({r["kind"].replace("_", " ")}; {r["task"] or "any task"}; {r["location"] or "any room"})'


def backfill():
    with _lock, db() as con:
        rows = con.execute("SELECT * FROM rules WHERE emb IS NULL").fetchall()
        if not rows:
            return
        vecs = embed([rule_text(r) for r in rows])
        for r, v in zip(rows, vecs):
            con.execute("UPDATE rules SET emb=? WHERE id=?", (v.tobytes(), r["id"]))


def embed_rule(con, rid):
    if STATE["embed"] == "ready" and EMBEDDER is not None:
        r = con.execute("SELECT * FROM rules WHERE id=?", (rid,)).fetchone()
        if r:
            con.execute("UPDATE rules SET emb=? WHERE id=?", (embed([rule_text(r)])[0].tobytes(), rid))


def load_models():
    global EMBEDDER, LLM
    if os.environ.get("EVERYDAY_NO_MODELS") == "1":
        return
    dev = "cpu"
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    STATE["device"] = dev
    try:
        STATE["embed"] = "loading"
        from sentence_transformers import SentenceTransformer
        name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        EMBEDDER = SentenceTransformer(name, device=dev)
        STATE["embed_name"] = name
        STATE["embed"] = "ready"
        backfill()
        print(f"[models] embeddings ready: {name} on {dev}")
    except Exception as e:
        STATE["embed"] = "off"
        STATE["embed_error"] = f"{type(e).__name__}: {e}"[:300]
        print("[models] embeddings unavailable:", STATE["embed_error"])
    if os.environ.get("EVERYDAY_LLM", "1") == "0":
        return
    try:
        STATE["llm"] = "loading"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if dev == "cuda" else torch.float32
        ).to(dev)
        model.eval()
        LLM = (tok, model)
        STATE["llm_name"] = name
        STATE["llm"] = "ready"
        print(f"[models] language model ready: {name} on {dev}")
    except Exception as e:
        STATE["llm"] = "off"
        STATE["llm_error"] = f"{type(e).__name__}: {e}"[:300]
        print("[models] language model unavailable:", STATE["llm_error"])


def llm_json(system, user, max_new_tokens=220):
    import torch
    tok, model = LLM
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with LLM_LOCK, torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in model output")
    return json.loads(m.group(0))


PARSE_SYSTEM = (
    "You turn a short request about a household chore into filters. Reply with JSON only, like "
    '{"task": "Laundry", "location": "Kitchen"}. '
    f"task must be one of {TASKS} or null. location must be one of {LOCATIONS} or null. "
    "Use null when the request does not say."
)
EXTRACT_SYSTEM = (
    "You turn a homeowner's note about how they like a household chore done into short rules for a home robot. "
    'Reply with JSON only: {"rules": [{"kind": "how", "text": "Fold bath towels in thirds with the folded edge facing out."}]}. '
    "kind must be one of: how (the way to do the task), where (where things go or live), "
    "avoid (things the robot must never do), off_limits (rooms or items the robot must not touch or enter). "
    "Write each text as one short imperative sentence in the owner's own words. Use at most 5 rules. "
    "Do not add details that are not in the note."
)
CORRECT_SYSTEM = (
    "A homeowner corrected a home robot. Rewrite their rule so that it includes the correction. "
    'Reply with JSON only: {"text": "one short imperative sentence"}. '
    "Keep details from the old rule that are still true."
)


def parse_query(q):
    out = keyword_parse(q)
    if STATE["llm"] == "ready" and (not out["task"] or not out["location"]):
        try:
            r = llm_json(PARSE_SYSTEM, q, 40)
            for key, allowed in (("task", TASKS), ("location", LOCATIONS)):
                if not out[key] and r.get(key) in allowed:
                    out[key] = r[key]
                    out["by"] = "language model"
        except Exception as e:
            print("[llm] parse failed:", e)
    return out


# -------------------------------------------------------- rule extraction
def heuristic_rules(note, task, location):
    parts = re.split(r"(?<=[.!?;])\s+|\n+", (note or "").strip())
    rules = []
    for p in parts:
        s = p.strip(" .;!?\t")
        if len(s) < 4:
            continue
        low = s.lower()
        if re.search(r"off[- ]limits|stay out|keep out|do not enter|don't enter|never enter", low):
            kind = "off_limits"
        elif re.search(r"\bnever\b|\bdon't\b|\bdo not\b|\bavoid\b|\bnot\b", low):
            kind = "avoid"
        elif re.search(r"\b(goes|go on|go in|lives|live in|stored?|return|put (them|it|these|those)?\s*(in|on|back)|place[sd]? .* (in|on)|stack .* (in|on))\b", low):
            kind = "where"
        else:
            kind = "how"
        rules.append({"kind": kind, "text": sent(s[0].upper() + s[1:])})
    return rules[:6]


def clean_rules(raw, task, location):
    out = []
    for r in raw[:6]:
        if not isinstance(r, dict):
            continue
        kind, text = r.get("kind"), str(r.get("text", "")).strip()
        if kind not in KINDS or not (4 <= len(text) <= 200):
            continue
        rt = r.get("task") if r.get("task") in TASKS else task
        rl = r.get("location") if r.get("location") in LOCATIONS else location
        scope = ("item" if kind == "off_limits" else None)
        if kind == "off_limits" and re.search(r"\b(entire|whole|room|stay out|keep out|do not enter|don't enter)\b", text.lower()):
            scope = "room" if re.search(r"\broom\b|\bentire\b|\bwhole\b|\benter\b|stay out|keep out", text.lower()) else "item"
        out.append({"kind": kind, "task": None if kind == "off_limits" else rt, "location": rl, "scope": scope, "text": sent(text)})
    return out


def extract_rules(note, task, location, title=""):
    rules, by = [], "template"
    if STATE["llm"] == "ready" and note.strip():
        try:
            r = llm_json(EXTRACT_SYSTEM, f"Title: {title}\nRoom: {location or 'unknown'}\nChore: {task or 'unknown'}\nNote: {note}")
            rules = clean_rules(r.get("rules", []), task, location)
            if rules:
                by = "language model"
        except Exception as e:
            print("[llm] extract failed:", e)
    if not rules:
        rules = clean_rules(heuristic_rules(note, task, location), task, location)
    return rules, by


# ------------------------------------------------------- plans / instructions
CHORE = {"Laundry": "Fold and put away the laundry", "Dishes": "Load and put away the dishes",
         "Doors": "Open and close the doors", "Cleaning": "Clean the surfaces", "Tidying": "Tidy up"}
BASELINE = {
    "Laundry": ["Pick up the item", "Fold it in half", "Place it on the nearest flat surface", "Move on to the next item"],
    "Dishes": ["Pick up the dish", "Put it in the first free spot", "Close the dishwasher or cupboard"],
    "Doors": ["Reach for the handle", "Open or close the door", "Let go of the handle"],
    "Cleaning": ["Pick up a cloth", "Wipe the surface", "Put the cloth down"],
    "Tidying": ["Pick up the item", "Put it on the nearest shelf", "Move on to the next item"],
}


def applies(r, task, location):
    if r["kind"] == "off_limits":
        return location is None or r["location"] in (None, location)
    return r["task"] in (None, task) and (location is None or r["location"] in (None, location))


def compose_instruction(rules, task, location):
    sel = [r for r in rules if applies(r, task, location)]
    order = {k: i for i, k in enumerate(KINDS)}
    sel.sort(key=lambda r: order[r["kind"]])
    head = CHORE.get(task, "Do the chore") + (f" in the {location.lower()}." if location else ".")
    return " ".join([head] + [sent(r["text"]) for r in sel]), sel


def build_plan(rules, task, location):
    instruction, sel = compose_instruction(rules, task, location)
    base = BASELINE.get(task, ["Do the chore"])
    how = [r for r in sel if r["kind"] == "how"]
    where = [r for r in sel if r["kind"] == "where"]
    constraints = [r for r in sel if r["kind"] in ("avoid", "off_limits")]
    blocked = None
    for r in sel:
        if r["kind"] == "off_limits" and r["scope"] == "room" and location and r["location"] == location:
            blocked = f"The {location.lower()} is off-limits: {r['text']}"
    steps = [{"text": base[0], "kind": None, "rule_id": None}]
    for r in how + where:
        steps.append({"text": r["text"], "kind": r["kind"], "rule_id": r["id"]})
    if not (how or where):
        steps = [{"text": t, "kind": None, "rule_id": None} for t in base]
    elif len(base) > 2:
        steps.append({"text": base[-1], "kind": None, "rule_id": None})
    return {
        "task": task, "location": location, "baseline": base, "steps": steps,
        "constraints": [{"rule_id": r["id"], "kind": r["kind"], "scope": r["scope"], "location": r["location"], "text": r["text"]} for r in constraints],
        "blocked": blocked, "instruction": instruction, "matched": len(sel), "known": bool(how or where),
    }


# ------------------------------------------------------------------ search
def search_rules(hid, q):
    house = get_house(hid)
    rules = house["rules"]
    if not q.strip():
        return {"q": q, "mode": "all", "parsed": {}, "results": [dict(r, score=None, why="", relevant=True) for r in rules]}
    parsed = parse_query(q)
    qtok = set(tokens(q))
    use_emb = STATE["embed"] == "ready" and EMBEDDER is not None
    embs = {}
    if use_emb:
        with db() as con:
            for row in con.execute("SELECT id, emb FROM rules WHERE household_id=?", (hid,)):
                if row["emb"] is not None:
                    embs[row["id"]] = np.frombuffer(row["emb"], dtype=np.float32)
        qvec = embed([q])[0]
    scored = []
    for r in rules:
        ctok = set(tokens(" ".join([r["text"], r["task"] or "", r["location"] or ""])))
        hit = sorted(qtok & ctok)
        kw = len(hit) / len(qtok) if qtok else 0
        tm = bool(parsed.get("task") and r["task"] == parsed["task"])
        lm = bool(parsed.get("location") and r["location"] == parsed["location"])
        sim = float(embs[r["id"]] @ qvec) if use_emb and r["id"] in embs else None
        score = (sim or 0) + 0.25 * tm + 0.25 * lm + (0.2 if sim is not None else 0.6) * kw
        why = []
        if tm:
            why.append(f"task {r['task']}")
        if lm:
            why.append(f"room {r['location']}")
        if hit and not (tm or lm):
            why.append("mentions " + ", ".join(hit[:3]))
        scored.append((score, r, " · ".join(why)))
    scored.sort(key=lambda s: s[0], reverse=True)
    best = scored[0][0] if scored else 0
    floor = max(0.3 if use_emb else 0.2, 0.55 * best)
    results = [dict(r, score=round(s, 3), why=w, relevant=bool(s >= floor)) for s, r, w in scored]
    return {"q": q, "mode": "meaning" if use_emb else "keywords", "parsed": parsed, "results": results}


# ------------------------------------------------------------------ export
def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "home"


def task_table(rules):
    """One language instruction per (task, room) pair that has at least one rule."""
    pairs = sorted({(r["task"], r["location"]) for r in rules if r["kind"] != "off_limits" and r["task"]},
                   key=lambda p: (TASKS.index(p[0]), p[1] or ""))
    table = []
    for i, (t, l) in enumerate(pairs):
        text, _ = compose_instruction(rules, t, l)
        table.append({"task_index": i, "task": text, "chore": t, "location": l})
    return table


def tasks_parquet(table):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return None
    t = pa.table({"task": [x["task"] for x in table], "task_index": pa.array([x["task_index"] for x in table], type=pa.int64())})
    buf = io.BytesIO()
    pq.write_table(t, buf)
    return buf.getvalue()


def iso_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_bundle(hid):
    h = get_house(hid)
    rules, demos = h["rules"], h["demos"]
    slug = slugify(h["name"])
    table = task_table(rules)
    index_of = {(t["chore"], t["location"]): t["task_index"] for t in table}
    files = {}  # path -> bytes (text or binary)
    texts = {}  # path -> str for preview

    rooms = {}
    for loc in LOCATIONS:
        rr = [r for r in rules if r["location"] == loc]
        rooms[loc] = {"off_limits": [{"scope": r["scope"], "text": r["text"]} for r in rr if r["kind"] == "off_limits"],
                      "rules": [{"kind": r["kind"], "task": r["task"], "text": r["text"]} for r in rr if r["kind"] != "off_limits"]}
    profile = {
        "schema": "everyday-ai.household-profile/v1", "generated_at": iso_now(),
        "household": {"id": h["id"], "name": h["name"], "description": h["description"], "sample": h["sample"]},
        "rooms": rooms,
        "rules": [{k: r[k] for k in ("id", "kind", "task", "location", "scope", "text", "source", "demo_id")} for r in rules],
        "demos": [{"id": d["id"], "title": d["title"], "task": d["task"], "location": d["location"], "note": d["note"],
                   "has_footage": bool(d["url"])} for d in demos],
        "corrections_logged": h["corrections"],
    }
    constraints = {
        "off_limits": [{"location": r["location"], "scope": r["scope"], "text": r["text"]} for r in rules if r["kind"] == "off_limits"],
        "never": [{"task": r["task"], "location": r["location"], "text": r["text"]} for r in rules if r["kind"] == "avoid"],
    }
    prefs = "\n".join(json.dumps({"household": slug, "id": r["id"], "kind": r["kind"], "task": r["task"], "location": r["location"],
                                  "scope": r["scope"], "text": r["text"]}) for r in rules)
    with_footage = [d for d in demos if d["url"]]
    info = {
        "codebase_version": "v3.0", "robot_type": "unspecified",
        "total_episodes": len(with_footage), "total_frames": 0, "total_tasks": len(table), "fps": 30, "features": {},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "everyday_ai": {"metadata_only": True, "profile": "profile.json", "constraints": "constraints.json",
                        "note": "Task instructions and constraints only. Frame data (observation.state, action) comes from the robot's own recordings."},
    }
    demo_lines = []
    for d in demos:
        fname = Path(d["url"]).name if d["url"] else None
        demo_lines.append(json.dumps({
            "demo_id": d["id"], "title": d["title"], "chore": d["task"], "location": d["location"],
            "task_index": index_of.get((d["task"], d["location"])),
            "instruction": next((t["task"] for t in table if t["task_index"] == index_of.get((d["task"], d["location"]))), None),
            "owner_note": d["note"], "kind": d["kind"], "duration_s": d["dur"],
            "file": f"demos/files/{fname}" if fname else None, "placeholder": not bool(d["url"]),
        }))
    readme = f"""Everyday AI household profile: {h['name']}
Generated {iso_now()}

WHAT THIS IS
The preferences of one home, written so a robot team can load them next to their
own training data. It is the "how this home likes things done" layer. It is not
a motion dataset.

FILES
profile.json          The full profile: rooms, rules, demos. Source of truth.
constraints.json      Off-limits places and "never" rules, machine-readable.
preferences.jsonl     One rule per line.
meta/tasks.parquet    Language instructions (task, task_index) in the layout of
                      LeRobot dataset v3. One instruction per chore and room.
meta/info.json        Dataset info using LeRobot v3 field names. Metadata only.
demos/index.jsonl     Each demo, its instruction and its clip file.
demos/files/          The demo clips that have footage.

WHAT IS NOT INCLUDED
Frame-by-frame observation.state and action columns. Those come from the robot's
own teleoperation or recordings, so this bundle will not load as a complete
LeRobot dataset on its own. Use the instructions and constraints alongside it.
Sample homes use placeholder demos without footage.
"""
    texts["README.txt"] = readme
    texts["profile.json"] = json.dumps(profile, indent=2)
    texts["constraints.json"] = json.dumps(constraints, indent=2)
    texts["preferences.jsonl"] = prefs + ("\n" if prefs else "")
    texts["meta/info.json"] = json.dumps(info, indent=2)
    texts["meta/tasks (preview)"] = "\n".join(json.dumps({"task_index": t["task_index"], "task": t["task"]}) for t in table)
    texts["demos/index.jsonl"] = "\n".join(demo_lines) + ("\n" if demo_lines else "")

    for p in ("README.txt", "profile.json", "constraints.json", "preferences.jsonl", "meta/info.json", "demos/index.jsonl"):
        files[p] = texts[p].encode()
    pq_bytes = tasks_parquet(table)
    if pq_bytes is not None:
        files["meta/tasks.parquet"] = pq_bytes
    else:
        files["meta/tasks.jsonl"] = texts["meta/tasks (preview)"].encode()
    clip_files = {}
    for d in with_footage:
        fp = UPLOADS / Path(d["url"]).name
        if fp.exists():
            clip_files[f"demos/files/{fp.name}"] = fp
    return {"slug": slug, "files": files, "texts": texts, "clips": clip_files, "table": table, "profile": profile,
            "parquet": pq_bytes is not None, "demos": demos, "rules": rules}


def validate_bundle(b):
    checks = []

    def add(ok, text, level=None):
        checks.append({"status": level or ("ok" if ok else "warn"), "text": text})
    rules, table = b["rules"], b["table"]
    add(len(rules) > 0, f"{len(rules)} rule(s) in the profile" if rules else "The profile has no rules yet")
    bad = [r for r in rules if r["kind"] not in KINDS or not (r["text"] or "").strip()]
    add(not bad, "Every rule has a kind and text" if not bad else f"{len(bad)} rule(s) are missing a kind or text")
    chores = {(r["task"], r["location"]) for r in rules if r["kind"] != "off_limits" and r["task"]}
    add(len(table) == len(chores), f"{len(table)} language instruction(s), one per chore and room")
    off = [r for r in rules if r["kind"] == "off_limits"]
    add(True, f"{len(off)} off-limits place(s) written to constraints.json", "ok")
    if b["parquet"]:
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(io.BytesIO(b["files"]["meta/tasks.parquet"]))
            add(t.num_rows == len(table) and set(t.column_names) >= {"task", "task_index"},
                f"meta/tasks.parquet reads back with {t.num_rows} row(s) and columns task, task_index")
        except Exception as e:
            add(False, f"meta/tasks.parquet could not be read: {e}")
    else:
        add(False, "pyarrow is not installed, so tasks were written as tasks.jsonl instead of tasks.parquet")
    missing = [d for d in b["demos"] if d["url"] and f"demos/files/{Path(d['url']).name}" not in b["clips"]]
    add(not missing, "All demo clips with footage are in the bundle" if not missing else f"{len(missing)} clip file(s) are missing on disk")
    ph = [d for d in b["demos"] if not d["url"]]
    if ph:
        add(True, f"{len(ph)} demo(s) are placeholders with no footage (sample content)", "info")
    add(True, "Frame data (observation.state, action) is not included. It comes from the robot's own recordings.", "info")
    return checks


# --------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_app):
    init_db()
    threading.Thread(target=load_models, daemon=True).start()
    yield


app = FastAPI(title="Everyday AI", lifespan=lifespan)


@app.get("/api/status")
def status():
    return dict(STATE, tasks=TASKS, locations=LOCATIONS, kinds=KINDS)


@app.get("/api/state")
def state():
    return {"households": get_state(), "status": status()}


@app.get("/api/search")
def search(household: str, q: str = ""):
    return search_rules(household, q[:200])


class ExtractIn(BaseModel):
    note: str
    task: Optional[str] = None
    location: Optional[str] = None
    title: str = ""


@app.post("/api/extract")
def extract(body: ExtractIn):
    if not body.note.strip():
        raise HTTPException(400, "Write a note first. A sentence or two about how you like it done is enough.")
    task = body.task if body.task in TASKS else None
    loc = body.location if body.location in LOCATIONS else None
    rules, by = extract_rules(body.note[:800], task, loc, body.title[:100])
    if not rules:
        raise HTTPException(400, "Couldn't find a rule in that note. Try a full sentence, like 'Fold towels in thirds.'")
    return {"rules": rules, "by": by}


@app.post("/api/demos")
async def create_demo(
    title: str = Form(...), note: str = Form(""), task: str = Form(...), location: str = Form(...),
    rules: str = Form("[]"), dur: float = Form(0), file: Optional[UploadFile] = File(None),
):
    if task not in TASKS or location not in LOCATIONS:
        raise HTTPException(400, "Pick a chore and a room.")
    try:
        raw_rules = json.loads(rules)
    except Exception:
        raise HTTPException(400, "Rules were not valid.")
    clean = clean_rules(raw_rules, task, location)
    if not clean:
        raise HTTPException(400, "Add at least one rule before saving.")
    did = "d" + uuid.uuid4().hex[:9]
    kind, fname = "note", None
    if file is not None and file.filename:
        ctype = file.content_type or ""
        if not (ctype.startswith("image/") or ctype.startswith("video/")):
            raise HTTPException(400, "Upload a photo or a video.")
        kind = "video" if ctype.startswith("video/") else "photo"
        ext = (Path(file.filename).suffix or (".mp4" if kind == "video" else ".jpg"))[:8]
        fname = did + ext
        size = 0
        with open(UPLOADS / fname, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    f.close()
                    (UPLOADS / fname).unlink(missing_ok=True)
                    raise HTTPException(413, "File is over 150 MB.")
                f.write(chunk)
    now = time.time()
    with _lock, db() as con:
        con.execute(
            "INSERT INTO demos (id,household_id,title,note,task,location,kind,dur,file,sample,created) VALUES (?,?,?,?,?,?,?,?,?,0,?)",
            (did, MINE, title.strip()[:80], note.strip()[:800], task, location, kind, dur, fname, now),
        )
        for i, r in enumerate(clean):
            rid = "r" + uuid.uuid4().hex[:9]
            con.execute(
                "INSERT INTO rules (id,household_id,demo_id,kind,task,location,scope,text,source,created) VALUES (?,?,?,?,?,?,?,?,'demo',?)",
                (rid, MINE, did, r["kind"], r["task"], r["location"], r["scope"], r["text"][:200], now + i * 0.001),
            )
            embed_rule(con, rid)
    return {"id": did, "rules": len(clean)}


@app.delete("/api/demos/{did}")
def delete_demo(did: str):
    with _lock, db() as con:
        d = con.execute("SELECT * FROM demos WHERE id=?", (did,)).fetchone()
        if not d:
            raise HTTPException(404, "No such demo.")
        require_mine(d["household_id"])
        con.execute("DELETE FROM rules WHERE demo_id=?", (did,))
        con.execute("DELETE FROM demos WHERE id=?", (did,))
    if d["file"]:
        (UPLOADS / d["file"]).unlink(missing_ok=True)
    return {"ok": True}


class RuleIn(BaseModel):
    kind: str
    text: str
    task: Optional[str] = None
    location: Optional[str] = None
    scope: Optional[str] = None


@app.post("/api/rules")
def add_rule(body: RuleIn):
    clean = clean_rules([body.model_dump()], body.task if body.task in TASKS else None, body.location if body.location in LOCATIONS else None)
    if not clean:
        raise HTTPException(400, "A rule needs a kind and a short sentence.")
    r = clean[0]
    rid = "r" + uuid.uuid4().hex[:9]
    with _lock, db() as con:
        con.execute(
            "INSERT INTO rules (id,household_id,demo_id,kind,task,location,scope,text,source,created) VALUES (?,?,NULL,?,?,?,?,?,'manual',?)",
            (rid, MINE, r["kind"], r["task"], r["location"], r["scope"], r["text"], time.time()),
        )
        embed_rule(con, rid)
    return {"id": rid}


@app.patch("/api/rules/{rid}")
def edit_rule(rid: str, body: RuleIn):
    with _lock, db() as con:
        r = con.execute("SELECT * FROM rules WHERE id=?", (rid,)).fetchone()
        if not r:
            raise HTTPException(404, "No such rule.")
        require_mine(r["household_id"])
        clean = clean_rules([body.model_dump()], body.task if body.task in TASKS else r["task"], body.location if body.location in LOCATIONS else r["location"])
        if not clean:
            raise HTTPException(400, "A rule needs a kind and a short sentence.")
        c = clean[0]
        con.execute("UPDATE rules SET kind=?, task=?, location=?, scope=?, text=?, emb=NULL WHERE id=?",
                    (c["kind"], c["task"], c["location"], c["scope"], c["text"], rid))
        embed_rule(con, rid)
    return {"ok": True}


@app.delete("/api/rules/{rid}")
def delete_rule(rid: str):
    with _lock, db() as con:
        r = con.execute("SELECT * FROM rules WHERE id=?", (rid,)).fetchone()
        if not r:
            raise HTTPException(404, "No such rule.")
        require_mine(r["household_id"])
        con.execute("DELETE FROM rules WHERE id=?", (rid,))
    return {"ok": True}


class AttemptIn(BaseModel):
    household_id: str
    chore: str
    location: Optional[str] = None


@app.post("/api/attempt")
def attempt(body: AttemptIn):
    house = get_house(body.household_id)
    parsed = parse_query(body.chore)
    task = parsed["task"]
    if not task:
        raise HTTPException(400, "I couldn't tell which chore that is. Try 'put away the towels' or 'load the dishwasher'.")
    loc = body.location if body.location in LOCATIONS else parsed["location"]
    plan = build_plan(house["rules"], task, loc)
    plan["chore"] = body.chore.strip()[:120]
    plan["parsed_by"] = parsed["by"]
    return plan


class ProposeIn(BaseModel):
    household_id: str
    rule_id: Optional[str] = None
    feedback: str


@app.post("/api/correct/propose")
def propose(body: ProposeIn):
    fb = body.feedback.strip()
    if not fb:
        raise HTTPException(400, "Say what the robot should do instead.")
    old = ""
    if body.rule_id:
        with db() as con:
            r = con.execute("SELECT text FROM rules WHERE id=?", (body.rule_id,)).fetchone()
        old = r["text"] if r else ""
    text, by = None, "template"
    if STATE["llm"] == "ready":
        try:
            r = llm_json(CORRECT_SYSTEM, f"Old rule: {old or '(none)'}\nOwner's correction: {fb}", 80)
            t = str(r.get("text", "")).strip()
            if 4 <= len(t) <= 200:
                text, by = sent(t), "language model"
        except Exception as e:
            print("[llm] correction failed:", e)
    if text is None:
        fb2 = re.sub(r"^(no|nope|not like that|that's wrong|wrong)[,.!\s]+", "", fb, flags=re.I).strip()
        fb2 = (fb2[:1].upper() + fb2[1:]) if fb2 else fb
        text = sent(f"{old.rstrip('.')}. Also: {fb2[:1].lower() + fb2[1:]}" if old else fb2)
    return {"text": text, "by": by}


class ApplyIn(BaseModel):
    household_id: str
    rule_id: Optional[str] = None
    text: str
    feedback: str = ""
    attempt: str = ""
    task: Optional[str] = None
    location: Optional[str] = None
    kind: str = "how"


@app.post("/api/correct/apply")
def apply_correction(body: ApplyIn):
    require_mine(body.household_id)
    text = sent(body.text.strip())
    if len(text) < 4:
        raise HTTPException(400, "The rule is empty.")
    with _lock, db() as con:
        old = ""
        if body.rule_id:
            r = con.execute("SELECT * FROM rules WHERE id=?", (body.rule_id,)).fetchone()
            if not r or r["household_id"] != MINE:
                raise HTTPException(404, "No such rule.")
            old = r["text"]
            con.execute("UPDATE rules SET text=?, source='correction', emb=NULL WHERE id=?", (text[:200], body.rule_id))
            rid = body.rule_id
        else:
            kind = body.kind if body.kind in KINDS else "how"
            rid = "r" + uuid.uuid4().hex[:9]
            con.execute(
                "INSERT INTO rules (id,household_id,demo_id,kind,task,location,scope,text,source,created) VALUES (?,?,NULL,?,?,?,?,?,'correction',?)",
                (rid, MINE, kind, body.task if body.task in TASKS else None, body.location if body.location in LOCATIONS else None,
                 "item" if kind == "off_limits" else None, text[:200], time.time()),
            )
        con.execute("INSERT INTO corrections (id,household_id,rule_id,attempt,feedback,old_text,new_text,created) VALUES (?,?,?,?,?,?,?,?)",
                    ("c" + uuid.uuid4().hex[:9], MINE, rid, body.attempt[:300], body.feedback[:300], old, text, time.time()))
        embed_rule(con, rid)
    return {"id": rid}


@app.get("/api/compare")
def compare(task: str, location: Optional[str] = None):
    if task not in TASKS:
        raise HTTPException(400, "Unknown chore.")
    loc = location if location in LOCATIONS else None
    out = []
    for h in get_state():
        instruction, sel = compose_instruction(h["rules"], task, loc)
        out.append({"id": h["id"], "name": h["name"], "sample": h["sample"], "instruction": instruction,
                    "rules": [r for r in sel if r["kind"] != "off_limits"], "off_limits": [r for r in sel if r["kind"] == "off_limits"]})
    return out


@app.get("/api/export/{hid}/preview")
def export_preview(hid: str):
    b = build_bundle(hid)
    names = ["README.txt", "profile.json", "constraints.json", "preferences.jsonl", "meta/info.json", "meta/tasks (preview)", "demos/index.jsonl"]
    return {"slug": b["slug"], "parquet": b["parquet"],
            "files": [{"path": n, "text": b["texts"][n][:20000]} for n in names],
            "clips": sorted(b["clips"].keys()), "tasks": len(b["table"])}


@app.get("/api/export/{hid}/validate")
def export_validate(hid: str):
    return {"checks": validate_bundle(build_bundle(hid))}


@app.get("/api/export/{hid}.zip")
def export_zip(hid: str):
    b = build_bundle(hid)
    buf = io.BytesIO()
    root = f"everyday-ai-{b['slug']}"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path, data in b["files"].items():
            z.writestr(f"{root}/{path}", data)
        for path, fp in b["clips"].items():
            z.write(fp, f"{root}/{path}")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{root}.zip"'})


app.mount("/files", StaticFiles(directory=UPLOADS), name="files")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0" if "--lan" in sys.argv else "127.0.0.1"
    uvicorn.run(app, host=host, port=int(os.environ.get("PORT", "8000")))
