# Everyday AI (local version)

Teach a home robot how *your* home works. You show a chore (photo or video, plus a sentence), the app turns it
into editable rules, and the result is a **household profile** a robot team can load.

Two sample homes are pre-filled. **My home** starts empty: everything you teach, correct or delete is saved there.

## Run it (Windows PowerShell)

```powershell
cd path\to\everyday-ai-local
python -m pip install -r requirements.txt
python server.py
```

Open http://127.0.0.1:8000 (or double-click `run.bat`).

First start downloads two small models (about 90 MB and 3 GB). The app works meanwhile, using keyword search and
template wording. The line under the title shows what is ready.

Options (set in PowerShell before `python server.py`):
`$env:EVERYDAY_LLM="0"` (no language model), `$env:EVERYDAY_NO_MODELS="1"` (no models), `$env:LLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"`, `$env:PORT="8001"`. Use `--lan` to open it from a phone on the same wifi.

## Tabs

- **Home**: rules by room, plus "ask this home" (search by meaning) and the demos.
- **Teach**: photo/video (optional), title, room, chore, note. "Suggest rules" drafts rules you can edit; Save adds them to My home.
- **Try it**: type a chore. Shows a general robot's plan next to the plan with the home's profile and the instruction text. "That's not right" lets you correct a rule; it is logged and applied. Off-limits rooms block the plan. **The robot is simulated.**
- **Compare homes**: the same chore across all homes.
- **Robot export**: download a zip and run a check on it.

## What the export contains

```
everyday-ai-<home>/
  README.txt            honest description of limits
  profile.json          rooms, rules, demos (our schema)
  constraints.json      off-limits places and "never" rules
  preferences.jsonl     one rule per line
  meta/info.json        LeRobot v3 field names (codebase_version "v3.0"), metadata only
  meta/tasks.parquet    task (language instruction) + task_index, as in LeRobot v3
  demos/index.jsonl     each demo, its instruction, its clip file
  demos/files/          your uploaded clips
```

**Important limit:** robot training data also needs per-frame `observation.state` and `action` streams, plus episode
data files. A homeowner's demos don't contain those, so this bundle is *not* a complete LeRobot dataset. It supplies the
instructions and constraints to use alongside the robot's own recordings. The layout follows the LeRobot v3 conventions
as documented at the time of writing; verify against the version your robot team uses. The "Check this bundle" button
only validates our own files, not a real training pipeline.

## Files

`server.py` (backend, models, export), `static/index.html` (UI), `everyday_homes.db` and `uploads/` (created on use; delete to reset).

## Troubleshooting

- `torchvision::nms does not exist`: `python -m pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`
- No `pyarrow`: tasks are written as `meta/tasks.jsonl` instead of parquet.
- Hover the status line to see model load errors.
- No login: anyone who can open the page can change My home.
- Sample demos show labeled placeholder frames, not footage.
