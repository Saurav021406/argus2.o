Argus learned-agent lab
Welcome! This guide walks you through the demonstration-driven vision-and-action pipeline in argus/ — from recording your first desktop demonstration to training and evaluating a policy.

Know what this is: the current planner supports a bounded set of tasks: opening Notepad, Calculator, or Paint; supported calculator button sequences; and drawing a line or a three-stroke zigzag in Paint. It is an experimental learned-agent workflow, not a general-purpose autonomous desktop agent.

This pipeline is separate from the Desktop, MySQL, and ML-monitoring MCP servers in argus_server.py.

Start here: what are you trying to do?
<details open>
<summary><strong>🧰 I am setting this up for the first time</strong></summary>1. Use Windows and Python 3.10 or newer.
2. Create and activate the project environment.
3. Install the core requirements.
4. Record a small batch of demonstrations.
5. Audit the demonstrations before training.
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m argus.data session --plan first
python -m argus.data audit
</details><details>
<summary><strong>🎥 I want to record demonstrations</strong></summary>Start a guided session:
python -m argus.data session --plan first
Or record one supported task:
python -m argus.data record --instruction "Open Notepad" --task open_app --app notepad
Recording starts after a countdown. Press F10 to stop. The recorder captures screen frames and input events, then writes the episode and action labels under datasets/raw by default.

Available guided plans are first, calc, press, extras, press2, and extras2. See the exact descriptions with:
python -m argus.data session --help
</details><details>
<summary><strong>🧪 I want to check whether my data is good</strong></summary>Run the checks before feature caching:
python -m argus.data validate datasets/raw/<episode>
python -m argus.data audit
python -m argus.data coverage
python -m argus.data stats
python -m argus.data overlay datasets/raw/<episode>
- Validate checks episode structure and referenced frame data.
- Audit checks whether the recorded actions match the declared task and supported plan.
- Coverage reports whether calculator buttons have enough consistent examples.
- Overlay draws action labels on the frames so you can visually inspect them.
</details><details>
<summary><strong>🧠 I want to cache features and train</strong></summary>Install PyTorch first, using a build appropriate for your hardware. Then install the additional BC requirements:
python -m pip install -r requirements-bc.txt
python -m argus.training cache --root datasets/raw --out datasets/features
python -m argus.training train --features datasets/features --out checkpoints/bc_v1.pt
The BC requirements file includes Transformers and NumPy and assumes PyTorch is already installed. The first real feature-cache run downloads the configured DINOv2-small model weights.
</details><details>
<summary><strong>▶️ I want to try a trained checkpoint</strong></summary>
Start with dry-run. It predicts on the live screen but does not execute mouse or keyboard actions.
python -m argus.core run --ckpt checkpoints/bc_v1.pt --instruction "Open Notepad" --dry-run
Only consider a live run after reviewing the checkpoint, the instruction, and your desktop:
python -m argus.core run --ckpt checkpoints/bc_v1.pt --instruction "Open Notepad"
</details><details>
<summary><strong>📊 I want to evaluate or improve a checkpoint</strong></summary>python -m argus.core suite --ckpt checkpoints/bc_v1.pt --levels 1,2
python -m argus.core summary
Other implemented commands:
python -m argus.core dagger --help
python -m argus.core explain --help
python -m argus.core compare --help
python -m argus.core inkcheck --help
DAgger collects learner-visited states and expert-labelled actions for later training rounds. Review prompt coverage and live execution settings before running it.
</details>The pipeline, at a glance
Human demonstration
        │
        ▼
Frame + input-event recording
        │
        ▼
Action labelling ──► validation / audit / coverage
        │
        ▼
Frozen DINOv2 feature cache
        │
        ▼
PyTorch behavioral cloning
        │
        ▼
Checkpoint ──► dry-run ──► evaluation
        │
        └──────── optional DAgger iteration
1. Environment setup
Use Windows and Python 3.10 or newer. Run commands from the repository root:
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
<details>
<summary><strong>Which requirements file do I need?</strong></summary>File	Purpose
requirements-core.txt	Screen capture/input and basic learned-agent workflow dependencies, plus pytest
requirements-bc.txt	Transformers and NumPy for the feature/policy workflow; PyTorch must already be installed
requirements.txt	Broader project dependency profile for the legacy MCP server and integrations


These are different dependency profiles, not interchangeable lockfiles.
</details>2. Record demonstrations
The recorder captures screen frames and mouse/keyboard events. It converts those events into frame-aligned discrete action labels.
python -m argus.data session --plan first
To record one task directly:
python -m argus.data record --instruction "Open Notepad" --task open_app --app notepad
Press F10 to stop recording.
<details>
<summary><strong>What is stored in an episode?</strong></summary>
An episode includes metadata, frame references and JPEG frames, raw input events, and frame-aligned action labels. DAgger episodes can additionally store the learner's executed actions, keeping the expert target separate from what the learner actually did.

Default raw-data location: datasets/raw.
</details>3. Validate, audit, and inspect
python -m argus.data validate datasets/raw/<episode>
python -m argus.data audit
python -m argus.data coverage
python -m argus.data stats
python -m argus.data overlay datasets/raw/<episode>
<details>
<summary><strong>Replay a demonstration safely</strong></summary>python -m argus.data replay datasets/raw/<episode> --dry-run
A replay without --dry-run executes the stored actions on the live desktop. Remove the flag only when you deliberately intend to replay those actions.
</details>4. Cache visual features
python -m argus.training cache --root datasets/raw --out datasets/features
The cache step skips episodes that fail the strict audit by default. It uses the frozen DINOv2-small visual encoder and stores spatial patch features so training does not need to recompute the vision model every epoch.
<details>
<summary><strong>Smoke-test the pipeline without downloading DINOv2</strong></summary>python -m argus.training cache --root datasets/raw --out datasets/features_test --fake
The random encoder is deterministic test data. It is useful for a pipeline smoke test, not a substitute for the real vision encoder when training a usable policy.
</details>5. Train behavioral cloning
python -m argus.training train --features datasets/features --out checkpoints/bc_v1.pt
The resulting PyTorch checkpoint stores policy weights, configuration, vocabulary, and metadata about the goals/actions seen during training.
How to read the metrics: offline teacher-forced accuracy is a training diagnostic. It is not the same as successful live task completion.

6. Run the agent
<details open>
<summary><strong>Recommended: dry-run first</strong></summary>python -m argus.core run --ckpt checkpoints/bc_v1.pt --instruction "Open Notepad" --dry-run
This predicts on the live screen but does not execute mouse or keyboard actions.
</details><details>
<summary><strong>Live execution (review before running)</strong></summary>python -m argus.core run --ckpt checkpoints/bc_v1.pt --instruction "Open Notepad"
The agent uses an observe → predict → act → wait-for-UI → observe cycle. The executor enables a PyAutoGUI failsafe; moving the mouse to a screen corner aborts. Keep your hands off the mouse and keyboard during a run.
</details>
A checkpoint records which action types and planner goals appeared in its training data. Requests that need untrained goals can be rejected rather than guessed.
7. Evaluate and iterate
python -m argus.core suite --ckpt checkpoints/bc_v1.pt --levels 1,2
python -m argus.core summary
The task suite includes calculator checks, refusal cases for unsupported instructions, Paint checks based on observed canvas ink, and held-out prompt support. Results and traces are written under the selected output directory (defaults are under runs).
<details>
<summary><strong>DAgger, explanations, comparisons, and Paint ink checks</strong></summary>python -m argus.core dagger --help
python -m argus.core explain --help
python -m argus.core compare --help
python -m argus.core inkcheck --help
DAgger collects learner-visited states and expert-labelled actions for subsequent training rounds. Calculator expert labels use demonstrated key locations; Paint stroke labels are derived from demonstration strokes. Review the rollout prompts, expert coverage, and live execution mode before running DAgger on a real desktop.
</details>8. Current limitations
- Task planning is template-based and intentionally narrow.
- The checkpoint cannot reliably act on goals absent from its training examples.
- Calculator button behaviour depends on Windows Calculator layout and consistent recorded click locations.
- TYPE execution uses PyAutoGUI's ASCII-oriented write path.
- The recorder defaults to the primary monitor (monitor=1); multi-monitor capture is not a generalised feature of this pipeline.
- Real training needs PyTorch and downloaded DINOv2 weights; the fake encoder is test-only.
- Live execution manipulates the actual desktop. Dry-run, action bounds, and the mouse-corner failsafe are guardrails, not a safety guarantee.
9. Tests
Run from the repository root after installing test and optional dependencies:
python -m pytest -q
The suite covers planner behaviour, recording/data validation, model shapes/training, DAgger, evaluation, and execution safeguards. Some tests depend on optional packages or Windows-specific components.
<sub>Suggested first milestone: record a few clean demonstrations → audit them → inspect overlays → cache features → train → dry-run → evaluate.</sub>
