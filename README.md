Argus learned-agent workflow
This guide covers the newer demonstration-driven vision and action pipeline in argus/. It is separate from the legacy desktop, MySQL, and ML-monitoring MCP servers in argus_server.py.

The current learned-agent planner supports a bounded set of task patterns:
- Open Notepad, Calculator, or Paint.
- Perform calculator button sequences supported by the task parser and the demonstrations available to the checkpoint.
- Draw a line or a three-stroke zigzag in Paint.
It is not a general-purpose agent that can interpret and carry out arbitrary desktop instructions.
1. Environment
Use Python 3.10 or newer on Windows. From the repository root, create and activate an environment:
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
For feature caching and behavioral-cloning training, install PyTorch into the environment first, using a build appropriate for your hardware, then install:
python -m pip install -r requirements-bc.txt
The BC requirements file includes Transformers and NumPy and assumes PyTorch is already installed. The first real feature-cache run downloads the configured DINOv2-small model weights. The --fake encoder option is for pipeline smoke tests only.
2. Record demonstrations
Start a guided batch:
python -m argus.data session --plan first
The session command supports first, calc, press, extras, press2, and extras2; inspect its help for the exact batch descriptions:
python -m argus.data session --help
You can also record one task directly:
python -m argus.data record --instruction "Open Notepad" --task open_app --app notepad
Recording starts after a countdown. Press F10 to stop. The recorder captures screen frames and input events, then writes the episode and action labels below datasets/raw by default.
3. Validate and audit data
python -m argus.data validate datasets/raw/<episode>
python -m argus.data audit
python -m argus.data coverage
python -m argus.data stats
python -m argus.data overlay datasets/raw/<episode>
Validation checks the episode structure and referenced frame data. The task audit checks whether the demonstration content agrees with its declared task and supported plan. Coverage reports whether the calculator buttons have enough consistent examples.

To preview a replay without sending mouse or keyboard input:
python -m argus.data replay datasets/raw/<episode> --dry-run
A replay without --dry-run executes the stored actions on the live desktop.
4. Cache visual features
python -m argus.training cache --root datasets/raw --out datasets/features
The cache step skips episodes that fail the strict audit by default. It runs the frozen DINOv2-small visual encoder and stores spatial patch features so training does not need to recompute the vision model on every epoch.

For a no-download pipeline test:
python -m argus.training cache --root datasets/raw --out datasets/features_test --fake
Random features are deterministic test data; they are not a substitute for the real vision encoder when training a usable policy.
5. Train behavioral cloning
python -m argus.training train --features datasets/features --out checkpoints/bc_v1.pt
Training creates a PyTorch checkpoint containing the policy weights, configuration, vocabulary, and metadata about the goals/actions seen during training. The training code reports validation and baseline metrics. Offline teacher-forced accuracy is a diagnostic; it is not the same as successful live task completion.
6. Run the agent
Start with a dry run:
python -m argus.core run --ckpt checkpoints/bc_v1.pt --instruction "Open Notepad" --dry-run
For a live run, remove --dry-run only after reviewing the instruction, checkpoint coverage, and desktop state:
python -m argus.core run --ckpt checkpoints/bc_v1.pt --instruction "Open Notepad"
The agent uses a closed-loop observe → predict → act → wait-for-UI → observe cycle. The executor has a PyAutoGUI failsafe; move the mouse to a screen corner to abort. Keep your hands off the mouse and keyboard during a run.

A checkpoint records which action types and planner goals appeared in its training data. Requests that need untrained goals can be rejected rather than guessed.
7. Evaluate and iterate
python -m argus.core suite --ckpt checkpoints/bc_v1.pt --levels 1,2
python -m argus.core summary
Other implemented core commands include:
python -m argus.core dagger --help
python -m argus.core explain --help
python -m argus.core compare --help
python -m argus.core inkcheck --help
DAgger collects learner-visited states and expert-labelled actions for subsequent training rounds. Calculator expert labels use the demonstrated key locations; Paint stroke labels are derived from demonstration strokes. Review the rollout prompts, expert coverage, and live execution mode before running DAgger on a real desktop.

The task suite contains calculator checks, refusal cases for unsupported instructions, Paint checks based on observed canvas ink, and held-out prompt support. Results and traces are written under the selected output directory (defaults are under runs).
8. Current limitations
- Task planning is template-based and intentionally narrow.
- The checkpoint cannot reliably act on goals absent from its training examples.
- Calculator button behaviour depends on Windows Calculator layout and the consistency of recorded click locations.
- TYPE execution uses PyAutoGUI's ASCII-oriented write path.
- The recorder defaults to the primary monitor (monitor=1); multi-monitor capture is not a generalised feature of this pipeline.
- Real training needs PyTorch and downloaded DINOv2 weights; the fake encoder is test-only.
- Live execution manipulates the actual desktop. Dry-run, action bounds, and the mouse-corner failsafe are useful guardrails, not a safety guarantee.
9. Tests
Run the repository tests from its root after installing the test and optional dependencies:
python -m pytest -q
The suite covers planner behaviour, recording/data validation, model shapes/training, DAgger, evaluation, and execution safeguards. Some tests depend on optional packages or Windows-specific components.
