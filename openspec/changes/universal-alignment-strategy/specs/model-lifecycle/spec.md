## ADDED Requirements

### Requirement: Every GPU-resident stage exposes load and unload entry points

Each pipeline stage that loads a model onto a GPU (separator, diarizer, transcriber, aligner, sentiment analyzer) SHALL expose explicit `load_model` and `unload_model` entry points. The stage module SHALL NOT rely on Python garbage collection, process exit, or implicit cleanup to free GPU memory.

#### Scenario: Each stage module exports the lifecycle functions

- **WHEN** inspecting each stage module's public API
- **THEN** every stage that touches a GPU model has `load_model` and `unload_model` callable entry points, and their signatures are documented in the module docstring

### Requirement: Unload releases VRAM deterministically

The system SHALL ensure that calling `unload_model` on any stage releases the stage's GPU memory before returning. Implementations SHALL invoke `gc.collect()` and `torch.cuda.empty_cache()` as part of unload. After unload, `torch.cuda.memory_allocated(device)` SHALL be within 50 MB of the pre-load baseline on the stage's target device.

#### Scenario: Unload releases VRAM for each stage

- **WHEN** each GPU-resident stage is loaded, runs one unit of work, and is unloaded in turn
- **THEN** `torch.cuda.memory_allocated(device)` after each stage's `unload_model` is within 50 MB of the pre-load baseline

### Requirement: Pipeline supports co-resident and sequential VRAM strategies

The pipeline SHALL support two VRAM strategies selectable via `--vram-strategy`:

- **co-resident**: All in-process GPU models (pyannote, ASR, aligner) are loaded before the file loop and remain resident for the entire batch. This is the fastest mode because model load time is paid once.
- **sequential**: Each stage loads its model, processes its workload, and unloads before the next stage loads. Peak VRAM equals the single largest stage.

The system SHALL default to co-resident when the VRAM preflight check projects that all models fit. The system SHALL automatically fall back to sequential and emit a warning when co-resident is projected to exceed available VRAM.

#### Scenario: Co-resident mode on a GPU with sufficient VRAM

- **WHEN** the pipeline runs with `--vram-strategy co-resident` on a GPU where the projected co-resident budget fits within available VRAM
- **THEN** all in-process GPU models are loaded once before the file loop begins, no `unload_model` calls occur between stages during the loop, and total wall-clock model loading time equals the sum of each model's single load time

#### Scenario: Sequential mode on a constrained GPU

- **WHEN** the pipeline runs with `--vram-strategy sequential` on any GPU
- **THEN** at most one heavyweight GPU model is resident at any time, and `torch.cuda.max_memory_allocated` during any stage reflects only that stage's model plus working memory

#### Scenario: Automatic fallback from co-resident to sequential

- **WHEN** the user does not specify `--vram-strategy` and the VRAM preflight projects that co-resident mode exceeds available VRAM
- **THEN** the pipeline falls back to sequential mode and emits a warning explaining the projected budget, available VRAM, and the fallback decision

### Requirement: VRAM budget registry provides per-model cost estimates

The system SHALL maintain a `model_budgets.toml` file in `src/` that maps model identifiers to their projected VRAM cost in MiB (weights plus typical peak activation overhead). The registry SHALL be user-editable and SHALL be the single source of truth for VRAM preflight calculations.

#### Scenario: Looking up a known model

- **WHEN** the preflight system looks up `"faster-whisper/large-v3"` in `model_budgets.toml`
- **THEN** it returns a VRAM estimate in MiB that reflects the model's weight size plus typical peak activation overhead at default batch size

#### Scenario: Looking up an unknown model

- **WHEN** a model identifier is not found in `model_budgets.toml`
- **THEN** the preflight system logs a warning identifying the unknown model, uses a conservative fallback estimate, and does not crash

### Requirement: VRAM preflight validates the pipeline configuration before processing

The pipeline SHALL run a VRAM budget preflight check before any models are loaded or files are processed. The preflight SHALL:

1. Query each target GPU's total VRAM and currently used VRAM via `gpu_utils`.
2. Subtract a configurable headroom margin (default 512 MiB) for CUDA context and fragmentation.
3. Look up the projected VRAM cost of each enabled stage's model from `model_budgets.toml`.
4. Compute the projected budget: sum of all models for co-resident, max of any single model for sequential.
5. Compare projected budget against available VRAM.

If the budget exceeds available VRAM, the preflight SHALL block the pipeline with an actionable error message that includes: per-model VRAM breakdown, total projected budget, available VRAM on the target GPU(s), and specific suggestions (switch VRAM strategy, use a smaller transcription model, free GPU memory, or switch to a different GPU).

#### Scenario: Preflight passes for co-resident mode

- **WHEN** the pipeline starts with co-resident mode and the sum of all model budgets plus headroom is less than available GPU VRAM
- **THEN** the preflight passes, logs the projected budget and available VRAM, and the pipeline proceeds to load models

#### Scenario: Preflight blocks an over-budget co-resident run

- **WHEN** the pipeline starts with co-resident mode and the sum of all model budgets plus headroom exceeds available GPU VRAM
- **THEN** the pipeline does not load any models, does not process any files, and prints an error that includes: each model's VRAM cost, the total projected budget, the available VRAM, and at least two actionable suggestions

#### Scenario: Preflight checks the smallest GPU in multi-GPU parallel runs

- **WHEN** the pipeline runs in parallel mode across multiple GPUs
- **THEN** the preflight validates the budget against the GPU with the smallest available VRAM, since all workers run identical configurations

### Requirement: Pipeline reports peak VRAM per stage

The pipeline's per-stage result SHALL record the peak VRAM observed during that stage's execution in bytes, measured via `torch.cuda.max_memory_allocated` after resetting the peak counter at the start of the stage. Reports and notifications SHALL surface this value.

#### Scenario: Recording and reporting peak VRAM

- **WHEN** the pipeline completes a stage on a CUDA device
- **THEN** the stage's result entry contains a non-null `peak_vram_bytes` field, and the CLI/Slack report shows a human-readable value (e.g., "6.3 GB") for that stage

### Requirement: Interactive run planner validates and configures pipeline runs

The system SHALL provide an `audio-refinery plan` CLI command that interactively presents the pipeline configuration, validates it against available hardware, and allows the user to adjust settings before committing to a run.

The planner SHALL display: discovered file count and total audio duration, GPU specs and available VRAM, enabled pipeline stages with their model choices, a VRAM budget visualization comparing projected usage against available capacity, and an estimated run time.

The planner SHALL allow the user to toggle stages on/off, switch transcription models, change VRAM strategy, and select target device(s). Each change SHALL immediately re-render the budget and time estimate.

#### Scenario: Planning a run on a single GPU

- **WHEN** the user runs `audio-refinery plan --source-dir /audio/test`
- **THEN** the planner displays the file count, GPU info, enabled stages with models, a VRAM budget visualization, and run time estimate, and waits for user input before proceeding

#### Scenario: Adjusting the transcription model in the planner

- **WHEN** the user selects a different transcription model (e.g., switching from large-v3 to medium) during the planning session
- **THEN** the VRAM budget visualization and run time estimate update immediately to reflect the new model's resource requirements

#### Scenario: Planner detects a VRAM budget problem

- **WHEN** the user's selected configuration exceeds available VRAM in co-resident mode
- **THEN** the planner highlights the budget overflow, suggests specific alternatives (switch to sequential, use a smaller model, disable a stage), and does not allow the run to proceed until the configuration fits or the user explicitly switches to sequential mode

#### Scenario: Non-interactive environment

- **WHEN** `audio-refinery plan` is invoked without a TTY (e.g., in a CI pipeline)
- **THEN** the command exits with a clear message directing the user to use `pipeline --plan profile.toml` for headless execution

### Requirement: Run profiles save and replay validated pipeline configurations

The system SHALL support saving a validated pipeline configuration from the planner as a **run profile** (TOML file). The `pipeline` command SHALL accept a `--plan <profile.toml>` flag that loads and replays the saved configuration without requiring interactive input.

#### Scenario: Saving a run profile

- **WHEN** the user confirms a configuration in the planner and chooses to save it
- **THEN** the system writes a TOML file containing all pipeline settings (source dir, stages, models, VRAM strategy, devices, language, compute type, batch size) and the file is valid input for `pipeline --plan`

#### Scenario: Replaying a run profile

- **WHEN** the user runs `audio-refinery pipeline --plan my-batch.toml`
- **THEN** the pipeline loads all settings from the TOML, runs the VRAM preflight against current GPU state, and proceeds if the budget fits

#### Scenario: Replaying a profile on a different GPU

- **WHEN** a run profile saved on a machine with a 4090 is replayed on a machine with a 3060
- **THEN** the VRAM preflight re-validates against the current GPU's available VRAM and blocks or auto-falls-back to sequential if the saved co-resident configuration no longer fits
