# Phase 5: Tauri UI & Configuration Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose VLM pipeline features (state diff, error correction filter, annotation visuals) as user-configurable settings through Rust config → Tauri commands → React UI.

**Architecture:** Four-layer config plumbing. New fields in `PipelineConfig` and a new `AnnotationConfig` section in the TOML config propagate through `UiConfig` (Tauri) to React toggle checkboxes and an advanced "Visual Annotation" settings section.

**Tech Stack:** Rust (serde/toml), Tauri 2 commands, React 19 + TypeScript

---

### Task 1: Add pipeline toggles + AnnotationConfig to Rust config

**Files:**
- Modify: `skill-generator/src/app/config.rs:64-91` (PipelineConfig struct + Default)
- Modify: `skill-generator/src/app/config.rs:7-19` (Config struct)

**Step 1: Add 2 new fields to `PipelineConfig`**

In `skill-generator/src/app/config.rs`, add to the `PipelineConfig` struct (after line 78):
```rust
    /// Use before/after screenshot state diffing for UnitTask enrichment
    #[serde(default = "default_true")]
    pub use_state_diff: bool,
    /// Filter out error correction tasks (undo, cancel, backspace-only)
    #[serde(default = "default_true")]
    pub use_error_correction_filter: bool,
```

Add a helper at the top of the file:
```rust
fn default_true() -> bool { true }
```

Update `Default for PipelineConfig` to include:
```rust
            use_state_diff: true,
            use_error_correction_filter: true,
```

**Step 2: Add `AnnotationConfig` to `Config` struct**

Add a new serde-compatible `AnnotationConfig` struct (note: this is the *config file* version, separate from `screenshot.rs`'s `AnnotationConfig`):

```rust
/// Visual annotation settings (dot, trajectory, crop)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationSettings {
    #[serde(default = "default_dot_radius")]
    pub dot_radius: u32,
    #[serde(default = "default_dot_color")]
    pub dot_color: [u8; 3],
    #[serde(default = "default_trajectory_ballistic_color")]
    pub trajectory_ballistic_color: [u8; 3],
    #[serde(default = "default_trajectory_searching_color")]
    pub trajectory_searching_color: [u8; 3],
    #[serde(default = "default_trajectory_thickness")]
    pub trajectory_thickness: u32,
    #[serde(default = "default_crop_size")]
    pub crop_size: u32,
}
```

Add default functions and Default impl matching the hardcoded values in `screenshot.rs`:
- dot_radius: 12, dot_color: [255, 40, 40]
- trajectory_ballistic_color: [40, 220, 40], trajectory_searching_color: [255, 160, 40]
- trajectory_thickness: 2, crop_size: 512

Add to Config struct:
```rust
    /// Visual annotation settings
    #[serde(default)]
    pub annotation: AnnotationSettings,
```

**Step 3: Update test literals**

Update the test `test_config_with_custom_pipeline_roundtrip` to include new pipeline fields. Update `test_pipeline_config_serialization_roundtrip` similarly.

Add a new test `test_annotation_settings_defaults` verifying all default values.

Add a test `test_old_config_without_annotation_section_deserializes` proving backward compat.

**Step 4: Run tests**

Run: `cargo test -p skill_generator -- config`
Expected: All config tests pass

**Step 5: Commit**

```bash
git add skill-generator/src/app/config.rs
git commit -m "feat: add state_diff, error_correction_filter toggles and AnnotationSettings to config"
```

---

### Task 2: Wire new config fields through Tauri UiConfig

**Files:**
- Modify: `skill-generator-ui/src-tauri/src/lib.rs:1289-1361` (UiConfig, get_config, save_config)
- Modify: `skill-generator-ui/src-tauri/src/lib.rs:203-216` (PipelineStatsInfo)

**Step 1: Add fields to `UiConfig` struct (line ~1307)**

After `use_context_tracking`:
```rust
    #[serde(default = "default_true")]
    pub use_state_diff: bool,
    #[serde(default = "default_true")]
    pub use_error_correction_filter: bool,
```

After `capture_screenshots`:
```rust
    // Annotation settings
    #[serde(default = "default_dot_radius")]
    pub dot_radius: u32,
    #[serde(default = "default_dot_color")]
    pub dot_color: [u8; 3],
    #[serde(default = "default_trajectory_ballistic_color")]
    pub trajectory_ballistic_color: [u8; 3],
    #[serde(default = "default_trajectory_searching_color")]
    pub trajectory_searching_color: [u8; 3],
    #[serde(default = "default_trajectory_thickness")]
    pub trajectory_thickness: u32,
    #[serde(default = "default_crop_size")]
    pub crop_size: u32,
```

Add the default functions:
```rust
fn default_true() -> bool { true }
fn default_dot_radius() -> u32 { 12 }
fn default_dot_color() -> [u8; 3] { [255, 40, 40] }
fn default_trajectory_ballistic_color() -> [u8; 3] { [40, 220, 40] }
fn default_trajectory_searching_color() -> [u8; 3] { [255, 160, 40] }
fn default_trajectory_thickness() -> u32 { 2 }
fn default_crop_size() -> u32 { 512 }
```

**Step 2: Add new stats fields to `PipelineStatsInfo` (line ~215)**

```rust
    pub state_diff_enriched_count: usize,
    pub error_correction_count: usize,
```

**Step 3: Wire `get_config()` (line ~1318)**

Add to the `Ok(UiConfig { ... })` block:
```rust
        use_state_diff: config.pipeline.use_state_diff,
        use_error_correction_filter: config.pipeline.use_error_correction_filter,
        dot_radius: config.annotation.dot_radius,
        dot_color: config.annotation.dot_color,
        trajectory_ballistic_color: config.annotation.trajectory_ballistic_color,
        trajectory_searching_color: config.annotation.trajectory_searching_color,
        trajectory_thickness: config.annotation.trajectory_thickness,
        crop_size: config.annotation.crop_size,
```

**Step 4: Wire `save_config()` (line ~1345)**

Add after the existing pipeline field assignments:
```rust
    full_config.pipeline.use_state_diff = config.use_state_diff;
    full_config.pipeline.use_error_correction_filter = config.use_error_correction_filter;
    full_config.annotation.dot_radius = config.dot_radius;
    full_config.annotation.dot_color = config.dot_color;
    full_config.annotation.trajectory_ballistic_color = config.trajectory_ballistic_color;
    full_config.annotation.trajectory_searching_color = config.trajectory_searching_color;
    full_config.annotation.trajectory_thickness = config.trajectory_thickness;
    full_config.annotation.crop_size = config.crop_size;
```

**Step 5: Wire stats in `generate_skill()` (line ~1014)**

Add to the `PipelineStatsInfo { ... }` block:
```rust
                    state_diff_enriched_count: skill.stats.state_diff_enriched_count,
                    error_correction_count: skill.stats.error_correction_count,
```

**Step 6: Build**

Run: `cd skill-generator-ui/src-tauri && cargo build`
Expected: Compiles successfully

**Step 7: Commit**

```bash
git add skill-generator-ui/src-tauri/src/lib.rs
git commit -m "feat: wire new pipeline toggles, annotation settings, and stats through Tauri UiConfig"
```

---

### Task 3: Wire GeneratorConfig toggles in generate_skill()

**Files:**
- Modify: `skill-generator-ui/src-tauri/src/lib.rs:950-960` (GeneratorConfig construction)
- Modify: `skill-generator/src/workflow/generator.rs:42-97` (GeneratorConfig struct + Default)
- Modify: `skill-generator/src/workflow/generator.rs:292-306` (state diff + error correction gating)

**Step 1: Add toggle fields to GeneratorConfig struct**

In `generator.rs`, add to `GeneratorConfig` after `use_context_tracking`:
```rust
    /// Use before/after screenshot state diffing for UnitTask enrichment
    pub use_state_diff: bool,
    /// Filter out error correction tasks (undo, cancel, backspace-only)
    pub use_error_correction_filter: bool,
```

Update `Default for GeneratorConfig`:
```rust
            use_state_diff: true,
            use_error_correction_filter: true,
```

**Step 2: Gate state diff on config toggle**

In `generator.rs` line ~292, change:
```rust
        if let (Some(ref mut tasks), Some(ref rt)) = (&mut unit_tasks, &runtime) {
            if self.config.recording_dir.is_some() {
```
to:
```rust
        if let (Some(ref mut tasks), Some(ref rt)) = (&mut unit_tasks, &runtime) {
            if self.config.use_state_diff && self.config.recording_dir.is_some() {
```

**Step 3: Gate error correction filtering on config toggle**

In `generator.rs` line ~1235, change the skip in `generate_steps_from_tasks()`:
```rust
            if task.is_error_correction {
```
to:
```rust
            if self.config.use_error_correction_filter && task.is_error_correction {
```

Also gate the counting at line ~300:
```rust
        if let Some(ref tasks) = unit_tasks {
            if self.config.use_error_correction_filter {
                stats.error_correction_count = tasks.iter().filter(|t| t.is_error_correction).count();
                ...
            }
        }
```

**Step 4: Gate error correction in skill_compiler.rs**

In `skill-generator/src/codegen/skill_compiler.rs` line ~148, the pre-filter needs a way to be disabled. The simplest approach: do NOT gate it here — the compiler always filters error corrections. The toggle controls only the generator pipeline counting + step generation. The compiler-level filter is a safety net.

(No change needed to skill_compiler.rs.)

**Step 5: Wire in Tauri generate_skill()**

In `lib.rs` line ~950, add to the `GeneratorConfig { ... }`:
```rust
            use_state_diff: config.pipeline.use_state_diff,
            use_error_correction_filter: config.pipeline.use_error_correction_filter,
```

**Step 6: Run tests**

Run: `cd skill-generator && cargo test`
Expected: All tests pass

**Step 7: Build Tauri**

Run: `cd skill-generator-ui/src-tauri && cargo build`
Expected: Compiles

**Step 8: Commit**

```bash
git add skill-generator/src/workflow/generator.rs skill-generator-ui/src-tauri/src/lib.rs
git commit -m "feat: gate state diff and error correction filter on config toggles"
```

---

### Task 4: Pass AnnotationConfig through screenshot analysis pipeline

**Files:**
- Modify: `skill-generator/src/semantic/screenshot_analysis.rs:297-323` (run_claude_vision_analysis signature)
- Modify: `skill-generator/src/semantic/screenshot_analysis.rs:440-508` (analyze_recording)
- Modify: `skill-generator-ui/src-tauri/src/lib.rs:1064-1078` (analyze_recording Tauri command)

**Step 1: Add `annotation_config` parameter to `run_claude_vision_analysis`**

Change signature at line ~297:
```rust
pub async fn run_claude_vision_analysis(
    low_confidence: &[ScreenshotAnalysis],
    screenshots_dir: &Path,
    events: &[EnrichedEvent],
    goal: &str,
    api_key: &str,
    config: &AnalysisConfig,
    annotation_config: &crate::semantic::screenshot::AnnotationConfig,
    client: &reqwest::Client,
) -> Vec<ScreenshotAnalysis> {
```

Replace line 323 `let ann_config = ...::default();` with use of the parameter:
```rust
    let ann_config = annotation_config.clone();
```

**Step 2: Add `annotation_config` to `analyze_recording`**

Change signature at line ~440:
```rust
pub fn analyze_recording(
    recording_dir: &Path,
    recording: &Recording,
    api_key: Option<&str>,
    config: &AnalysisConfig,
    annotation_config: &crate::semantic::screenshot::AnnotationConfig,
) -> RecordingAnalysis {
```

Pass it through to `run_claude_vision_analysis` at line ~500:
```rust
                let vision_results = rt.block_on(run_claude_vision_analysis(
                    &low_confidence,
                    &screenshots_dir,
                    &recording.events,
                    goal,
                    key,
                    config,
                    annotation_config,
                    &client,
                ));
```

**Step 3: Build `AnnotationConfig` from `AnnotationSettings` in Tauri**

In `lib.rs` line ~1064, after loading `app_config`, construct the annotation config:
```rust
        let annotation_config = skill_generator::semantic::screenshot::AnnotationConfig {
            dot_radius: app_config.annotation.dot_radius as i32,
            dot_color: app_config.annotation.dot_color,
            box_color: [0, 255, 100],  // not configurable, keep default
            box_thickness: 2,          // not configurable, keep default
            crop_size: app_config.annotation.crop_size,
            jpeg_quality: 85,          // not configurable, keep default
            trajectory_ballistic_color: app_config.annotation.trajectory_ballistic_color,
            trajectory_searching_color: app_config.annotation.trajectory_searching_color,
            trajectory_thickness: app_config.annotation.trajectory_thickness,
        };
```

Pass it to `analyze_recording`:
```rust
        let analysis = sa::analyze_recording(
            &rec_dir,
            &recording,
            api_key_value.as_deref(),
            &config,
            &annotation_config,
        );
```

**Step 4: Update CLI main.rs call if it calls analyze_recording**

Check if `skill-generator/src/main.rs` calls `analyze_recording` — if so, pass `AnnotationConfig::default()`.

**Step 5: Run tests**

Run: `cd skill-generator && cargo test`
Expected: All tests pass (tests use `AnnotationConfig::default()` directly)

**Step 6: Build all**

Run: `cd skill-generator && cargo build --release && cd ../skill-generator-ui/src-tauri && cargo build`
Expected: Both compile

**Step 7: Commit**

```bash
git add skill-generator/src/semantic/screenshot_analysis.rs skill-generator-ui/src-tauri/src/lib.rs
git commit -m "feat: pass AnnotationConfig from settings through screenshot analysis pipeline"
```

---

### Task 5: Add React UI for new pipeline toggles and annotation settings

**Files:**
- Modify: `skill-generator-ui/src/App.tsx:67-81` (UiConfig interface)
- Modify: `skill-generator-ui/src/App.tsx:44-57` (PipelineStatsInfo interface)
- Modify: `skill-generator-ui/src/App.tsx:1463-1528` (Settings tab toggles)
- Modify: `skill-generator-ui/src/App.tsx:1043-1048` (Stats display)
- Modify: `skill-generator-ui/src/App.css` (new styles)

**Step 1: Update TypeScript interfaces**

Add to `PipelineStatsInfo` (after `screenshot_enhanced_count`):
```typescript
  state_diff_enriched_count: number;
  error_correction_count: number;
```

Add to `UiConfig` (after `capture_screenshots`):
```typescript
  use_state_diff: boolean;
  use_error_correction_filter: boolean;
  dot_radius: number;
  dot_color: number[];
  trajectory_ballistic_color: number[];
  trajectory_searching_color: number[];
  trajectory_thickness: number;
  crop_size: number;
```

**Step 2: Add pipeline toggle checkboxes**

After the "Context Tracking" toggle (line ~1517), add before the "Recording Features" heading:

```tsx
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={config.use_state_diff}
                    onChange={(e) => setConfig({ ...config, use_state_diff: e.target.checked })}
                  />
                  <span>State Diff Analysis</span>
                  <span className="config-hint">Before/after screenshot comparison for UI state changes</span>
                </label>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={config.use_error_correction_filter}
                    onChange={(e) => setConfig({ ...config, use_error_correction_filter: e.target.checked })}
                  />
                  <span>Error Correction Filter</span>
                  <span className="config-hint">Remove undo, cancel, and backspace-only actions from output</span>
                </label>
```

**Step 3: Add "Visual Annotation" collapsible section**

After "Recording Features" section and before the Save button, add:

```tsx
                <h3 className="config-section-title config-section-collapsible"
                    onClick={() => setShowAnnotationSettings(!showAnnotationSettings)}>
                  {showAnnotationSettings ? "▾" : "▸"} Visual Annotation
                </h3>
                {showAnnotationSettings && (
                  <div className="annotation-settings">
                    <label>
                      <span className="config-label">Dot Radius (px)</span>
                      <input type="number" min="4" max="24" step="1"
                        value={config.dot_radius}
                        onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, dot_radius: v }); }}
                      />
                    </label>
                    <label>
                      <span className="config-label">Dot Color (R,G,B)</span>
                      <div className="color-inputs">
                        <input type="number" min="0" max="255" value={config.dot_color[0]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) { const c = [...config.dot_color]; c[0] = v; setConfig({ ...config, dot_color: c }); }}} />
                        <input type="number" min="0" max="255" value={config.dot_color[1]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) { const c = [...config.dot_color]; c[1] = v; setConfig({ ...config, dot_color: c }); }}} />
                        <input type="number" min="0" max="255" value={config.dot_color[2]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) { const c = [...config.dot_color]; c[2] = v; setConfig({ ...config, dot_color: c }); }}} />
                        <span className="color-preview" style={{ backgroundColor: `rgb(${config.dot_color.join(',')})` }} />
                      </div>
                    </label>
                    <label>
                      <span className="config-label">Ballistic Trajectory Color</span>
                      <div className="color-inputs">
                        <input type="number" min="0" max="255" value={config.trajectory_ballistic_color[0]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) { const c = [...config.trajectory_ballistic_color]; c[0] = v; setConfig({ ...config, trajectory_ballistic_color: c }); }}} />
                        <input type="number" min="0" max="255" value={config.trajectory_ballistic_color[1]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) { const c = [...config.trajectory_ballistic_color]; c[1] = v; setConfig({ ...config, trajectory_ballistic_color: c }); }}} />
                        <input type="number" min="0" max="255" value={config.trajectory_ballistic_color[2]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) { const c = [...config.trajectory_ballistic_color]; c[2] = v; setConfig({ ...config, trajectory_ballistic_color: c }); }}} />
                        <span className="color-preview" style={{ backgroundColor: `rgb(${config.trajectory_ballistic_color.join(',')})` }} />
                      </div>
                    </label>
                    <label>
                      <span className="config-label">Searching Trajectory Color</span>
                      <div className="color-inputs">
                        <input type="number" min="0" max="255" value={config.trajectory_searching_color[0]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) { const c = [...config.trajectory_searching_color]; c[0] = v; setConfig({ ...config, trajectory_searching_color: c }); }}} />
                        <input type="number" min="0" max="255" value={config.trajectory_searching_color[1]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) { const c = [...config.trajectory_searching_color]; c[1] = v; setConfig({ ...config, trajectory_searching_color: c }); }}} />
                        <input type="number" min="0" max="255" value={config.trajectory_searching_color[2]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) { const c = [...config.trajectory_searching_color]; c[2] = v; setConfig({ ...config, trajectory_searching_color: c }); }}} />
                        <span className="color-preview" style={{ backgroundColor: `rgb(${config.trajectory_searching_color.join(',')})` }} />
                      </div>
                    </label>
                    <label>
                      <span className="config-label">Trajectory Thickness (px)</span>
                      <input type="number" min="1" max="5" step="1"
                        value={config.trajectory_thickness}
                        onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, trajectory_thickness: v }); }}
                      />
                    </label>
                    <label>
                      <span className="config-label">Crop Size (px)</span>
                      <input type="number" min="256" max="1024" step="64"
                        value={config.crop_size}
                        onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, crop_size: v }); }}
                      />
                    </label>
                  </div>
                )}
```

Add state variable:
```typescript
const [showAnnotationSettings, setShowAnnotationSettings] = useState(false);
```

**Step 4: Display new stats in generation results**

In the stats display area (line ~1043), add after the existing stats:
```tsx
                {pipelineStats.state_diff_enriched_count > 0 && ` / ${pipelineStats.state_diff_enriched_count} diffs`}
                {pipelineStats.error_correction_count > 0 && ` / ${pipelineStats.error_correction_count} filtered`}
```

**Step 5: Add CSS styles**

In `App.css`, add:
```css
.config-section-collapsible {
  cursor: pointer;
  user-select: none;
}

.config-section-collapsible:hover {
  color: var(--accent);
}

.annotation-settings {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-left: 8px;
  border-left: 2px solid var(--border);
}

.color-inputs {
  display: flex;
  gap: 6px;
  align-items: center;
}

.color-inputs input {
  width: 60px;
}

.color-preview {
  display: inline-block;
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: 1px solid var(--border);
  flex-shrink: 0;
}
```

**Step 6: Build frontend**

Run: `cd skill-generator-ui && npm run build`
Expected: Compiles successfully

**Step 7: Commit**

```bash
git add skill-generator-ui/src/App.tsx skill-generator-ui/src/App.css
git commit -m "feat: add UI toggles for state diff, error correction filter, and annotation settings"
```

---

### Task 6: Full build and test verification

**Step 1: Run all Rust tests**

Run: `cd skill-generator && cargo test`
Expected: All tests pass

**Step 2: Build release**

Run: `cd skill-generator && cargo build --release`
Expected: Compiles

**Step 3: Build frontend**

Run: `cd skill-generator-ui && npm run build`
Expected: Compiles

**Step 4: Build Tauri**

Run: `cd skill-generator-ui/src-tauri && cargo build`
Expected: Compiles

**Step 5: Verify backward compat**

Ensure existing `config.toml` files without `[annotation]` section or new pipeline fields still load correctly (covered by `#[serde(default)]` attributes).
