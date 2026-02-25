# Phase 5: Tauri UI & Configuration Integration

## Summary

Expose the VLM pipeline features from Phases 1-4 (visual anchoring, state diffing, trajectory visualization, error correction filtering) as user-configurable settings through the full stack: Rust config → Tauri commands → React UI.

## Architecture

```
config.toml  →  Config (Rust)  →  UiConfig (Tauri)  →  UiConfig (React)
     ↓                ↓                  ↓                     ↓
  [pipeline]     PipelineConfig     get_config()        Toggle checkboxes
  [annotation]   AnnotationConfig   save_config()       Advanced sliders
```

## Changes by Layer

### Layer 1: Rust Config (`skill-generator/src/app/config.rs`)

Add 2 fields to `PipelineConfig`:
- `use_state_diff: bool` (default: true) — Phase 2 before/after screenshot diffing
- `use_error_correction_filter: bool` (default: true) — Phase 4 undo/cancel filtering

Add new `AnnotationConfig` section to `Config`:
```rust
pub struct AnnotationConfig {
    pub dot_radius: u32,                    // 12
    pub dot_color: [u8; 3],                 // [255, 40, 40]
    pub trajectory_ballistic_color: [u8; 3], // [40, 220, 40]
    pub trajectory_searching_color: [u8; 3], // [255, 160, 40]
    pub trajectory_thickness: u32,           // 2
    pub crop_size: u32,                      // 512
}
```

### Layer 2: Tauri Commands (`skill-generator-ui/src-tauri/src/lib.rs`)

- Add matching fields to `UiConfig` struct
- Wire through `get_config()` / `save_config()`
- Wire `use_state_diff` and `use_error_correction_filter` into GeneratorConfig in `generate_skill()`
- Add `state_diff_enriched_count` and `error_correction_count` to `PipelineStatsInfo`

### Layer 3: React UI (`skill-generator-ui/src/App.tsx`)

- Extend `UiConfig` TypeScript interface
- Add 2 pipeline toggles: "State Diff Analysis", "Error Correction Filter"
- Add collapsible "Visual Annotation" advanced section:
  - Dot radius slider (4-24)
  - Color hex inputs for dot, ballistic trajectory, searching trajectory
  - Trajectory thickness slider (1-5)
  - Crop size input (256-1024)
- Display new pipeline stats in generation results

### Layer 4: Generator Wiring

- Gate state diff enrichment on `use_state_diff` in generator.rs
- Gate error correction filtering on `use_error_correction_filter` in generator.rs
- Pass AnnotationConfig from config through to screenshot analysis pipeline

## Non-Goals

- No new crates or dependencies
- No new Tauri commands (reuse existing get_config/save_config)
- No schema migration — new fields use `#[serde(default)]` for backward compat
