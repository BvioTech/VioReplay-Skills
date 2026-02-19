# Replay Skill

**Turn screen recordings into agent skills.**

Record yourself performing any workflow on your computer, and Replay Skill will automatically generate a reusable, parameterized [Claude Code SKILL.md](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#create-custom-slash-commands) file that an AI agent can replay on demand.

[![macOS](https://img.shields.io/badge/macOS-12.3+-blue.svg)](https://www.apple.com/macos/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

---

## How It Works

```
You record a workflow    →    Replay Skill watches    →    Out comes a SKILL.md
(click, type, scroll)         every event + UI context       ready for Claude Code
```

1. **Record** — Hit the record button and perform your workflow normally. Replay Skill captures every mouse movement, click, keystroke, and scroll event with microsecond precision.
2. **Enrich** — Each event is automatically enriched with semantic context: what button you clicked, what text field you typed into, what menu you selected — pulled from the macOS Accessibility API and, when needed, OCR vision fallback.
3. **Analyze** — A multi-stage pipeline segments your actions into cognitive chunks using GOMS modeling, infers intent using Fitts' Law kinematic analysis, and extracts variables (emails, dates, URLs) for parameterization.
4. **Generate** — A verified `SKILL.md` file is produced with step-by-step instructions, fallback UI selectors, formal postconditions, and `{{mustache}}` template variables — ready to drop into `~/.claude/skills/`.

---

## Download

Go to the [Releases](https://github.com/aezizhu/Replay-Skill/releases) page and download the installer for your platform:

| Platform | File | Notes |
|----------|------|-------|
| **macOS (Apple Silicon)** | `Replay-Skill_*_aarch64.dmg` | M1/M2/M3/M4 Macs |
| **macOS (Intel)** | `Replay-Skill_*_x64.dmg` | Intel Macs |

> **Note:** The core recording engine currently uses macOS-specific APIs (Quartz Event Tap, Accessibility API, mach_absolute_time). Windows and Linux builds are planned — contributions welcome! See [Cross-Platform Roadmap](#cross-platform-roadmap) below.

### macOS Installation

1. Download the `.dmg` file for your architecture from [Releases](https://github.com/aezizhu/Replay-Skill/releases).
2. Open the `.dmg` and drag **Replay Skill** into your Applications folder.
3. On first launch, macOS may show a security warning. Go to **System Settings → Privacy & Security** and click **Open Anyway**.
4. Grant the required permissions when prompted (see [Permissions](#permissions)).

---

## Permissions (macOS)

Replay Skill needs two macOS permissions to function:

| Permission | Required? | Why | How to Grant |
|------------|-----------|-----|--------------|
| **Accessibility** | Yes | Captures keyboard/mouse events and reads UI element metadata | System Settings → Privacy & Security → Accessibility → Add Replay Skill |
| **Screen Recording** | Optional | Enables OCR-based vision fallback for apps with opaque UIs (e.g., Electron apps) | System Settings → Privacy & Security → Screen Recording → Add Replay Skill |

After granting permissions, restart the app.

---

## Quick Start

### Using the Desktop App (Recommended)

1. Launch **Replay Skill** from Applications.
2. Enter an **Anthropic API key** (optional — enables AI-powered skill naming and variable inference).
3. Type a **goal** describing what you're about to do (e.g., "Create a new invoice for a customer").
4. Click **Record** and perform your workflow.
5. Click **Stop** when done.
6. Click **Generate Skill** — a `SKILL.md` file is created and saved to `~/.claude/skills/`.
7. Use the skill in Claude Code: `/skill-name`

### Using the CLI

```bash
# Build from source
cd skill-generator
cargo build --release

# Initialize configuration
./target/release/skill-gen init

# Record a 60-second workflow
./target/release/skill-gen record --duration 60 --output my_workflow --goal "Create a new document"

# Generate SKILL.md from the recording
./target/release/skill-gen generate --input ~/.skill_generator/recordings/my_workflow.json

# Validate the generated skill
./target/release/skill-gen validate ~/.claude/skills/my_workflow/SKILL.md
```

---

## Features

### Zero-Lag Event Capture
- Lock-free SPSC ring buffer — event capture adds <1ms latency to your input
- Microsecond-precision timestamps via `mach_absolute_time()`
- Sustained 1000 Hz capture rate with zero event drops

### Semantic Understanding
- **Accessibility API** integration reads UI element roles, labels, identifiers, and values
- **Vision fallback** with OCR for opaque UIs (Electron, Chromium, games)
- **LLM-based context reconstruction** fills gaps when both fail

### Intelligent Chunking
- **GOMS cognitive model** detects natural task boundaries by analyzing hesitation patterns
- **Fitts' Law kinematic analysis** classifies mouse movements as ballistic, searching, or corrective
- **Action clustering** merges low-level events (keystrokes → "Type", click dropdown + click option → "Select")

### Smart Variable Extraction
- Automatically detects emails, dates, URLs, phone numbers, and other user-specific data
- Replaces hardcoded values with `{{mustache}}` template variables
- Goal-aware matching — variables mentioned in your goal description are prioritized

### Formal Verification
- Every generated step includes Hoare triple postconditions (`{P} C {Q}`)
- State assertions with configurable timeouts
- Fallback selector chains ranked by stability (AX Identifier → Text Content → CSS Selector → Position)

### Configurable Pipeline
The desktop app exposes all pipeline parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| RDP Epsilon | 3.0 px | Trajectory simplification threshold |
| Hesitation Threshold | 0.7 | GOMS mental operator detection sensitivity |
| Min Pause | 300 ms | Minimum pause to trigger a chunk boundary |
| Model | Claude Opus 4.6 | LLM for semantic synthesis and skill inference |
| Vision Model | Claude Sonnet 4.5 | LLM for screenshot analysis |
| Temperature | 0.3 | Lower = more deterministic output |

---

## Building from Source

### Prerequisites

- macOS 12.3+ (Monterey or later)
- [Rust](https://rustup.rs/) 1.70+
- [Node.js](https://nodejs.org/) 20+
- Xcode Command Line Tools (`xcode-select --install`)

### Build the Desktop App

```bash
git clone https://github.com/aezizhu/Replay-Skill.git
cd Replay-Skill/skill-generator-ui

# Install frontend dependencies
npm install

# Build the Tauri app (produces .dmg in src-tauri/target/release/bundle/)
npx tauri build
```

### Build the CLI Only

```bash
cd Replay-Skill/skill-generator
cargo build --release
# Binary: target/release/skill-gen
```

### Run in Development Mode

```bash
cd Replay-Skill/skill-generator-ui
npm install
npx tauri dev
```

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  CGEventTap │───▶│ Ring Buffer │───▶│  Semantic   │───▶│  Recording  │
│  (capture)  │    │ (lock-free) │    │  Enrichment │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  SKILL.md   │◀───│  Codegen &  │◀───│  Variable   │◀───│   Intent    │
│   Output    │    │ Validation  │    │ Extraction  │    │  Inference  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Module Overview

| Module | Description |
|--------|-------------|
| `capture/` | Lock-free event capture via Quartz Event Tap + ring buffer |
| `time/` | High-precision timing via `mach_absolute_time` |
| `analysis/` | RDP trajectory simplification, kinematic segmentation, intent inference |
| `chunking/` | GOMS-based cognitive chunking, context stack tracking, action clustering |
| `semantic/` | Accessibility API integration, vision OCR fallback, LLM context reconstruction |
| `synthesis/` | Variable extraction, selector ranking, LLM semantic synthesis |
| `codegen/` | Multi-pass SKILL.md compiler, markdown builder, validation |
| `verification/` | Postcondition extraction, Hoare triple generation |
| `workflow/` | High-level recording and skill generation orchestration |
| `app/` | CLI interface and configuration management |

For a deep technical reference, see [docs/ARCHITECTURE.md](skill-generator/docs/ARCHITECTURE.md).

---

## Generated SKILL.md Example

```yaml
---
name: create_invoice
description: Create a new invoice for a customer
context: fork
allowed-tools: Read, Bash
argument-hint: [customer_id] [date]
---

## Variables

- `{{customer_id}}`: Customer identifier (type: string)
- `{{date}}`: Invoice date (type: date)

## Steps

### Step 1: Click on "New Invoice"

**Target**: `new-invoice-btn` (type: AxIdentifier)
**Fallback selectors**:
  - `New Invoice` (TextContent)
  - `245,120` (RelativePosition)

**Verification**:
  type: state_assertion
  timeout_ms: 5000
  conditions:
    - window.title contains 'New Invoice'
```

---

## Configuration

Configuration is stored in `~/.skill_generator/config.toml`:

```toml
[capture]
ring_buffer_size = 8192      # Must be power of 2
sampling_rate_hz = 0         # Screenshot sampling (0 = disabled)
vision_fallback = true       # Enable OCR for opaque UIs

[analysis]
rdp_epsilon_px = 3.0         # Trajectory simplification threshold
hesitation_threshold = 0.7   # GOMS mental operator sensitivity
min_pause_ms = 300           # Minimum pause for chunk boundary

[codegen]
model = "claude-opus-4-6"    # LLM model for synthesis
temperature = 0.3            # Lower = more deterministic
include_screenshots = false  # Include screenshots in skill output
```

---

## Cross-Platform Roadmap

The recording engine currently relies on macOS-specific APIs. Here's what's needed for each platform:

| Component | macOS (Done) | Windows (Planned) | Linux (Planned) |
|-----------|-------------|-------------------|-----------------|
| Event Capture | Quartz Event Tap | Windows Hooks API (`SetWindowsHookEx`) | libinput / XRecord |
| UI Semantics | Accessibility API (AX) | UI Automation (UIA) | AT-SPI2 |
| Timing | `mach_absolute_time` | `QueryPerformanceCounter` | `clock_gettime(CLOCK_MONOTONIC)` |
| OCR Fallback | Apple Vision | Windows.Media.Ocr | Tesseract |
| Screenshots | Core Graphics | Desktop Duplication API | PipeWire / X11 |

The Tauri UI layer and the analysis/codegen pipeline are already cross-platform. Contributions to implement platform-specific capture backends are very welcome!

---

## Troubleshooting

### "Accessibility permissions not granted"

1. Open **System Settings → Privacy & Security → Accessibility**
2. Click the lock icon and authenticate
3. Add Replay Skill (or Terminal, if running the CLI)
4. Restart the application

### "Event tap creation failed"

This usually means accessibility permissions haven't been granted. Follow the steps above.

### "Vision fallback timeout"

Enable Screen Recording permission: **System Settings → Privacy & Security → Screen Recording** → add the app.

### High CPU usage during recording

- Increase `ring_buffer_size` in config if events are being dropped
- Reduce `sampling_rate_hz` for screenshots
- Close other apps that heavily use the accessibility API

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run `cargo fmt` and `cargo clippy` before committing
4. Add tests for new functionality
5. Push and open a Pull Request

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Tauri](https://tauri.app/) — Cross-platform desktop app framework
- [rtrb](https://github.com/mgeier/rtrb) — Real-time safe SPSC ring buffer
- Apple Core Graphics and Accessibility frameworks
- [Claude Code](https://claude.ai/claude-code) Agent Skills specification
