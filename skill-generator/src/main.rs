//! Skill Generator - macOS Skill Generation Engine
//!
//! Transforms user interaction telemetry into Claude Code SKILL.md artifacts.

use skill_generator::app::cli::{Cli, Commands, ConfigAction};
use skill_generator::app::config::Config;
use skill_generator::capture::event_tap::EventTap;
use skill_generator::capture::ring_buffer::EventRingBuffer;
use skill_generator::time::timebase::MachTimebase;
use skill_generator::workflow::generator::SkillGenerator;
use skill_generator::workflow::recording::Recording;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    // Parse CLI arguments first so we can use --verbose to set log level
    let cli = Cli::parse_args();

    // Initialize tracing (--verbose enables debug-level output)
    let default_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level)),
        )
        .init();

    // Initialize timebase
    MachTimebase::init();

    // Load config
    let config = if let Some(path) = &cli.config {
        Config::load(path)?
    } else {
        Config::load_default()?
    };

    // Execute command
    match cli.command {
        Commands::Record {
            duration,
            output,
            goal,
        } => {
            run_record(duration, output, goal, &config)?;
        }
        Commands::Generate {
            input,
            output,
            name,
        } => {
            run_generate(&input, output, name, &config)?;
        }
        Commands::Test {
            skill,
            repeat,
            dry_run,
        } => {
            run_test(&skill, repeat, dry_run)?;
        }
        Commands::Validate { skill } => {
            run_validate(&skill)?;
        }
        Commands::List { detailed } => {
            run_list(detailed)?;
        }
        Commands::Init { force } => {
            run_init(force, &config)?;
        }
        Commands::Delete { name, force } => {
            run_delete(&name, force)?;
        }
        Commands::Config { action } => {
            run_config(action, &config)?;
        }
    }

    Ok(())
}

fn run_record(
    duration: u64,
    output: Option<String>,
    goal: Option<String>,
    config: &Config,
) -> anyhow::Result<()> {
    info!("Starting recording for {} seconds", duration);

    if let Some(g) = &goal {
        info!("Goal: {}", g);
    }

    // Create ring buffer for event capture
    let buffer = EventRingBuffer::with_capacity(config.capture.ring_buffer_size);
    let (producer, mut consumer) = buffer.split();

    // Create recording
    let output_name = output.clone().unwrap_or_else(|| {
        chrono::Local::now()
            .format("recording_%Y%m%d_%H%M%S")
            .to_string()
    });
    let mut recording = Recording::new(output_name.clone(), goal);

    // Create and start event tap
    let mut event_tap = EventTap::new()?;
    
    match event_tap.start(producer) {
        Ok(()) => {
            info!("Event tap started successfully");
        }
        Err(e) => {
            warn!("Failed to start event tap: {}", e);
            warn!("Please enable Accessibility permissions in System Preferences > Security & Privacy > Privacy > Accessibility");
            warn!("Running in demo mode (no actual event capture)");
        }
    }

    info!("Recording... Press Ctrl+C to stop");

    let start_time = std::time::Instant::now();

    // Set up Ctrl+C handler
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_flag_handler = stop_flag.clone();

    ctrlc::set_handler(move || {
        stop_flag_handler.store(true, std::sync::atomic::Ordering::SeqCst);
    })?;

    // Recording loop
    loop {
        // Check if we should stop
        if stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        // Check duration limit
        if duration > 0 && start_time.elapsed().as_secs() >= duration {
            break;
        }

        // Drain events from buffer
        let batch = consumer.pop_batch(100);
        for slot in batch {
            recording.add_raw_event(slot.event);
        }

        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Stop the event tap
    event_tap.stop();

    let elapsed = start_time.elapsed();
    recording.finalize(elapsed.as_millis() as u64);

    info!("Recording stopped after {:.1}s", elapsed.as_secs_f64());
    info!("Captured {} events", recording.len());

    // Save recording
    let recordings_dir = Cli::recordings_dir();
    std::fs::create_dir_all(&recordings_dir)?;

    let output_path = recordings_dir.join(format!("{}.json", output_name));
    recording.save(&output_path)?;
    info!("Saved recording to {:?}", output_path);

    Ok(())
}

fn run_generate(
    input: &std::path::Path,
    output: Option<std::path::PathBuf>,
    name: Option<String>,
    config: &Config,
) -> anyhow::Result<()> {
    info!("Generating skill from {:?}", input);

    // Load recording
    if !input.exists() {
        anyhow::bail!("Recording file not found: {:?}", input);
    }

    let recording = Recording::load(input)?;

    info!(
        "Loaded recording '{}' with {} events",
        recording.metadata.name,
        recording.len()
    );

    // Determine skill name
    let skill_name = name.unwrap_or_else(|| {
        recording.metadata.name.clone()
    });

    // Create skill generator with config
    let gen_config = skill_generator::workflow::generator::GeneratorConfig {
        api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
        use_llm_semantic: config.codegen.model.contains("claude"),
        ..Default::default()
    };
    let generator = SkillGenerator::with_config(gen_config);

    // Generate skill
    let skill = match generator.generate(&recording) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to generate skill: {}", e);
            anyhow::bail!("Skill generation failed: {}", e);
        }
    };

    info!("Generated skill with {} steps", skill.steps.len());

    // Validate
    let validation = generator.validate(&skill);
    if !validation.passed {
        warn!("Skill validation has issues:");
        for err in &validation.errors {
            warn!("  - {:?}: {}", err.error_type, err.message);
        }
    }

    // Output directory
    let output_dir = output.unwrap_or_else(Cli::skills_dir);
    let skill_dir = output_dir.join(&skill_name);
    std::fs::create_dir_all(&skill_dir)?;

    let skill_path = skill_dir.join("SKILL.md");

    // Render and save
    generator.save_skill(&skill, &skill_path)?;
    info!("Generated skill at {:?}", skill_path);

    // Print summary
    println!("\nSkill Generated Successfully!");
    println!("  Name: {}", skill.name);
    println!("  Steps: {}", skill.steps.len());
    println!("  Variables: {}", skill.variables.len());
    println!("  Output: {:?}", skill_path);

    Ok(())
}

fn run_test(
    skill: &std::path::Path,
    repeat: u32,
    dry_run: bool,
) -> anyhow::Result<()> {
    info!("Testing skill {:?}", skill);

    if !skill.exists() {
        anyhow::bail!("Skill file not found: {:?}", skill);
    }

    let content = std::fs::read_to_string(skill)?;

    for i in 1..=repeat {
        info!("Test run {}/{}", i, repeat);

        if dry_run {
            info!("  Dry run - validating only");
        }

        // Validate
        let validator = skill_generator::codegen::validation::SkillValidator::new();
        let result = validator.validate_markdown(&content);

        if result.passed {
            info!("  Validation: PASSED");
        } else {
            error!("  Validation: FAILED");
            for err in &result.errors {
                error!("    - {:?}: {}", err.error_type, err.message);
            }
        }
    }

    Ok(())
}

fn run_validate(skill: &std::path::Path) -> anyhow::Result<()> {
    info!("Validating {:?}", skill);

    if !skill.exists() {
        anyhow::bail!("Skill file not found: {:?}", skill);
    }

    let content = std::fs::read_to_string(skill)?;
    let validator = skill_generator::codegen::validation::SkillValidator::new();
    let result = validator.validate_markdown(&content);

    if result.passed {
        println!("Validation PASSED");
        Ok(())
    } else {
        println!("Validation FAILED:");
        for err in &result.errors {
            println!("  - [{:?}] {}", err.error_type, err.message);
            if let Some(loc) = &err.location {
                println!("    at {}", loc);
            }
        }
        anyhow::bail!("Validation failed with {} errors", result.errors.len())
    }
}

fn run_list(detailed: bool) -> anyhow::Result<()> {
    let recordings_dir = Cli::recordings_dir();

    if !recordings_dir.exists() {
        println!("No recordings found in {}", recordings_dir.display());
        println!("Start a recording with: skill-gen record");
        return Ok(());
    }

    println!("Recordings in {:?}:", recordings_dir);

    let mut entries: Vec<_> = std::fs::read_dir(&recordings_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();

    entries.sort_by_key(|e| e.path());

    for entry in &entries {
        let path = entry.path();
        let file_name = path.file_name().unwrap_or_default().to_string_lossy();

        if detailed {
            match Recording::load(&path) {
                Ok(recording) => {
                    let m = &recording.metadata;
                    let duration_secs = m.duration_ms as f64 / 1000.0;
                    let goal_str = m.goal.as_deref().unwrap_or("-");
                    println!(
                        "  {}  ({} events, {:.1}s, goal: {})",
                        file_name, m.event_count, duration_secs, goal_str
                    );
                }
                Err(_) => {
                    let fs_meta = entry.metadata()?;
                    println!("  {}  ({} bytes, failed to parse)", file_name, fs_meta.len());
                }
            }
        } else {
            println!("  {}", file_name);
        }
    }

    if entries.is_empty() {
        println!("  (none)");
        println!("Start a recording with: skill-gen record");
    }

    Ok(())
}

fn run_init(force: bool, config: &Config) -> anyhow::Result<()> {
    let config_path = Config::default_path();

    if config_path.exists() && !force {
        anyhow::bail!(
            "Config already exists at {:?}. Use --force to overwrite.",
            config_path
        );
    }

    config.save_default()?;
    println!("Created config at {:?}", config_path);
    println!("\nConfig content:\n{}", config.to_toml()?);

    // Create directories
    std::fs::create_dir_all(Cli::recordings_dir())?;
    std::fs::create_dir_all(Cli::skills_dir())?;

    println!("\nCreated directories:");
    println!("  Recordings: {:?}", Cli::recordings_dir());
    println!("  Skills: {:?}", Cli::skills_dir());

    Ok(())
}

fn run_delete(name: &str, force: bool) -> anyhow::Result<()> {
    let recordings_dir = Cli::recordings_dir();

    // Try exact filename first, then add .json extension
    let candidates = vec![
        recordings_dir.join(name),
        recordings_dir.join(format!("{}.json", name)),
    ];

    let target = candidates
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| anyhow::anyhow!("Recording '{}' not found in {:?}", name, recordings_dir))?;

    if !force {
        // Show what will be deleted
        let file_size = std::fs::metadata(&target)?.len();
        println!("Will delete: {} ({} bytes)", target.display(), file_size);
        println!("Use --force to skip this prompt, or re-run with -f");
        return Ok(());
    }

    std::fs::remove_file(&target)?;
    info!("Deleted recording: {}", target.display());
    println!("Deleted: {}", target.display());

    Ok(())
}

fn run_config(action: ConfigAction, config: &Config) -> anyhow::Result<()> {
    match action {
        ConfigAction::Show => {
            let toml_str = config.to_toml()?;
            println!("Configuration ({:?}):\n", Config::default_path());
            println!("{}", toml_str);
        }
        ConfigAction::Get { key } => {
            let toml_str = config.to_toml()?;
            // Simple key lookup in TOML output
            let value = find_toml_value(&toml_str, &key);
            match value {
                Some(v) => println!("{} = {}", key, v),
                None => {
                    anyhow::bail!("Configuration key '{}' not found", key);
                }
            }
        }
        ConfigAction::Set { key, value } => {
            let config_path = Config::default_path();
            if !config_path.exists() {
                anyhow::bail!(
                    "No config file found. Run 'skill-gen init' first."
                );
            }

            // Load, modify, and save
            let mut toml_content = std::fs::read_to_string(&config_path)?;
            if set_toml_value(&mut toml_content, &key, &value) {
                std::fs::write(&config_path, &toml_content)?;
                println!("Set {} = {}", key, value);
            } else {
                anyhow::bail!("Failed to set '{}'. Key may not exist in config.", key);
            }
        }
        ConfigAction::Reset { force } => {
            let config_path = Config::default_path();

            if config_path.exists() && !force {
                println!("Config exists at {:?}", config_path);
                println!("Use --force to reset to defaults");
                return Ok(());
            }

            let default_config = Config::default();
            default_config.save_default()?;
            println!("Configuration reset to defaults at {:?}", config_path);
        }
    }

    Ok(())
}

/// Simple TOML value lookup by dotted key
fn find_toml_value<'a>(toml_str: &'a str, key: &str) -> Option<&'a str> {
    let parts: Vec<&str> = key.split('.').collect();
    let leaf_key = parts.last()?;

    // Find the right section
    let mut in_section = parts.len() == 1; // Top-level key
    let section_name = if parts.len() > 1 { parts[0] } else { "" };

    for line in toml_str.lines() {
        let trimmed = line.trim();

        // Check for section header
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let section = &trimmed[1..trimmed.len() - 1];
            in_section = section == section_name;
            continue;
        }

        if in_section {
            if let Some(eq_pos) = trimmed.find('=') {
                let line_key = trimmed[..eq_pos].trim();
                if line_key == *leaf_key {
                    return Some(trimmed[eq_pos + 1..].trim());
                }
            }
        }
    }

    None
}

/// Simple TOML value setter by dotted key
fn set_toml_value(toml_str: &mut String, key: &str, value: &str) -> bool {
    let parts: Vec<&str> = key.split('.').collect();
    let leaf_key = parts.last().unwrap();

    let section_name = if parts.len() > 1 { parts[0] } else { "" };
    let mut in_section = parts.len() == 1;
    let mut found = false;

    let lines: Vec<String> = toml_str.lines().map(|l| l.to_string()).collect();
    let mut new_lines = Vec::with_capacity(lines.len());

    for line in &lines {
        let trimmed = line.trim();

        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let section = &trimmed[1..trimmed.len() - 1];
            in_section = section == section_name;
        }

        if in_section && !found {
            if let Some(eq_pos) = trimmed.find('=') {
                let line_key = trimmed[..eq_pos].trim();
                if line_key == *leaf_key {
                    new_lines.push(format!("{} = {}", leaf_key, value));
                    found = true;
                    continue;
                }
            }
        }

        new_lines.push(line.clone());
    }

    if found {
        *toml_str = new_lines.join("\n");
        // Ensure trailing newline
        if !toml_str.ends_with('\n') {
            toml_str.push('\n');
        }
    }

    found
}
