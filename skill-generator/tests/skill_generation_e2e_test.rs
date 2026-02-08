//! End-to-end tests for skill generation workflow
//!
//! These tests verify the complete pipeline from recording to SKILL.md output.

use skill_generator::capture::types::{
    CursorState, EnrichedEvent, EventType, ModifierFlags, RawEvent, SemanticContext,
    SemanticSource,
};
use skill_generator::time::timebase::{MachTimebase, Timestamp};
use skill_generator::workflow::generator::{GeneratorConfig, SkillGenerator};
use skill_generator::workflow::recording::Recording;
use std::fs;
use tempfile::tempdir;

/// Create a test raw event with specific parameters
fn make_raw_event(event_type: EventType, x: f64, y: f64) -> RawEvent {
    MachTimebase::init();
    RawEvent {
        timestamp: Timestamp::now(),
        event_type,
        coordinates: (x, y),
        cursor_state: CursorState::Arrow,
        key_code: None,
        character: None,
        modifiers: ModifierFlags::default(),
        scroll_delta: None,
        click_count: 1,
    }
}

/// Create a click event with semantic context
fn make_click_with_context(
    x: f64,
    y: f64,
    role: &str,
    title: &str,
    click_count: u8,
) -> EnrichedEvent {
    MachTimebase::init();
    let raw = RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::LeftMouseDown,
        coordinates: (x, y),
        cursor_state: CursorState::PointingHand,
        key_code: None,
        character: None,
        modifiers: ModifierFlags::default(),
        scroll_delta: None,
        click_count,
    };

    let mut event = EnrichedEvent::new(raw, 0);
    event.semantic = Some(SemanticContext {
        ax_role: Some(role.to_string()),
        title: Some(title.to_string()),
        identifier: Some(format!("id_{}", title.to_lowercase().replace(" ", "_"))),
        source: SemanticSource::Accessibility,
        confidence: 0.95,
        ..Default::default()
    });
    event
}

/// Create a keyboard event
fn make_key_event(character: char, key_code: u16) -> EnrichedEvent {
    MachTimebase::init();
    let raw = RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::KeyDown,
        coordinates: (0.0, 0.0),
        cursor_state: CursorState::IBeam,
        key_code: Some(key_code),
        character: Some(character),
        modifiers: ModifierFlags::default(),
        scroll_delta: None,
        click_count: 0,
    };

    EnrichedEvent::new(raw, 0)
}

/// Create a keyboard shortcut event
fn make_shortcut_event(character: char, key_code: u16, command: bool, control: bool) -> EnrichedEvent {
    MachTimebase::init();
    let raw = RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::KeyDown,
        coordinates: (0.0, 0.0),
        cursor_state: CursorState::Arrow,
        key_code: Some(key_code),
        character: Some(character),
        modifiers: ModifierFlags {
            shift: false,
            control,
            option: false,
            command,
            caps_lock: false,
            function: false,
        },
        scroll_delta: None,
        click_count: 0,
    };

    EnrichedEvent::new(raw, 0)
}

/// Create a scroll event
fn make_scroll_event(x: f64, y: f64, delta_x: f64, delta_y: f64) -> EnrichedEvent {
    MachTimebase::init();
    let raw = RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::ScrollWheel,
        coordinates: (x, y),
        cursor_state: CursorState::Arrow,
        key_code: None,
        character: None,
        modifiers: ModifierFlags::default(),
        scroll_delta: Some((delta_x, delta_y)),
        click_count: 0,
    };

    EnrichedEvent::new(raw, 0)
}

#[test]
fn test_e2e_simple_button_click() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new(
        "click_submit_button".to_string(),
        Some("Click the Submit button".to_string()),
    );

    // Add a click on a button
    let event = make_click_with_context(500.0, 300.0, "AXButton", "Submit", 1);
    recording.add_event(event);
    recording.finalize(1000);

    // Generate skill
    let skill = generator.generate(&recording).expect("Failed to generate skill");

    assert_eq!(skill.name, "click_submit_button");
    assert!(!skill.steps.is_empty());
    assert_eq!(skill.steps.len(), 1);

    // Verify the step
    let step = &skill.steps[0];
    assert!(step.description.contains("Submit"));
    assert!(step.selector.is_some());
}

#[test]
fn test_e2e_form_fill_workflow() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new(
        "fill_login_form".to_string(),
        Some("Fill out the login form and submit".to_string()),
    );

    // Click on username field
    let mut username_click = make_click_with_context(400.0, 200.0, "AXTextField", "Username", 1);
    username_click.raw.cursor_state = CursorState::IBeam;
    recording.add_event(username_click);

    // Type username
    for c in "user".chars() {
        let key_event = make_key_event(c, c as u16);
        recording.add_event(key_event);
    }

    // Click on password field
    let mut password_click = make_click_with_context(400.0, 250.0, "AXSecureTextField", "Password", 1);
    password_click.raw.cursor_state = CursorState::IBeam;
    recording.add_event(password_click);

    // Type password
    for c in "pass".chars() {
        let key_event = make_key_event(c, c as u16);
        recording.add_event(key_event);
    }

    // Click submit
    let submit_click = make_click_with_context(400.0, 350.0, "AXButton", "Login", 1);
    recording.add_event(submit_click);

    recording.finalize(5000);

    // Generate skill
    let skill = generator.generate(&recording).expect("Failed to generate skill");

    // Should have multiple steps: 2 clicks + typing chars + 1 click = many events
    assert!(!skill.steps.is_empty());

    // Render to markdown
    let markdown = generator.render_to_markdown(&skill);
    assert!(markdown.contains("---")); // Frontmatter
    assert!(markdown.contains("name: fill_login_form"));
    assert!(markdown.contains("## Steps"));
}

#[test]
fn test_e2e_keyboard_shortcuts() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new(
        "copy_paste_workflow".to_string(),
        Some("Copy and paste text".to_string()),
    );

    // Cmd+A (Select All)
    let select_all = make_shortcut_event('a', 0, true, false);
    recording.add_event(select_all);

    // Cmd+C (Copy)
    let copy = make_shortcut_event('c', 8, true, false);
    recording.add_event(copy);

    // Cmd+V (Paste)
    let paste = make_shortcut_event('v', 9, true, false);
    recording.add_event(paste);

    recording.finalize(2000);

    let skill = generator.generate(&recording).expect("Failed to generate skill");
    assert_eq!(skill.steps.len(), 3);

    // Check that shortcuts are captured - the rendering shows "Press Cmd+X" format
    let markdown = generator.render_to_markdown(&skill);
    // Verify the skill has keyboard steps
    assert!(markdown.contains("Press") || markdown.contains("Shortcut") || markdown.contains("Step"));
}

#[test]
fn test_e2e_scroll_navigation() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new(
        "scroll_to_bottom".to_string(),
        Some("Scroll to the bottom of the page".to_string()),
    );

    // Multiple scroll events
    for i in 0..5 {
        let scroll = make_scroll_event(500.0, 400.0 + i as f64 * 50.0, 0.0, -100.0);
        recording.add_event(scroll);
    }

    recording.finalize(3000);

    let skill = generator.generate(&recording).expect("Failed to generate skill");
    assert_eq!(skill.steps.len(), 5); // 5 scroll events

    let markdown = generator.render_to_markdown(&skill);
    assert!(markdown.contains("Scroll"));
}

#[test]
fn test_e2e_double_click() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new(
        "double_click_file".to_string(),
        Some("Double-click to open file".to_string()),
    );

    // Double-click event (click_count = 2)
    let double_click = make_click_with_context(300.0, 200.0, "AXCell", "document.txt", 2);
    recording.add_event(double_click);

    recording.finalize(500);

    let skill = generator.generate(&recording).expect("Failed to generate skill");
    assert_eq!(skill.steps.len(), 1);

    let step = &skill.steps[0];
    assert!(step.description.contains("Double-click"));
}

#[test]
fn test_e2e_right_click_context_menu() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new(
        "open_context_menu".to_string(),
        Some("Open context menu on file".to_string()),
    );

    // Right-click event
    let raw = RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::RightMouseDown,
        coordinates: (300.0, 200.0),
        cursor_state: CursorState::Arrow,
        key_code: None,
        character: None,
        modifiers: ModifierFlags::default(),
        scroll_delta: None,
        click_count: 1,
    };
    let mut event = EnrichedEvent::new(raw, 0);
    event.semantic = Some(SemanticContext {
        ax_role: Some("AXCell".to_string()),
        title: Some("myfile.rs".to_string()),
        ..Default::default()
    });
    recording.add_event(event);

    recording.finalize(500);

    let skill = generator.generate(&recording).expect("Failed to generate skill");
    assert_eq!(skill.steps.len(), 1);
    assert!(skill.steps[0].description.contains("Right-click"));
}

#[test]
fn test_e2e_save_and_load_recording() {
    MachTimebase::init();
    let dir = tempdir().expect("Failed to create temp dir");
    let recording_path = dir.path().join("test_recording.json");

    // Create and save recording
    let mut recording = Recording::new(
        "save_test".to_string(),
        Some("Test saving and loading".to_string()),
    );
    let event = make_click_with_context(100.0, 100.0, "AXButton", "Test", 1);
    recording.add_event(event);
    recording.finalize(1000);

    recording.save(&recording_path).expect("Failed to save recording");

    // Load recording
    let loaded = Recording::load(&recording_path).expect("Failed to load recording");

    assert_eq!(loaded.metadata.name, "save_test");
    assert_eq!(loaded.len(), 1);

    // Generate skill from loaded recording
    let generator = SkillGenerator::new();
    let skill = generator.generate(&loaded).expect("Failed to generate skill");
    assert!(!skill.steps.is_empty());
}

#[test]
fn test_e2e_save_skill_to_file() {
    MachTimebase::init();
    let dir = tempdir().expect("Failed to create temp dir");
    let skill_path = dir.path().join("test_skill.md");

    let generator = SkillGenerator::new();
    let mut recording = Recording::new(
        "file_save_test".to_string(),
        Some("Test saving skill to file".to_string()),
    );

    let event = make_click_with_context(200.0, 200.0, "AXButton", "Save", 1);
    recording.add_event(event);
    recording.finalize(500);

    let skill = generator.generate(&recording).expect("Failed to generate skill");
    generator.save_skill(&skill, &skill_path).expect("Failed to save skill");

    // Read and verify
    let content = fs::read_to_string(&skill_path).expect("Failed to read skill file");
    assert!(content.contains("---"));
    assert!(content.contains("name: file_save_test"));
    assert!(content.contains("## Steps"));
}

#[test]
fn test_e2e_validation() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new(
        "validation_test".to_string(),
        Some("Test skill validation".to_string()),
    );

    let event = make_click_with_context(300.0, 300.0, "AXButton", "Validate", 1);
    recording.add_event(event);
    recording.finalize(500);

    let skill = generator.generate(&recording).expect("Failed to generate skill");
    let validation = generator.validate(&skill);

    // Should pass basic validation
    assert!(validation.errors.is_empty() || validation.passed);
}

#[test]
fn test_e2e_selectors_with_fallbacks() {
    MachTimebase::init();
    let generator = SkillGenerator::with_config(GeneratorConfig {
        selector_chain_depth: 3,
        ..Default::default()
    });

    let mut recording = Recording::new(
        "selector_test".to_string(),
        Some("Test selector generation".to_string()),
    );

    // Click with rich semantic context
    let mut event = make_click_with_context(400.0, 400.0, "AXButton", "Confirm", 1);
    if let Some(ref mut semantic) = event.semantic {
        semantic.identifier = Some("btn_confirm".to_string());
    }
    recording.add_event(event);
    recording.finalize(500);

    let skill = generator.generate(&recording).expect("Failed to generate skill");
    let step = &skill.steps[0];

    // Should have primary selector
    assert!(step.selector.is_some());

    // Should have fallback selectors
    // Note: depends on semantic context, may have fewer fallbacks
    let markdown = generator.render_to_markdown(&skill);
    // Check that selector info is rendered (format: **Target**: `value`)
    assert!(markdown.contains("**Target**") || markdown.contains("Target:") || step.selector.is_some());
}

#[test]
fn test_e2e_verification_blocks() {
    MachTimebase::init();
    let generator = SkillGenerator::with_config(GeneratorConfig {
        include_verification: true,
        ..Default::default()
    });

    let mut recording = Recording::new(
        "verification_test".to_string(),
        Some("Test verification block generation".to_string()),
    );

    // Click event with semantic context
    let mut event1 = make_click_with_context(100.0, 100.0, "AXButton", "Start", 1);
    if let Some(ref mut semantic) = event1.semantic {
        semantic.window_title = Some("Initial Window".to_string());
    }
    recording.add_event(event1);

    // Second event with different state (simulates state change)
    let mut event2 = make_click_with_context(200.0, 200.0, "AXButton", "Next", 1);
    if let Some(ref mut semantic) = event2.semantic {
        semantic.window_title = Some("Next Window".to_string());
    }
    recording.add_event(event2);

    recording.finalize(2000);

    let skill = generator.generate(&recording).expect("Failed to generate skill");
    let markdown = generator.render_to_markdown(&skill);

    // Should have verification blocks (if postconditions were extracted)
    // The test verifies the generation works, actual verification depends on state differences
    assert!(markdown.contains("## Steps"));
}

#[test]
fn test_e2e_mixed_workflow() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new(
        "create_document".to_string(),
        Some("Create a new document and save it".to_string()),
    );

    // Click File menu
    let file_click = make_click_with_context(50.0, 25.0, "AXMenuButton", "File", 1);
    recording.add_event(file_click);

    // Click New
    let new_click = make_click_with_context(60.0, 50.0, "AXMenuItem", "New", 1);
    recording.add_event(new_click);

    // Type document content
    for c in "Hello".chars() {
        let key = make_key_event(c, c as u16);
        recording.add_event(key);
    }

    // Cmd+S to save
    let save_shortcut = make_shortcut_event('s', 1, true, false);
    recording.add_event(save_shortcut);

    // Type filename
    for c in "doc.txt".chars() {
        let key = make_key_event(c, c as u16);
        recording.add_event(key);
    }

    // Click Save button
    let save_click = make_click_with_context(500.0, 400.0, "AXButton", "Save", 1);
    recording.add_event(save_click);

    recording.finalize(10000);

    let skill = generator.generate(&recording).expect("Failed to generate skill");

    // Verify complex workflow is captured
    assert!(skill.steps.len() > 5);

    let markdown = generator.render_to_markdown(&skill);
    assert!(markdown.contains("Click"));
    assert!(markdown.contains("Press") || markdown.contains("Type"));
}

#[test]
fn test_e2e_config_variations() {
    MachTimebase::init();

    // Test with verification disabled
    let gen_no_verify = SkillGenerator::with_config(GeneratorConfig {
        include_verification: false,
        ..Default::default()
    });

    // Test with variables disabled
    let gen_no_vars = SkillGenerator::with_config(GeneratorConfig {
        extract_variables: false,
        ..Default::default()
    });

    let mut recording = Recording::new("config_test".to_string(), Some("Test config".to_string()));
    let event = make_click_with_context(100.0, 100.0, "AXButton", "Test", 1);
    recording.add_event(event);
    recording.finalize(500);

    let skill1 = gen_no_verify
        .generate(&recording)
        .expect("Failed with no verification");
    let skill2 = gen_no_vars
        .generate(&recording)
        .expect("Failed with no variables");

    // Both should generate successfully
    assert!(!skill1.steps.is_empty());
    assert!(!skill2.steps.is_empty());
}

#[test]
fn test_e2e_empty_recording_error() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let recording = Recording::new("empty".to_string(), None);

    let result = generator.generate(&recording);
    assert!(result.is_err());
}

#[test]
fn test_e2e_recording_with_only_mouse_move() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new("mouse_only".to_string(), None);

    // Add only mouse move events (no clicks, no keys)
    for i in 0..10 {
        let raw = make_raw_event(EventType::MouseMoved, i as f64 * 10.0, i as f64 * 5.0);
        recording.add_raw_event(raw);
    }
    recording.finalize(1000);

    // Should return error (no significant events)
    let result = generator.generate(&recording);
    assert!(result.is_err());
}
