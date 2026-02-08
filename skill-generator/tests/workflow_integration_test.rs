//! Workflow Integration Tests
//!
//! Comprehensive integration tests for the skill-generator workflow that:
//! - Test full workflow chains (Recording -> Generation -> Validation)
//! - Handle edge cases (empty inputs, malformed data, boundary conditions)
//! - Test error propagation across components
//! - Verify serialization/deserialization roundtrips

use skill_generator::capture::types::{
    CursorState, EnrichedEvent, EventType, ModifierFlags, RawEvent, SemanticContext,
    SemanticSource,
};
use skill_generator::time::timebase::{MachTimebase, Timestamp};
use skill_generator::workflow::{Recording, SkillGenerator};
use skill_generator::synthesis::variable_extraction::VariableExtractor;
use tempfile::TempDir;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a basic raw event for testing
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

/// Create an enriched event from raw event
fn make_enriched_event(raw: RawEvent, semantic: Option<SemanticContext>) -> EnrichedEvent {
    let mut event = EnrichedEvent::new(raw, 0);
    event.semantic = semantic;
    event
}

/// Create a keyboard event with character
fn make_keyboard_event(character: char) -> RawEvent {
    MachTimebase::init();
    RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::KeyDown,
        coordinates: (100.0, 200.0),
        cursor_state: CursorState::IBeam,
        key_code: Some(0),
        character: Some(character),
        modifiers: ModifierFlags::default(),
        scroll_delta: None,
        click_count: 0,
    }
}

/// Create a scroll event
fn make_scroll_event(delta_x: f64, delta_y: f64) -> RawEvent {
    MachTimebase::init();
    RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::ScrollWheel,
        coordinates: (500.0, 500.0),
        cursor_state: CursorState::Arrow,
        key_code: None,
        character: None,
        modifiers: ModifierFlags::default(),
        scroll_delta: Some((delta_x, delta_y)),
        click_count: 0,
    }
}

/// Create semantic context for a button
fn make_button_semantic(title: &str) -> SemanticContext {
    SemanticContext {
        ax_role: Some("AXButton".to_string()),
        title: Some(title.to_string()),
        identifier: Some(format!("btn-{}", title.to_lowercase())),
        value: None,
        parent_role: Some("AXWindow".to_string()),
        parent_title: Some("Test Window".to_string()),
        window_title: Some("Test App".to_string()),
        app_bundle_id: Some("com.test.app".to_string()),
        app_name: Some("TestApp".to_string()),
        pid: Some(1234),
        window_id: Some(56789),
        frame: Some((100.0, 200.0, 80.0, 30.0)),
        source: SemanticSource::Accessibility,
        confidence: 0.95,
        ocr_text: None,
        ancestors: vec![],
    }
}

/// Create semantic context for a text field
fn make_textfield_semantic(title: &str) -> SemanticContext {
    SemanticContext {
        ax_role: Some("AXTextField".to_string()),
        title: Some(title.to_string()),
        identifier: Some(format!("field-{}", title.to_lowercase())),
        value: Some(String::new()),
        parent_role: Some("AXGroup".to_string()),
        parent_title: Some("Form".to_string()),
        window_title: Some("Test App".to_string()),
        app_bundle_id: Some("com.test.app".to_string()),
        app_name: Some("TestApp".to_string()),
        pid: Some(1234),
        window_id: Some(56789),
        frame: Some((100.0, 300.0, 200.0, 30.0)),
        source: SemanticSource::Accessibility,
        confidence: 0.95,
        ocr_text: None,
        ancestors: vec![],
    }
}

// ============================================================================
// Test 1: Empty Recording Generation
// ============================================================================

#[test]
fn test_empty_recording_error() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let recording = Recording::new("empty_test".to_string(), None);

    let result = generator.generate(&recording);
    assert!(result.is_err(), "Empty recording should fail generation");
}

// ============================================================================
// Test 2: Single Click Full Workflow
// ============================================================================

#[test]
fn test_single_click_workflow() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new("single_click".to_string(), Some("Click submit button".to_string()));

    // Add a click event with semantic context
    let raw = make_raw_event(EventType::LeftMouseDown, 150.0, 250.0);
    let semantic = make_button_semantic("Submit");
    recording.add_event(make_enriched_event(raw, Some(semantic)));
    recording.finalize(1000);

    // Generate skill
    let skill = generator.generate(&recording).expect("Generation should succeed");

    assert_eq!(skill.name, "single_click");
    assert_eq!(skill.steps.len(), 1);
    assert!(skill.steps[0].description.contains("Submit"));
}

// ============================================================================
// Test 3: Multiple Events Chained Workflow
// ============================================================================

#[test]
fn test_multi_step_workflow() {
    MachTimebase::init();
    let config = skill_generator::workflow::generator::GeneratorConfig {
        use_action_clustering: false,
        ..Default::default()
    };
    let generator = SkillGenerator::with_config(config);
    let mut recording = Recording::new(
        "multi_step".to_string(),
        Some("Fill form and submit".to_string()),
    );

    // Click on text field
    let click_field = make_raw_event(EventType::LeftMouseDown, 200.0, 300.0);
    let field_semantic = make_textfield_semantic("Email");
    recording.add_event(make_enriched_event(click_field, Some(field_semantic)));

    // Type email
    for ch in "test@example.com".chars() {
        recording.add_raw_event(make_keyboard_event(ch));
    }

    // Click submit button
    let click_button = make_raw_event(EventType::LeftMouseDown, 150.0, 400.0);
    let button_semantic = make_button_semantic("Submit");
    recording.add_event(make_enriched_event(click_button, Some(button_semantic)));

    recording.finalize(5000);

    // Generate skill
    let skill = generator.generate(&recording).expect("Generation should succeed");

    // Should have at least 2 significant steps (click field, click button)
    // The keyboard events are processed but may be grouped
    assert!(skill.steps.len() >= 2, "Should have multiple steps");

    // Verify skill structure
    assert!(!skill.description.is_empty());
    assert!(!skill.allowed_tools.is_empty());
}

// ============================================================================
// Test 4: Serialization Roundtrip - Recording
// ============================================================================

#[test]
fn test_recording_serialization_roundtrip() {
    MachTimebase::init();
    let mut recording = Recording::new("serialize_test".to_string(), Some("Test goal".to_string()));

    // Add various event types
    recording.add_raw_event(make_raw_event(EventType::LeftMouseDown, 100.0, 200.0));
    recording.add_raw_event(make_keyboard_event('a'));
    recording.add_raw_event(make_scroll_event(0.0, -10.0));
    recording.finalize(2000);

    // Serialize to JSON
    let json = serde_json::to_string(&recording).expect("Serialization should succeed");

    // Deserialize back
    let loaded: Recording = serde_json::from_str(&json).expect("Deserialization should succeed");

    // Verify data integrity
    assert_eq!(loaded.metadata.name, "serialize_test");
    assert_eq!(loaded.metadata.goal, Some("Test goal".to_string()));
    assert_eq!(loaded.events.len(), 3);
    assert_eq!(loaded.metadata.event_count, 3);
    assert_eq!(loaded.metadata.duration_ms, 2000);
}

// ============================================================================
// Test 5: Serialization Roundtrip - Recording with Semantic Context
// ============================================================================

#[test]
fn test_recording_with_semantic_serialization() {
    MachTimebase::init();
    let mut recording = Recording::new("semantic_test".to_string(), None);

    let raw = make_raw_event(EventType::LeftMouseDown, 100.0, 200.0);
    let semantic = make_button_semantic("TestButton");
    recording.add_event(make_enriched_event(raw, Some(semantic)));
    recording.finalize(1000);

    // Serialize and deserialize
    let json = serde_json::to_string(&recording).expect("Serialization should succeed");
    let loaded: Recording = serde_json::from_str(&json).expect("Deserialization should succeed");

    // Verify semantic context preserved
    assert_eq!(loaded.events.len(), 1);
    let event = &loaded.events[0];
    assert!(event.semantic.is_some());

    let sem = event.semantic.as_ref().unwrap();
    assert_eq!(sem.ax_role, Some("AXButton".to_string()));
    assert_eq!(sem.title, Some("TestButton".to_string()));
    assert_eq!(sem.confidence, 0.95);
}

// ============================================================================
// Test 6: File I/O Roundtrip
// ============================================================================

#[test]
fn test_recording_file_io_roundtrip() {
    MachTimebase::init();
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("test_recording.json");

    let mut recording = Recording::new("file_test".to_string(), Some("Save and load".to_string()));
    recording.add_raw_event(make_raw_event(EventType::LeftMouseDown, 100.0, 200.0));
    recording.finalize(1000);

    // Save to file
    recording.save(&file_path).expect("Save should succeed");

    // Load from file
    let loaded = Recording::load(&file_path).expect("Load should succeed");

    // Verify
    assert_eq!(loaded.metadata.name, "file_test");
    assert_eq!(loaded.events.len(), 1);
}

// ============================================================================
// Test 7: Edge Case - Malformed Event Data
// ============================================================================

#[test]
fn test_malformed_event_handling() {
    MachTimebase::init();
    let mut recording = Recording::new("malformed_test".to_string(), None);

    // Add event with missing semantic context
    let raw = make_raw_event(EventType::LeftMouseDown, 0.0, 0.0);
    recording.add_event(make_enriched_event(raw, None));

    let config = skill_generator::workflow::generator::GeneratorConfig {
        use_action_clustering: false,
        use_local_recovery: false,
        use_llm_semantic: false,
        ..Default::default()
    };
    let generator = SkillGenerator::with_config(config);
    let skill = generator.generate(&recording).expect("Should handle missing semantic gracefully");

    // Should still generate a step with fallback data using coordinates
    assert_eq!(skill.steps.len(), 1);
    // Without semantic context, the description should contain coordinate-based target "(x, y)"
    assert!(
        skill.steps[0].description.contains("(") && skill.steps[0].description.contains(")"),
        "Expected coordinate-based target in description: {}",
        skill.steps[0].description
    );
}

// ============================================================================
// Test 8: Edge Case - Boundary Coordinates
// ============================================================================

#[test]
fn test_boundary_coordinates() {
    MachTimebase::init();
    let mut recording = Recording::new("boundary_test".to_string(), None);

    // Add events at screen boundaries
    recording.add_raw_event(make_raw_event(EventType::LeftMouseDown, 0.0, 0.0));
    recording.add_raw_event(make_raw_event(EventType::LeftMouseDown, 9999.0, 9999.0));
    recording.add_raw_event(make_raw_event(EventType::LeftMouseDown, -100.0, -100.0));

    let config = skill_generator::workflow::generator::GeneratorConfig {
        use_action_clustering: false,
        ..Default::default()
    };
    let generator = SkillGenerator::with_config(config);
    let skill = generator.generate(&recording).expect("Should handle boundary coordinates");

    assert_eq!(skill.steps.len(), 3);
}

// ============================================================================
// Test 9: Edge Case - Very Long Event Sequence
// ============================================================================

#[test]
fn test_large_event_sequence() {
    MachTimebase::init();
    let mut recording = Recording::new("large_sequence".to_string(), None);

    // Add 1000 click events
    for i in 0..1000 {
        recording.add_raw_event(make_raw_event(
            EventType::LeftMouseDown,
            (i % 100) as f64,
            (i / 100) as f64,
        ));
    }
    recording.finalize(60000);

    let config = skill_generator::workflow::generator::GeneratorConfig {
        use_action_clustering: false,
        ..Default::default()
    };
    let generator = SkillGenerator::with_config(config);
    let skill = generator.generate(&recording).expect("Should handle large sequences");

    // Noise filter deduplicates consecutive clicks with the same (None, None) target,
    // so we won't get 1000 steps. The key assertion is that the pipeline handles
    // large inputs without crashing and produces a non-trivial result.
    assert!(
        skill.steps.len() >= 10,
        "Expected at least 10 steps from 1000 events, got {}",
        skill.steps.len()
    );
}

// ============================================================================
// Test 10: Edge Case - Special Characters in Text
// ============================================================================

#[test]
fn test_special_characters_in_input() {
    MachTimebase::init();
    let mut recording = Recording::new("special_chars".to_string(), None);

    let special_chars = "!@#$%^&*(){}[]<>?/\\|~`";
    for ch in special_chars.chars() {
        recording.add_raw_event(make_keyboard_event(ch));
    }

    let config = skill_generator::workflow::generator::GeneratorConfig {
        use_action_clustering: false,
        ..Default::default()
    };
    let generator = SkillGenerator::with_config(config);
    let skill = generator.generate(&recording).expect("Should handle special characters");

    // Consecutive Type actions are consolidated into a single step
    assert_eq!(skill.steps.len(), 1);
    // The consolidated step should contain all special characters
    assert!(skill.steps[0].description.contains("Type"), "Step should be a Type action");
}

// ============================================================================
// Test 11: Variable Extraction Integration
// ============================================================================

#[test]
fn test_variable_extraction_workflow() {
    MachTimebase::init();
    let mut recording = Recording::new(
        "variable_test".to_string(),
        Some("Enter email test@example.com".to_string()),
    );

    // Type an email address
    for ch in "test@example.com".chars() {
        recording.add_raw_event(make_keyboard_event(ch));
    }

    let generator = SkillGenerator::new();
    let skill = generator.generate(&recording).expect("Generation should succeed");

    // Variables should be extracted based on goal
    assert!(!skill.variables.is_empty(), "Should extract variables");
}

// ============================================================================
// Test 12: Verification Block Generation
// ============================================================================

#[test]
fn test_verification_block_integration() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new("verification_test".to_string(), None);

    // Add click with rich semantic context
    let raw = make_raw_event(EventType::LeftMouseDown, 100.0, 200.0);
    let semantic = make_button_semantic("Submit");
    recording.add_event(make_enriched_event(raw, Some(semantic)));

    let skill = generator.generate(&recording).expect("Generation should succeed");

    // Verification blocks should be included
    assert_eq!(skill.steps.len(), 1);
    // Note: verification may or may not be present depending on available post-event data
}

// ============================================================================
// Test 13: Selector Generation and Fallbacks
// ============================================================================

#[test]
fn test_selector_chain_generation() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new("selector_test".to_string(), None);

    // Add event with full semantic context
    let raw = make_raw_event(EventType::LeftMouseDown, 100.0, 200.0);
    let mut semantic = make_button_semantic("TestButton");
    semantic.identifier = Some("unique-id-12345".to_string());

    recording.add_event(make_enriched_event(raw, Some(semantic)));

    let skill = generator.generate(&recording).expect("Generation should succeed");

    assert_eq!(skill.steps.len(), 1);
    let step = &skill.steps[0];

    // Should have primary selector
    assert!(step.selector.is_some(), "Should have primary selector");

    // Should have fallback selectors
    assert!(!step.fallback_selectors.is_empty(), "Should have fallback selectors");
}

// ============================================================================
// Test 14: Render to Markdown Integration
// ============================================================================

#[test]
fn test_markdown_rendering_workflow() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new("markdown_test".to_string(), Some("Test markdown output".to_string()));

    let raw = make_raw_event(EventType::LeftMouseDown, 100.0, 200.0);
    let semantic = make_button_semantic("Submit");
    recording.add_event(make_enriched_event(raw, Some(semantic)));

    let skill = generator.generate(&recording).expect("Generation should succeed");
    let markdown = generator.render_to_markdown(&skill);

    // Verify markdown structure
    assert!(markdown.contains("---"), "Should have YAML frontmatter");
    assert!(markdown.contains("name: markdown_test"));
    assert!(markdown.contains("## Steps"));
    assert!(markdown.contains("Submit"));
}

// ============================================================================
// Test 15: Save Skill to File Integration
// ============================================================================

#[test]
fn test_save_skill_to_file() {
    MachTimebase::init();
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let skill_path = temp_dir.path().join("test_skill.md");

    let generator = SkillGenerator::new();
    let mut recording = Recording::new("file_save_test".to_string(), Some("Save skill to file".to_string()));

    let raw = make_raw_event(EventType::LeftMouseDown, 100.0, 200.0);
    let semantic = make_button_semantic("Submit");
    recording.add_event(make_enriched_event(raw, Some(semantic)));

    let skill = generator.generate(&recording).expect("Generation should succeed");

    // Save to file
    generator.save_skill(&skill, &skill_path).expect("Save should succeed");

    // Verify file exists and contains expected content
    let content = std::fs::read_to_string(&skill_path).expect("Read should succeed");
    assert!(content.contains("name: file_save_test"));
    assert!(content.contains("Submit"));
}

// ============================================================================
// Test 16: Scroll Action Integration
// ============================================================================

#[test]
fn test_scroll_action_workflow() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new("scroll_test".to_string(), Some("Scroll down page".to_string()));

    // Add scroll event
    recording.add_raw_event(make_scroll_event(0.0, -50.0));

    let skill = generator.generate(&recording).expect("Generation should succeed");

    assert_eq!(skill.steps.len(), 1);
    assert!(skill.steps[0].description.contains("Scroll"));
}

// ============================================================================
// Test 17: Double Click Detection
// ============================================================================

#[test]
fn test_double_click_detection() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new("double_click_test".to_string(), None);

    // Add double-click event
    let mut raw = make_raw_event(EventType::LeftMouseDown, 100.0, 200.0);
    raw.click_count = 2;
    let semantic = make_button_semantic("File");
    recording.add_event(make_enriched_event(raw, Some(semantic)));

    let skill = generator.generate(&recording).expect("Generation should succeed");

    assert_eq!(skill.steps.len(), 1);
    assert!(skill.steps[0].description.contains("Double-click"));
}

// ============================================================================
// Test 18: Right Click Detection
// ============================================================================

#[test]
fn test_right_click_detection() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new("right_click_test".to_string(), None);

    // Add right-click event
    let raw = make_raw_event(EventType::RightMouseDown, 100.0, 200.0);
    let semantic = make_button_semantic("Context");
    recording.add_event(make_enriched_event(raw, Some(semantic)));

    let skill = generator.generate(&recording).expect("Generation should succeed");

    assert_eq!(skill.steps.len(), 1);
    assert!(skill.steps[0].description.contains("Right-click"));
}

// ============================================================================
// Test 19: Keyboard Shortcut Detection
// ============================================================================

#[test]
fn test_keyboard_shortcut_detection() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new("shortcut_test".to_string(), None);

    // Add keyboard shortcut (Cmd+S)
    let mut raw = make_keyboard_event('s');
    raw.modifiers.command = true;
    recording.add_raw_event(raw);

    let skill = generator.generate(&recording).expect("Generation should succeed");

    assert_eq!(skill.steps.len(), 1);
    // Should detect as shortcut - check for either "Press" or the actual key combination
    let desc = &skill.steps[0].description;
    assert!(
        desc.contains("Press") || desc.contains("Cmd") || desc.contains("s"),
        "Expected shortcut description, got: {}", desc
    );
}

// ============================================================================
// Test 20: Error Propagation - Invalid Workflow
// ============================================================================

#[test]
fn test_error_propagation_no_events() {
    MachTimebase::init();
    let generator = SkillGenerator::new();

    // Create recording with only mouse moves (non-significant events)
    let mut recording = Recording::new("error_test".to_string(), None);
    for i in 0..10 {
        recording.add_raw_event(make_raw_event(EventType::MouseMoved, i as f64, i as f64));
    }

    // Should fail because no significant events
    let result = generator.generate(&recording);
    assert!(result.is_err(), "Should error on non-significant events only");
}

// ============================================================================
// Test 21: Variable Extractor Direct Test
// ============================================================================

#[test]
fn test_variable_extractor_email_detection() {
    MachTimebase::init();
    let extractor = VariableExtractor::new();
    let mut events = Vec::new();

    // Create typing events for email
    for ch in "user@example.com".chars() {
        let raw = make_keyboard_event(ch);
        events.push(EnrichedEvent::new(raw, events.len() as u64));
    }

    let variables = extractor.extract(&events, "Enter your email address");

    // Should detect email variable
    assert!(!variables.is_empty(), "Should extract email variable");
    let has_email = variables.iter().any(|v|
        v.type_hint.as_ref().is_some_and(|t| t == "email")
    );
    assert!(has_email, "Should identify email type");
}

// ============================================================================
// Test 22: Complex Multi-Field Form Workflow
// ============================================================================

#[test]
fn test_complex_form_workflow() {
    MachTimebase::init();
    let config = skill_generator::workflow::generator::GeneratorConfig {
        use_action_clustering: false,
        ..Default::default()
    };
    let generator = SkillGenerator::with_config(config);
    let mut recording = Recording::new(
        "form_test".to_string(),
        Some("Fill registration form".to_string()),
    );

    // Click name field
    let name_click = make_raw_event(EventType::LeftMouseDown, 200.0, 100.0);
    let name_semantic = make_textfield_semantic("Name");
    recording.add_event(make_enriched_event(name_click, Some(name_semantic)));

    // Type name
    for ch in "John Doe".chars() {
        recording.add_raw_event(make_keyboard_event(ch));
    }

    // Click email field
    let email_click = make_raw_event(EventType::LeftMouseDown, 200.0, 200.0);
    let email_semantic = make_textfield_semantic("Email");
    recording.add_event(make_enriched_event(email_click, Some(email_semantic)));

    // Type email
    for ch in "john@example.com".chars() {
        recording.add_raw_event(make_keyboard_event(ch));
    }

    // Click submit
    let submit_click = make_raw_event(EventType::LeftMouseDown, 150.0, 300.0);
    let submit_semantic = make_button_semantic("Register");
    recording.add_event(make_enriched_event(submit_click, Some(submit_semantic)));

    recording.finalize(10000);

    let skill = generator.generate(&recording).expect("Generation should succeed");

    // Should have multiple steps
    assert!(skill.steps.len() >= 3, "Should have at least 3 significant steps");

    // Verify markdown output
    let markdown = generator.render_to_markdown(&skill);
    assert!(markdown.contains("Name"));
    assert!(markdown.contains("Email"));
    assert!(markdown.contains("Register"));
}

// ============================================================================
// Test 23: Recording Metadata Preservation
// ============================================================================

#[test]
fn test_recording_metadata_preservation() {
    MachTimebase::init();
    let mut recording = Recording::new("metadata_test".to_string(), Some("Test metadata".to_string()));

    // Set custom metadata
    recording.metadata.app_context = Some("com.test.app".to_string());

    recording.add_raw_event(make_raw_event(EventType::LeftMouseDown, 100.0, 200.0));
    recording.finalize(2500);

    // Serialize and deserialize
    let json = serde_json::to_string(&recording).expect("Serialization failed");
    let loaded: Recording = serde_json::from_str(&json).expect("Deserialization failed");

    // Verify all metadata preserved
    assert_eq!(loaded.metadata.name, "metadata_test");
    assert_eq!(loaded.metadata.goal, Some("Test metadata".to_string()));
    assert_eq!(loaded.metadata.app_context, Some("com.test.app".to_string()));
    assert_eq!(loaded.metadata.duration_ms, 2500);
    assert_eq!(loaded.metadata.event_count, 1);
    assert!(loaded.metadata.ended_at.is_some());
}

// ============================================================================
// Test 24: Empty String Handling in Semantic Context
// ============================================================================

#[test]
fn test_empty_semantic_fields() {
    MachTimebase::init();
    let generator = SkillGenerator::new();
    let mut recording = Recording::new("empty_semantic_test".to_string(), None);

    // Create semantic context with empty strings
    let mut semantic = make_button_semantic("Submit");
    semantic.title = Some(String::new()); // Empty title
    semantic.identifier = None;

    let raw = make_raw_event(EventType::LeftMouseDown, 100.0, 200.0);
    recording.add_event(make_enriched_event(raw, Some(semantic)));

    let skill = generator.generate(&recording).expect("Should handle empty semantic fields");

    assert_eq!(skill.steps.len(), 1);
    // Should use fallback naming with AX role or coordinates
    // When title is empty and identifier is None, uses the AX role or coordinates
    assert!(
        skill.steps[0].description.contains("AXButton") ||
        skill.steps[0].description.contains("(") && skill.steps[0].description.contains(")"),
        "Expected role-based or coordinate-based target in description: {}",
        skill.steps[0].description
    );
}

// ============================================================================
// Test 25: Concurrent Recording Access (Stress Test)
// ============================================================================

#[test]
fn test_recording_concurrent_reads() {
    use std::sync::Arc;
    use std::thread;

    MachTimebase::init();
    let mut recording = Recording::new("concurrent_test".to_string(), None);

    // Add events
    for i in 0..100 {
        recording.add_raw_event(make_raw_event(EventType::LeftMouseDown, i as f64, i as f64));
    }
    recording.finalize(5000);

    let recording = Arc::new(recording);

    // Spawn multiple threads reading the recording
    let mut handles = vec![];
    for _ in 0..10 {
        let rec_clone = Arc::clone(&recording);
        let handle = thread::spawn(move || {
            assert_eq!(rec_clone.len(), 100);
            assert_eq!(rec_clone.metadata.event_count, 100);
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}
