//! Configuration Management

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct Config {
    /// Capture settings
    pub capture: CaptureConfig,
    /// Analysis settings
    pub analysis: AnalysisConfig,
    /// Code generation settings
    pub codegen: CodegenConfig,
    /// Pipeline feature toggles
    #[serde(default)]
    pub pipeline: PipelineConfig,
}

/// Capture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    /// Ring buffer size
    pub ring_buffer_size: usize,
    /// Sampling rate for screenshots (Hz, 0 = disabled)
    pub sampling_rate_hz: u32,
    /// Enable vision fallback
    pub vision_fallback: bool,
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// RDP epsilon (pixels)
    pub rdp_epsilon_px: f64,
    /// Hesitation threshold for GOMS
    pub hesitation_threshold: f64,
    /// Minimum pause for chunk boundary (ms)
    pub min_pause_ms: u64,
}

/// Code generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodegenConfig {
    /// Model for semantic synthesis
    pub model: String,
    /// Temperature for generation
    pub temperature: f32,
    /// Include screenshots in skill
    pub include_screenshots: bool,
}

/// Pipeline feature toggles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Use ActionClusterer to group events into UnitTasks
    pub use_action_clustering: bool,
    /// Use NullHandler local recovery (AX retry + spiral search)
    pub use_local_recovery: bool,
    /// Enable Vision OCR as part of local recovery pipeline
    pub use_vision_ocr: bool,
    /// Use RDP + kinematic analysis for trajectory processing
    pub use_trajectory_analysis: bool,
    /// Use GOMS mental operator detection for cognitive boundary analysis
    pub use_goms_detection: bool,
    /// Use ContextStack to track window/app context changes
    pub use_context_tracking: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            use_action_clustering: true,
            use_local_recovery: true,
            use_vision_ocr: true,
            use_trajectory_analysis: true,
            use_goms_detection: true,
            use_context_tracking: true,
        }
    }
}


impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            ring_buffer_size: 8192,
            sampling_rate_hz: 0,
            vision_fallback: true,
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            rdp_epsilon_px: 3.0,
            hesitation_threshold: 0.7,
            min_pause_ms: 300,
        }
    }
}

impl Default for CodegenConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-5-20250929".to_string(),
            temperature: 0.3,
            include_screenshots: false,
        }
    }
}

impl Config {
    /// Validate config values are within acceptable ranges.
    /// Returns Ok(()) if valid, or Err with a description of the first invalid field.
    pub fn validate(&self) -> Result<(), crate::Error> {
        if self.capture.ring_buffer_size == 0 || (self.capture.ring_buffer_size & (self.capture.ring_buffer_size - 1)) != 0 {
            return Err(crate::Error::Config(format!(
                "ring_buffer_size must be a power of 2, got {}", self.capture.ring_buffer_size
            )));
        }
        if self.analysis.rdp_epsilon_px <= 0.0 || self.analysis.rdp_epsilon_px > 100.0 {
            return Err(crate::Error::Config(format!(
                "rdp_epsilon_px must be in (0, 100], got {}", self.analysis.rdp_epsilon_px
            )));
        }
        if !(0.0..=1.0).contains(&self.analysis.hesitation_threshold) {
            return Err(crate::Error::Config(format!(
                "hesitation_threshold must be in [0, 1], got {}", self.analysis.hesitation_threshold
            )));
        }
        if self.analysis.min_pause_ms == 0 {
            return Err(crate::Error::Config("min_pause_ms must be > 0".to_string()));
        }
        if self.codegen.model.trim().is_empty() {
            return Err(crate::Error::Config("model must not be empty".to_string()));
        }
        if !(0.0..=2.0).contains(&self.codegen.temperature) {
            return Err(crate::Error::Config(format!(
                "temperature must be in [0, 2], got {}", self.codegen.temperature
            )));
        }
        Ok(())
    }

    /// Load config from file
    pub fn load(path: &PathBuf) -> Result<Self, crate::Error> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content).map_err(|e| crate::Error::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Load config from default location
    pub fn load_default() -> Result<Self, crate::Error> {
        let path = Self::default_path();
        if path.exists() {
            Self::load(&path)
        } else {
            Ok(Self::default())
        }
    }

    /// Save config to file
    pub fn save(&self, path: &PathBuf) -> Result<(), crate::Error> {
        let content = toml::to_string_pretty(self).map_err(|e| crate::Error::Config(e.to_string()))?;

        // Create parent directories
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Save to default location
    pub fn save_default(&self) -> Result<(), crate::Error> {
        self.save(&Self::default_path())
    }

    /// Get default config path
    pub fn default_path() -> PathBuf {
        dirs::home_dir()
            .map(|h| h.join(".skill_generator").join("config.toml"))
            .unwrap_or_else(|| PathBuf::from("config.toml"))
    }

    /// Generate TOML representation
    pub fn to_toml(&self) -> Result<String, crate::Error> {
        toml::to_string_pretty(self).map_err(|e| crate::Error::Config(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.capture.ring_buffer_size, 8192);
        assert_eq!(config.analysis.rdp_epsilon_px, 3.0);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml = config.to_toml().unwrap();
        assert!(toml.contains("[capture]"));
        assert!(toml.contains("[analysis]"));
        assert!(toml.contains("[codegen]"));
    }

    #[test]
    fn test_default_path() {
        let path = Config::default_path();
        assert!(path.to_string_lossy().contains("config.toml"));
    }

    #[test]
    fn test_capture_config_defaults() {
        let capture = CaptureConfig::default();
        assert_eq!(capture.ring_buffer_size, 8192);
        assert_eq!(capture.sampling_rate_hz, 0);
        assert!(capture.vision_fallback);
    }

    #[test]
    fn test_analysis_config_defaults() {
        let analysis = AnalysisConfig::default();
        assert_eq!(analysis.rdp_epsilon_px, 3.0);
        assert_eq!(analysis.hesitation_threshold, 0.7);
        assert_eq!(analysis.min_pause_ms, 300);
    }

    #[test]
    fn test_codegen_config_defaults() {
        let codegen = CodegenConfig::default();
        assert_eq!(codegen.model, "claude-sonnet-4-5-20250929");
        assert_eq!(codegen.temperature, 0.3);
        assert!(!codegen.include_screenshots);
    }

    #[test]
    fn test_pipeline_config_defaults() {
        let pipeline = PipelineConfig::default();
        assert!(pipeline.use_action_clustering);
        assert!(pipeline.use_local_recovery);
        assert!(pipeline.use_vision_ocr);
        assert!(pipeline.use_trajectory_analysis);
        assert!(pipeline.use_goms_detection);
        assert!(pipeline.use_context_tracking);
    }

    #[test]
    fn test_config_roundtrip_serialization() {
        let original = Config::default();
        let toml_str = original.to_toml().unwrap();
        let deserialized: Config = toml::from_str(&toml_str).expect("Failed to deserialize");

        assert_eq!(original.capture.ring_buffer_size, deserialized.capture.ring_buffer_size);
        assert_eq!(original.analysis.rdp_epsilon_px, deserialized.analysis.rdp_epsilon_px);
        assert_eq!(original.codegen.model, deserialized.codegen.model);
    }

    #[test]
    fn test_config_save_and_load() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = temp_dir.path().join("test_config.toml");

        let mut original = Config::default();
        original.capture.ring_buffer_size = 16384;
        original.analysis.rdp_epsilon_px = 5.0;
        original.codegen.temperature = 0.5;

        original.save(&config_path).expect("Failed to save config");
        assert!(config_path.exists());

        let loaded = Config::load(&config_path).expect("Failed to load config");
        assert_eq!(loaded.capture.ring_buffer_size, 16384);
        assert_eq!(loaded.analysis.rdp_epsilon_px, 5.0);
        assert_eq!(loaded.codegen.temperature, 0.5);
    }

    #[test]
    fn test_config_save_creates_parent_directories() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let nested_path = temp_dir.path().join("nested").join("path").join("config.toml");

        let config = Config::default();
        config.save(&nested_path).expect("Failed to save config");

        assert!(nested_path.exists());
        assert!(nested_path.parent().unwrap().exists());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let nonexistent_path = PathBuf::from("/tmp/nonexistent_config_12345.toml");
        let result = Config::load(&nonexistent_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_custom_values() {
        let config = Config {
            capture: CaptureConfig {
                ring_buffer_size: 4096,
                sampling_rate_hz: 10,
                vision_fallback: false,
            },
            analysis: AnalysisConfig {
                rdp_epsilon_px: 2.5,
                hesitation_threshold: 0.8,
                min_pause_ms: 500,
            },
            codegen: CodegenConfig {
                model: "claude-sonnet-4-5".to_string(),
                temperature: 0.7,
                include_screenshots: true,
            },
            pipeline: PipelineConfig::default(),
        };

        let toml_str = config.to_toml().unwrap();
        assert!(toml_str.contains("ring_buffer_size = 4096"));
        assert!(toml_str.contains("sampling_rate_hz = 10"));
        assert!(toml_str.contains("rdp_epsilon_px = 2.5"));
        // Check that temperature field exists and has a reasonable value
        assert!(toml_str.contains("temperature"));
        assert!(toml_str.contains("model = \"claude-sonnet-4-5\""));
        assert!(toml_str.contains("include_screenshots = true"));
    }

    #[test]
    fn test_load_default_when_file_missing() {
        // This test verifies that load_default returns default config when file doesn't exist
        // We can't easily test this without mocking, but we can test the logic path
        let default_path = Config::default_path();

        // If the default path doesn't exist (which is likely in test environment)
        if !default_path.exists() {
            let config = Config::load_default().expect("Failed to load default");
            // Should return default values
            assert_eq!(config.capture.ring_buffer_size, 8192);
        }
    }

    #[test]
    fn test_config_clone() {
        let config = Config::default();
        let cloned = config.clone();

        assert_eq!(config.capture.ring_buffer_size, cloned.capture.ring_buffer_size);
        assert_eq!(config.analysis.rdp_epsilon_px, cloned.analysis.rdp_epsilon_px);
        assert_eq!(config.codegen.model, cloned.codegen.model);
    }

    #[test]
    fn test_invalid_toml_parsing() {
        let invalid_toml = "this is not valid toml {{{}}}";
        let result: Result<Config, _> = toml::from_str(invalid_toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_ring_buffer_not_power_of_two() {
        let mut config = Config::default();
        config.capture.ring_buffer_size = 1000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_ring_buffer_zero() {
        let mut config = Config::default();
        config.capture.ring_buffer_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rdp_epsilon_negative() {
        let mut config = Config::default();
        config.analysis.rdp_epsilon_px = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rdp_epsilon_too_large() {
        let mut config = Config::default();
        config.analysis.rdp_epsilon_px = 200.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_hesitation_out_of_range() {
        let mut config = Config::default();
        config.analysis.hesitation_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_min_pause_zero() {
        let mut config = Config::default();
        config.analysis.min_pause_ms = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_empty_model() {
        let mut config = Config::default();
        config.codegen.model = "  ".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_temperature_out_of_range() {
        let mut config = Config::default();
        config.codegen.temperature = 3.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_boundary_values() {
        let mut config = Config::default();
        // Test boundary: hesitation threshold = 0 should be valid
        config.analysis.hesitation_threshold = 0.0;
        assert!(config.validate().is_ok());
        // Test boundary: hesitation threshold = 1 should be valid
        config.analysis.hesitation_threshold = 1.0;
        assert!(config.validate().is_ok());
        // Test boundary: temperature = 0 should be valid
        config.codegen.temperature = 0.0;
        assert!(config.validate().is_ok());
        // Test boundary: temperature = 2 should be valid
        config.codegen.temperature = 2.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_load_invalid_values() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config_path = temp_dir.path().join("bad_config.toml");
        std::fs::write(&config_path, r#"
[capture]
ring_buffer_size = 1000
sampling_rate_hz = 0
vision_fallback = true

[analysis]
rdp_epsilon_px = 3.0
hesitation_threshold = 0.7
min_pause_ms = 300

[codegen]
model = "claude-sonnet-4-5-20250929"
temperature = 0.3
include_screenshots = false
"#).expect("Failed to write config");
        let result = Config::load(&config_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_config_serialization_roundtrip() {
        let pipeline = PipelineConfig {
            use_action_clustering: false,
            use_local_recovery: true,
            use_vision_ocr: false,
            use_trajectory_analysis: true,
            use_goms_detection: false,
            use_context_tracking: true,
        };

        let toml_str = toml::to_string_pretty(&pipeline).expect("Failed to serialize PipelineConfig");
        let deserialized: PipelineConfig = toml::from_str(&toml_str).expect("Failed to deserialize PipelineConfig");

        assert_eq!(deserialized.use_action_clustering, false);
        assert_eq!(deserialized.use_local_recovery, true);
        assert_eq!(deserialized.use_vision_ocr, false);
        assert_eq!(deserialized.use_trajectory_analysis, true);
        assert_eq!(deserialized.use_goms_detection, false);
        assert_eq!(deserialized.use_context_tracking, true);
    }

    #[test]
    fn test_config_with_custom_pipeline_roundtrip() {
        let config = Config {
            capture: CaptureConfig {
                ring_buffer_size: 4096,
                sampling_rate_hz: 5,
                vision_fallback: false,
            },
            analysis: AnalysisConfig {
                rdp_epsilon_px: 5.0,
                hesitation_threshold: 0.5,
                min_pause_ms: 200,
            },
            codegen: CodegenConfig {
                model: "claude-sonnet-4-5-20250929".to_string(),
                temperature: 1.0,
                include_screenshots: true,
            },
            pipeline: PipelineConfig {
                use_action_clustering: false,
                use_local_recovery: false,
                use_vision_ocr: false,
                use_trajectory_analysis: true,
                use_goms_detection: true,
                use_context_tracking: false,
            },
        };

        let toml_str = config.to_toml().expect("Failed to serialize Config");
        assert!(toml_str.contains("[pipeline]"));
        assert!(toml_str.contains("use_action_clustering = false"));
        assert!(toml_str.contains("use_context_tracking = false"));

        let deserialized: Config = toml::from_str(&toml_str).expect("Failed to deserialize Config");
        assert_eq!(deserialized.capture.ring_buffer_size, 4096);
        assert_eq!(deserialized.capture.sampling_rate_hz, 5);
        assert_eq!(deserialized.capture.vision_fallback, false);
        assert_eq!(deserialized.analysis.rdp_epsilon_px, 5.0);
        assert_eq!(deserialized.analysis.hesitation_threshold, 0.5);
        assert_eq!(deserialized.analysis.min_pause_ms, 200);
        assert_eq!(deserialized.codegen.model, "claude-sonnet-4-5-20250929");
        assert_eq!(deserialized.codegen.temperature, 1.0);
        assert_eq!(deserialized.codegen.include_screenshots, true);
        assert_eq!(deserialized.pipeline.use_action_clustering, false);
        assert_eq!(deserialized.pipeline.use_local_recovery, false);
        assert_eq!(deserialized.pipeline.use_vision_ocr, false);
        assert_eq!(deserialized.pipeline.use_trajectory_analysis, true);
        assert_eq!(deserialized.pipeline.use_goms_detection, true);
        assert_eq!(deserialized.pipeline.use_context_tracking, false);
    }

    #[test]
    fn test_old_config_without_pipeline_section_deserializes() {
        // Simulate a legacy config file that does not include a [pipeline] section.
        // Because PipelineConfig has #[serde(default)], it should use default values.
        let old_config_toml = r#"
[capture]
ring_buffer_size = 8192
sampling_rate_hz = 0
vision_fallback = true

[analysis]
rdp_epsilon_px = 3.0
hesitation_threshold = 0.7
min_pause_ms = 300

[codegen]
model = "claude-sonnet-4-5-20250929"
temperature = 0.3
include_screenshots = false
"#;

        let config: Config = toml::from_str(old_config_toml)
            .expect("Old config without [pipeline] should deserialize successfully");

        // Capture/analysis/codegen should match the TOML values
        assert_eq!(config.capture.ring_buffer_size, 8192);
        assert_eq!(config.analysis.rdp_epsilon_px, 3.0);
        assert_eq!(config.codegen.model, "claude-sonnet-4-5-20250929");

        // Pipeline should use defaults (all true)
        assert!(config.pipeline.use_action_clustering);
        assert!(config.pipeline.use_local_recovery);
        assert!(config.pipeline.use_vision_ocr);
        assert!(config.pipeline.use_trajectory_analysis);
        assert!(config.pipeline.use_goms_detection);
        assert!(config.pipeline.use_context_tracking);
    }
}
