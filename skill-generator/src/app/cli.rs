//! Command-Line Interface

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Skill Generator - Transform user interactions into Claude Code skills
#[derive(Parser, Debug)]
#[command(name = "skill-gen")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Subcommand to run
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Config file path
    #[arg(short, long, global = true)]
    pub config: Option<PathBuf>,
}

/// Available commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start recording user interactions
    Record {
        /// Recording duration in seconds (0 = until stopped)
        #[arg(short, long, default_value = "60")]
        duration: u64,

        /// Output file name (without extension)
        #[arg(short, long)]
        output: Option<String>,

        /// Goal description for the skill
        #[arg(short, long)]
        goal: Option<String>,
    },

    /// Generate SKILL.md from a recording
    Generate {
        /// Input recording file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for SKILL.md
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Skill name (inferred if not provided)
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Test a generated skill
    Test {
        /// Path to SKILL.md
        #[arg(short, long)]
        skill: PathBuf,

        /// Number of test repetitions
        #[arg(short, long, default_value = "1")]
        repeat: u32,

        /// Dry run (don't execute, just validate)
        #[arg(long)]
        dry_run: bool,
    },

    /// Validate a SKILL.md file
    Validate {
        /// Path to SKILL.md
        skill: PathBuf,
    },

    /// List recordings
    List {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Initialize configuration
    Init {
        /// Force overwrite existing config
        #[arg(short, long)]
        force: bool,
    },

    /// Delete a recording
    Delete {
        /// Recording name or ID to delete
        name: String,

        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },

    /// View or modify configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

/// Config subcommands
#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Show current configuration
    Show,

    /// Set a configuration value
    Set {
        /// Configuration key (e.g., "codegen.model", "recording.duration")
        key: String,

        /// Value to set
        value: String,
    },

    /// Get a specific configuration value
    Get {
        /// Configuration key
        key: String,
    },

    /// Reset configuration to defaults
    Reset {
        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },
}

impl Cli {
    /// Parse command line arguments
    pub fn parse_args() -> Self {
        Self::parse()
    }

    /// Get the recording directory
    pub fn recordings_dir() -> PathBuf {
        dirs::home_dir()
            .map(|h| h.join(".skill_generator").join("recordings"))
            .unwrap_or_else(|| PathBuf::from("recordings"))
    }

    /// Get the skills output directory
    pub fn skills_dir() -> PathBuf {
        dirs::home_dir()
            .map(|h| h.join(".claude").join("skills"))
            .unwrap_or_else(|| PathBuf::from("skills"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_recordings_dir() {
        let dir = Cli::recordings_dir();
        assert!(dir.to_string_lossy().contains("recordings"));
    }

    #[test]
    fn test_skills_dir() {
        let dir = Cli::skills_dir();
        assert!(dir.to_string_lossy().contains("skills"));
    }

    #[test]
    fn test_cli_parse_record_command_with_defaults() {
        let args = vec!["skill-gen", "record"];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Record { duration, output, goal } => {
                assert_eq!(duration, 60);
                assert!(output.is_none());
                assert!(goal.is_none());
            }
            _ => panic!("Expected Record command"),
        }
    }

    #[test]
    fn test_cli_parse_record_command_with_all_options() {
        let args = vec![
            "skill-gen",
            "record",
            "--duration", "120",
            "--output", "my-recording",
            "--goal", "Click the submit button",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Record { duration, output, goal } => {
                assert_eq!(duration, 120);
                assert_eq!(output.as_deref(), Some("my-recording"));
                assert_eq!(goal.as_deref(), Some("Click the submit button"));
            }
            _ => panic!("Expected Record command"),
        }
    }

    #[test]
    fn test_cli_parse_generate_command() {
        let args = vec![
            "skill-gen",
            "generate",
            "--input", "/path/to/recording.json",
            "--output", "/path/to/output",
            "--name", "my-skill",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Generate { input, output, name } => {
                assert_eq!(input, PathBuf::from("/path/to/recording.json"));
                assert_eq!(output, Some(PathBuf::from("/path/to/output")));
                assert_eq!(name.as_deref(), Some("my-skill"));
            }
            _ => panic!("Expected Generate command"),
        }
    }

    #[test]
    fn test_cli_parse_test_command() {
        let args = vec![
            "skill-gen",
            "test",
            "--skill", "/path/to/SKILL.md",
            "--repeat", "5",
            "--dry-run",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Test { skill, repeat, dry_run } => {
                assert_eq!(skill, PathBuf::from("/path/to/SKILL.md"));
                assert_eq!(repeat, 5);
                assert!(dry_run);
            }
            _ => panic!("Expected Test command"),
        }
    }

    #[test]
    fn test_cli_parse_test_command_defaults() {
        let args = vec![
            "skill-gen",
            "test",
            "--skill", "/path/to/SKILL.md",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Test { skill, repeat, dry_run } => {
                assert_eq!(skill, PathBuf::from("/path/to/SKILL.md"));
                assert_eq!(repeat, 1);
                assert!(!dry_run);
            }
            _ => panic!("Expected Test command"),
        }
    }

    #[test]
    fn test_cli_parse_validate_command() {
        let args = vec![
            "skill-gen",
            "validate",
            "/path/to/SKILL.md",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Validate { skill } => {
                assert_eq!(skill, PathBuf::from("/path/to/SKILL.md"));
            }
            _ => panic!("Expected Validate command"),
        }
    }

    #[test]
    fn test_cli_parse_list_command() {
        let args = vec![
            "skill-gen",
            "list",
            "--detailed",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::List { detailed } => {
                assert!(detailed);
            }
            _ => panic!("Expected List command"),
        }
    }

    #[test]
    fn test_cli_parse_list_command_defaults() {
        let args = vec![
            "skill-gen",
            "list",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::List { detailed } => {
                assert!(!detailed);
            }
            _ => panic!("Expected List command"),
        }
    }

    #[test]
    fn test_cli_parse_init_command() {
        let args = vec![
            "skill-gen",
            "init",
            "--force",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Init { force } => {
                assert!(force);
            }
            _ => panic!("Expected Init command"),
        }
    }

    #[test]
    fn test_cli_parse_init_command_defaults() {
        let args = vec![
            "skill-gen",
            "init",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Init { force } => {
                assert!(!force);
            }
            _ => panic!("Expected Init command"),
        }
    }

    #[test]
    fn test_cli_global_verbose_flag() {
        let args = vec![
            "skill-gen",
            "--verbose",
            "record",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        assert!(cli.verbose);
    }

    #[test]
    fn test_cli_global_config_flag() {
        let args = vec![
            "skill-gen",
            "--config", "/path/to/config.toml",
            "record",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.config, Some(PathBuf::from("/path/to/config.toml")));
    }

    #[test]
    fn test_cli_verbose_shorthand() {
        let args = vec![
            "skill-gen",
            "-v",
            "record",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        assert!(cli.verbose);
    }

    #[test]
    fn test_cli_config_shorthand() {
        let args = vec![
            "skill-gen",
            "-c", "/custom/config.toml",
            "record",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.config, Some(PathBuf::from("/custom/config.toml")));
    }

    #[test]
    fn test_cli_record_duration_shorthand() {
        let args = vec![
            "skill-gen",
            "record",
            "-d", "300",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Record { duration, .. } => {
                assert_eq!(duration, 300);
            }
            _ => panic!("Expected Record command"),
        }
    }

    #[test]
    fn test_cli_invalid_command_fails() {
        let args = vec!["skill-gen", "invalid-command"];
        let result = Cli::try_parse_from(args);
        assert!(result.is_err());
    }

    #[test]
    fn test_cli_missing_required_argument_fails() {
        let args = vec!["skill-gen", "generate"];
        let result = Cli::try_parse_from(args);
        assert!(result.is_err());
    }

    #[test]
    fn test_cli_verify_command_structure() {
        let cmd = Cli::command();

        // Verify subcommands exist
        let subcommands: Vec<_> = cmd.get_subcommands().map(|s| s.get_name()).collect();
        assert!(subcommands.contains(&"record"));
        assert!(subcommands.contains(&"generate"));
        assert!(subcommands.contains(&"test"));
        assert!(subcommands.contains(&"validate"));
        assert!(subcommands.contains(&"list"));
        assert!(subcommands.contains(&"init"));
        assert!(subcommands.contains(&"delete"));
        assert!(subcommands.contains(&"config"));
    }

    #[test]
    fn test_recordings_dir_fallback() {
        // Even if home_dir returns None, we should get a fallback
        let dir = Cli::recordings_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_skills_dir_fallback() {
        // Even if home_dir returns None, we should get a fallback
        let dir = Cli::skills_dir();
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn test_cli_parse_delete_command() {
        let args = vec![
            "skill-gen",
            "delete",
            "my-recording",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Delete { name, force } => {
                assert_eq!(name, "my-recording");
                assert!(!force);
            }
            _ => panic!("Expected Delete command"),
        }
    }

    #[test]
    fn test_cli_parse_delete_command_force() {
        let args = vec![
            "skill-gen",
            "delete",
            "old-recording",
            "--force",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Delete { name, force } => {
                assert_eq!(name, "old-recording");
                assert!(force);
            }
            _ => panic!("Expected Delete command"),
        }
    }

    #[test]
    fn test_cli_parse_config_show() {
        let args = vec![
            "skill-gen",
            "config",
            "show",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Config { action: ConfigAction::Show } => {}
            _ => panic!("Expected Config Show"),
        }
    }

    #[test]
    fn test_cli_parse_config_set() {
        let args = vec![
            "skill-gen",
            "config",
            "set",
            "codegen.model",
            "claude-sonnet-4-5-20250929",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Config { action: ConfigAction::Set { key, value } } => {
                assert_eq!(key, "codegen.model");
                assert_eq!(value, "claude-sonnet-4-5-20250929");
            }
            _ => panic!("Expected Config Set"),
        }
    }

    #[test]
    fn test_cli_parse_config_get() {
        let args = vec![
            "skill-gen",
            "config",
            "get",
            "codegen.temperature",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Config { action: ConfigAction::Get { key } } => {
                assert_eq!(key, "codegen.temperature");
            }
            _ => panic!("Expected Config Get"),
        }
    }

    #[test]
    fn test_cli_parse_config_reset() {
        let args = vec![
            "skill-gen",
            "config",
            "reset",
            "--force",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Config { action: ConfigAction::Reset { force } } => {
                assert!(force);
            }
            _ => panic!("Expected Config Reset"),
        }
    }

    #[test]
    fn test_cli_parse_config_reset_defaults() {
        let args = vec![
            "skill-gen",
            "config",
            "reset",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Config { action: ConfigAction::Reset { force } } => {
                assert!(!force);
            }
            _ => panic!("Expected Config Reset"),
        }
    }

    #[test]
    fn test_cli_record_with_zero_duration() {
        let args = vec![
            "skill-gen",
            "record",
            "--duration", "0",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Record { duration, .. } => {
                assert_eq!(duration, 0);
            }
            _ => panic!("Expected Record command"),
        }
    }
}
