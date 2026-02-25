import { useState, useEffect, useRef, useCallback, Component } from "react";
import type { ReactNode, ErrorInfo } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import "./App.css";

interface RecordingStatus {
  is_recording: boolean;
  event_count: number;
  duration_seconds: number;
  has_accessibility: boolean;
  has_screen_recording: boolean;
  hotkey_stopped: boolean;
}

interface RecordingInfo {
  name: string;
  path: string;
  event_count: number;
  duration_ms: number;
  created_at: string;
  goal: string | null;
  has_screenshots: boolean;
  has_analysis: boolean;
}

interface ScreenshotAnalysisInfo {
  sequence: number;
  filename: string;
  intent: string;
  confidence: number;
  tier: string;
  user_edited: boolean;
  image_path: string;
}

interface AnalysisInfo {
  recording_name: string;
  analyses: ScreenshotAnalysisInfo[];
  total_screenshots: number;
  ocr_count: number;
  vision_count: number;
  edited_count: number;
}

interface PipelineStatsInfo {
  local_recovery_count: number;
  llm_enriched_count: number;
  goms_boundaries_count: number;
  context_transitions_count: number;
  unit_tasks_count: number;
  significant_events_count: number;
  trajectory_adjustments_count: number;
  noise_filtered_count: number;
  variables_count: number;
  generated_steps_count: number;
  screenshot_enhanced_count: number;
  state_diff_enriched_count: number;
  error_correction_count: number;
  warnings: string[];
}

interface GeneratedSkillInfo {
  name: string;
  path: string;
  steps_count: number;
  variables_count: number;
  stats: PipelineStatsInfo;
}

interface UiConfig {
  rdp_epsilon_px: number;
  hesitation_threshold: number;
  min_pause_ms: number;
  model: string;
  vision_model: string;
  temperature: number;
  use_action_clustering: boolean;
  use_local_recovery: boolean;
  use_vision_ocr: boolean;
  use_trajectory_analysis: boolean;
  use_goms_detection: boolean;
  use_context_tracking: boolean;
  capture_screenshots: boolean;
  use_state_diff: boolean;
  use_error_correction_filter: boolean;
  dot_radius: number;
  dot_color: number[];
  trajectory_ballistic_color: number[];
  trajectory_searching_color: number[];
  trajectory_thickness: number;
  crop_size: number;
}

const MODEL_OPTIONS = [
  { value: "claude-opus-4-6", label: "Claude Opus 4.6" },
  { value: "claude-sonnet-4-5-20250929", label: "Claude Sonnet 4.5" },
  { value: "claude-haiku-4-5-20250929", label: "Claude Haiku 4.5" },
];

interface SkillListEntry {
  name: string;
  category: string;
  path: string;
  size_bytes: number;
  modified_at: string;
}

type Tab = "record" | "recordings" | "skills" | "settings";

function MarkdownPreview({ content }: { content: string }) {
  const lines = content.split("\n");
  const elements: React.ReactNode[] = [];
  let i = 0;
  let key = 0;

  function renderInline(text: string): React.ReactNode[] {
    const parts: React.ReactNode[] = [];
    let remaining = text;
    let partKey = 0;

    while (remaining.length > 0) {
      // Inline code
      const codeMatch = remaining.match(/^`([^`]+)`/);
      if (codeMatch) {
        parts.push(<code key={partKey++} className="md-inline-code">{codeMatch[1]}</code>);
        remaining = remaining.slice(codeMatch[0].length);
        continue;
      }
      // Bold
      const boldMatch = remaining.match(/^\*\*([^*]+)\*\*/);
      if (boldMatch) {
        parts.push(<strong key={partKey++}>{boldMatch[1]}</strong>);
        remaining = remaining.slice(boldMatch[0].length);
        continue;
      }
      // Italic
      const italicMatch = remaining.match(/^\*([^*]+)\*/);
      if (italicMatch) {
        parts.push(<em key={partKey++}>{italicMatch[1]}</em>);
        remaining = remaining.slice(italicMatch[0].length);
        continue;
      }
      // Plain char
      const nextSpecial = remaining.slice(1).search(/[`*]/);
      if (nextSpecial === -1) {
        parts.push(remaining);
        break;
      }
      parts.push(remaining.slice(0, nextSpecial + 1));
      remaining = remaining.slice(nextSpecial + 1);
    }
    return parts;
  }

  while (i < lines.length) {
    const line = lines[i];

    // Code block
    if (line.startsWith("```")) {
      const lang = line.slice(3).trim().replace(/[^a-zA-Z0-9_-]/g, "");
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      elements.push(
        <pre key={key++} className="md-code-block" data-lang={lang || undefined}>
          <code>{codeLines.join("\n")}</code>
        </pre>
      );
      continue;
    }

    // Heading
    const headingMatch = line.match(/^(#{1,4})\s+(.*)/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const text = renderInline(headingMatch[2]);
      if (level === 1) elements.push(<h1 key={key++} className="md-heading">{text}</h1>);
      else if (level === 2) elements.push(<h2 key={key++} className="md-heading">{text}</h2>);
      else if (level === 3) elements.push(<h3 key={key++} className="md-heading">{text}</h3>);
      else elements.push(<h4 key={key++} className="md-heading">{text}</h4>);
      i++;
      continue;
    }

    // Horizontal rule
    if (/^---+\s*$/.test(line)) {
      elements.push(<hr key={key++} className="md-hr" />);
      i++;
      continue;
    }

    // List item
    if (/^[-*]\s/.test(line)) {
      const items: React.ReactNode[] = [];
      while (i < lines.length && /^[-*]\s/.test(lines[i])) {
        items.push(<li key={items.length}>{renderInline(lines[i].replace(/^[-*]\s/, ""))}</li>);
        i++;
      }
      elements.push(<ul key={key++} className="md-list">{items}</ul>);
      continue;
    }

    // Numbered list
    if (/^\d+\.\s/.test(line)) {
      const items: React.ReactNode[] = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        items.push(<li key={items.length}>{renderInline(lines[i].replace(/^\d+\.\s/, ""))}</li>);
        i++;
      }
      elements.push(<ol key={key++} className="md-list">{items}</ol>);
      continue;
    }

    // Empty line
    if (line.trim() === "") {
      i++;
      continue;
    }

    // Paragraph
    elements.push(<p key={key++} className="md-paragraph">{renderInline(line)}</p>);
    i++;
  }

  return <div className="md-preview">{elements}</div>;
}

class ErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ErrorBoundary caught:", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <p>{this.state.error?.message}</p>
          <button
            className="btn btn-primary"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [eventCount, setEventCount] = useState(0);
  const [duration, setDuration] = useState(0);
  const [hasAccessibility, setHasAccessibility] = useState(false);
  const [hasScreenRecording, setHasScreenRecording] = useState(false);
  const [eventsPerSecond, setEventsPerSecond] = useState(0);
  const lastEventCountRef = useRef(0);

  // Input state
  const [recordingName, setRecordingName] = useState(() => localStorage.getItem("lastRecordingName") || "");
  const [goal, setGoal] = useState(() => localStorage.getItem("lastGoal") || "");
  const [apiKey, setApiKey] = useState("");
  const [maskedApiKey, setMaskedApiKey] = useState<string | null>(null);

  // Recordings list
  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [loadingRecordings, setLoadingRecordings] = useState(false);
  const [loadingSkills, setLoadingSkills] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);
  const [exportingConfig, setExportingConfig] = useState(false);
  const [exportingRecording, setExportingRecording] = useState<string | null>(null);

  // UI state
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>(() => {
    const saved = localStorage.getItem("activeTab");
    return (saved === "record" || saved === "recordings" || saved === "skills" || saved === "settings") ? saved : "record";
  });
  const [generating, setGenerating] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  // Skill preview state
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const [previewName, setPreviewName] = useState<string | null>(null);

  // Track generated skills by recording path -> skill path
  const [generatedSkills, setGeneratedSkills] = useState<Record<string, string>>({});

  // Skills library state
  const [skills, setSkills] = useState<SkillListEntry[]>([]);
  const [skillPreviewContent, setSkillPreviewContent] = useState<string | null>(null);
  const [skillPreviewName, setSkillPreviewName] = useState<string | null>(null);
  const [confirmDeleteSkill, setConfirmDeleteSkill] = useState<string | null>(null);

  // Pipeline stats
  const [pipelineStats, setPipelineStats] = useState<PipelineStatsInfo | null>(null);

  // Screenshot review state
  const [reviewingRecording, setReviewingRecording] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisInfo | null>(null);
  const [analyzing, setAnalyzing] = useState<string | null>(null);
  const [selectedScreenshot, setSelectedScreenshot] = useState<ScreenshotAnalysisInfo | null>(null);

  // Config state
  const [config, setConfig] = useState<UiConfig | null>(null);
  const [showAnnotationSettings, setShowAnnotationSettings] = useState(false);

  // Timer ref
  const timerRef = useRef<number | null>(null);

  // Debounce timers for intent IPC persistence (keyed by sequence number)
  const intentDebounceRef = useRef<Record<number, number>>({});

  // Check permissions and load recordings on mount
  useEffect(() => {
    checkStatus();
    loadRecordings();
    loadSkills();
    loadApiKey();
    loadConfig();
  }, []);

  // Auto-dismiss notifications after 5 seconds
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  // Persist UI state to localStorage
  useEffect(() => { localStorage.setItem("activeTab", activeTab); }, [activeTab]);
  useEffect(() => { localStorage.setItem("lastRecordingName", recordingName); }, [recordingName]);
  useEffect(() => { localStorage.setItem("lastGoal", goal); }, [goal]);

  // Poll status while recording
  useEffect(() => {
    if (isRecording) {
      timerRef.current = window.setInterval(() => {
        checkStatus();
      }, 500);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  // Calculate events per second
  useEffect(() => {
    if (isRecording) {
      const rate = (eventCount - lastEventCountRef.current) * 2; // x2 because poll is 500ms
      setEventsPerSecond(Math.max(0, rate));
      lastEventCountRef.current = eventCount;
    } else {
      setEventsPerSecond(0);
      lastEventCountRef.current = 0;
    }
  }, [eventCount, isRecording]);

  async function checkStatus() {
    try {
      const status = await invoke<RecordingStatus>("get_recording_status");
      setIsRecording(status.is_recording);
      setEventCount(status.event_count);
      setDuration(status.duration_seconds);
      setHasAccessibility(status.has_accessibility);
      setHasScreenRecording(status.has_screen_recording);

      // Auto-stop when global hotkey (Cmd+Opt+Ctrl+S) was pressed
      if (status.hotkey_stopped && status.is_recording) {
        handleStopRecording();
      }
    } catch (e) {
      console.error("Failed to get status:", e);
    }
  }

  async function loadRecordings() {
    setLoadingRecordings(true);
    try {
      const list = await invoke<RecordingInfo[]>("list_recordings");
      setRecordings(list);
    } catch (e) {
      console.error("Failed to load recordings:", e);
      setError("Failed to load recordings from ~/.skill_generator/recordings. Check permissions and disk space.");
    } finally {
      setLoadingRecordings(false);
    }
  }

  async function loadApiKey() {
    try {
      const key = await invoke<string | null>("get_api_key");
      setMaskedApiKey(key);
    } catch (e) {
      console.error("Failed to load API key:", e);
    }
  }

  async function loadConfig() {
    try {
      const cfg = await invoke<UiConfig>("get_config");
      setConfig(cfg);
    } catch (e) {
      console.error("Failed to load config:", e);
    }
  }

  async function loadSkills() {
    setLoadingSkills(true);
    try {
      const list = await invoke<SkillListEntry[]>("list_generated_skills");
      setSkills(list);
    } catch (e) {
      console.error("Failed to load skills:", e);
    } finally {
      setLoadingSkills(false);
    }
  }

  const handleStartRecording = useCallback(async () => {
    setError(null);
    setSuccess(null);
    try {
      await invoke("start_recording", {
        name: recordingName || "",
        goal: goal || null
      });
      setIsRecording(true);
      setEventCount(0);
      setDuration(0);
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      if (errorMsg.includes("Accessibility")) {
        setError(`${errorMsg}. To fix: System Settings > Privacy & Security > Accessibility > Enable this app`);
      } else {
        setError(errorMsg);
      }
    }
  }, [recordingName, goal]);

  const handleStopRecording = useCallback(async () => {
    setError(null);
    try {
      const info = await invoke<RecordingInfo>("stop_recording");
      setIsRecording(false);
      setSuccess(`Recording saved: ${info.name} (${info.event_count} events)`);
      setRecordingName("");
      setGoal("");
      await loadRecordings();
      setActiveTab("recordings");
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      setError(errorMsg);
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.metaKey && e.key === "r" && !isRecording && hasAccessibility) {
        e.preventDefault();
        handleStartRecording();
      }
      if (e.key === "Escape" && isRecording) {
        e.preventDefault();
        handleStopRecording();
      }
      if (e.key === "Escape" && selectedScreenshot) {
        e.preventDefault();
        setSelectedScreenshot(null);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isRecording, hasAccessibility, selectedScreenshot, handleStartRecording, handleStopRecording]);

  async function handleGenerateSkill(recordingPath: string) {
    setError(null);
    setSuccess(null);
    setPreviewContent(null);
    setPreviewName(null);
    setPipelineStats(null);
    setGenerating(recordingPath);
    try {
      const skill = await invoke<GeneratedSkillInfo>("generate_skill", {
        recordingPath
      });
      setSuccess(`SKILL.md generated: ${skill.name} (${skill.steps_count} steps, ${skill.variables_count} variables)`);
      setPipelineStats(skill.stats);
      setGeneratedSkills(prev => ({ ...prev, [recordingPath]: skill.path }));
      await loadSkills();

      // Load preview
      const content = await readSkillFile(skill.path);
      if (content) { setPreviewContent(content); setPreviewName(skill.name); }
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      if (errorMsg.includes("Accessibility")) {
        setError(`${errorMsg}. To fix: System Settings > Privacy & Security > Accessibility > Enable this app`);
      } else if (errorMsg.includes("NoSignificantEvents")) {
        setError("No click or keystroke events found in this recording. Try recording a longer interaction.");
      } else {
        setError(errorMsg);
      }
    } finally {
      setGenerating(null);
    }
  }

  async function handleSaveApiKey() {
    setError(null);
    try {
      await invoke("set_api_key", { key: apiKey });
      setSuccess("API Key saved");
      setApiKey("");
      loadApiKey();
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      setError(errorMsg);
    }
  }

  function validateConfig(c: UiConfig): string | null {
    if (c.rdp_epsilon_px <= 0 || c.rdp_epsilon_px > 100) return "RDP epsilon must be between 0 and 100";
    if (c.hesitation_threshold < 0 || c.hesitation_threshold > 1) return "Hesitation threshold must be between 0 and 1";
    if (c.min_pause_ms === 0) return "Min pause must be greater than 0";
    if (c.temperature < 0 || c.temperature > 2) return "Temperature must be between 0 and 2";
    if (!c.model || c.model.trim() === "") return "Inference model cannot be empty";
    if (!c.vision_model || c.vision_model.trim() === "") return "Vision model cannot be empty";
    return null;
  }

  async function handleSaveConfig() {
    if (!config || savingConfig) return;
    setError(null);
    const validationError = validateConfig(config);
    if (validationError) {
      setError(validationError);
      return;
    }
    setSavingConfig(true);
    try {
      await invoke("save_config", { config });
      setSuccess("Configuration saved");
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      setError(errorMsg);
    } finally {
      setSavingConfig(false);
    }
  }

  async function handleOpenInFinder(path: string) {
    try {
      await invoke("open_in_finder", { path });
    } catch (e) {
      console.error("Failed to open in Finder:", e);
      setError("Failed to open in Finder");
    }
  }

  function handleDeleteRecording(name: string) {
    if (confirmDelete === name) {
      performDelete(name);
    } else {
      setConfirmDelete(name);
    }
  }

  async function performDelete(name: string) {
    setError(null);
    setSuccess(null);
    setConfirmDelete(null);
    try {
      await invoke("delete_recording", { name });
      setRecordings((prev) => prev.filter((r) => r.name !== name));
      // Clear any UI state tied to the deleted recording
      if (reviewingRecording === name) {
        setReviewingRecording(null);
        setAnalysisData(null);
        setSelectedScreenshot(null);
      }
      setGeneratedSkills((prev) => {
        const next = { ...prev };
        for (const key of Object.keys(next)) {
          if (key.includes(name)) delete next[key];
        }
        return next;
      });
      if (skillPreviewName && skillPreviewName.includes(name)) {
        setSkillPreviewContent(null);
        setSkillPreviewName(null);
      }
      setSuccess(`Deleted recording: ${name}`);
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      setError(`Failed to delete: ${errorMsg}`);
      await loadRecordings();
    }
  }

  async function handleDownloadSkill(path: string, name: string) {
    const content = await readSkillFile(path);
    if (!content) return;
    const blob = new Blob([content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}-SKILL.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleExportConfig() {
    if (exportingConfig) return;
    setExportingConfig(true);
    try {
      const toml = await invoke<string>("export_config");
      const blob = new Blob([toml], { type: "application/toml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "skill-generator-config.toml";
      a.click();
      URL.revokeObjectURL(url);
      setSuccess("Config exported");
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      setError(`Failed to export config: ${errorMsg}`);
    } finally {
      setExportingConfig(false);
    }
  }

  function handleImportConfig() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".toml";
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const tomlContent = await file.text();
        await invoke("import_config", { tomlContent });
        setSuccess("Config imported successfully");
        await loadConfig();
      } catch (e: unknown) {
        const errorMsg = e instanceof Error ? e.message : String(e);
        setError(`Failed to import config: ${errorMsg}`);
      }
    };
    input.click();
  }

  async function handleExportRecording(path: string, name: string) {
    if (exportingRecording) return;
    setExportingRecording(path);
    try {
      const content = await invoke<string>("read_recording_file", { recordingPath: path });
      const blob = new Blob([content], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${name}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      setError(`Failed to export recording: ${errorMsg}`);
    } finally {
      setExportingRecording(null);
    }
  }

  async function handleDeleteSkill(path: string, name: string) {
    if (confirmDeleteSkill === path) {
      setConfirmDeleteSkill(null);
      try {
        await invoke("delete_skill", { skillPath: path });
        setSkills((prev) => prev.filter((s) => s.path !== path));
        setSuccess(`Deleted skill: ${name}`);
        if (skillPreviewName === name) {
          setSkillPreviewContent(null);
          setSkillPreviewName(null);
        }
      } catch (e: unknown) {
        const errorMsg = e instanceof Error ? e.message : String(e);
        setError(`Failed to delete: ${errorMsg}`);
        await loadSkills();
      }
    } else {
      setConfirmDeleteSkill(path);
    }
  }

  async function readSkillFile(path: string): Promise<string | null> {
    try {
      return await invoke<string>("read_skill_file", { path });
    } catch {
      setError("Failed to load skill file");
      return null;
    }
  }

  async function copyToClipboard(text: string, label = "Copied to clipboard") {
    try {
      await navigator.clipboard.writeText(text);
      setSuccess(label);
    } catch {
      setError("Failed to copy to clipboard");
    }
  }

  async function handleAnalyzeRecording(recordingName: string) {
    setError(null);
    setSuccess(null);
    setAnalyzing(recordingName);
    try {
      const result = await invoke<AnalysisInfo>("analyze_recording", { recordingName });
      setAnalysisData(result);
      setReviewingRecording(recordingName);
      setSuccess(`Analysis complete: ${result.total_screenshots} screenshots (${result.ocr_count} OCR, ${result.vision_count} Vision)`);
      await loadRecordings();
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      setError(`Analysis failed: ${errorMsg}`);
    } finally {
      setAnalyzing(null);
    }
  }

  async function handleOpenReview(recordingName: string) {
    setError(null);
    try {
      const result = await invoke<AnalysisInfo | null>("get_analysis", { recordingName });
      if (result) {
        setAnalysisData(result);
        setReviewingRecording(recordingName);
      } else {
        setError("No analysis found. Run analysis first.");
      }
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      setError(`Failed to load analysis: ${errorMsg}`);
    }
  }

  function handleUpdateIntent(sequence: number, newIntent: string) {
    if (!reviewingRecording) return;
    const recordingName = reviewingRecording;

    // Update local state immediately for responsive UI
    setAnalysisData(prev => {
      if (!prev) return prev;
      const updatedAnalyses = prev.analyses.map(a =>
        a.sequence === sequence
          ? { ...a, intent: newIntent, user_edited: true, tier: "edited" }
          : a
      );
      return {
        ...prev,
        analyses: updatedAnalyses,
        edited_count: updatedAnalyses.filter(a => a.user_edited).length,
      };
    });

    // Debounce the IPC persistence call (500ms)
    if (intentDebounceRef.current[sequence]) {
      clearTimeout(intentDebounceRef.current[sequence]);
    }
    intentDebounceRef.current[sequence] = window.setTimeout(async () => {
      delete intentDebounceRef.current[sequence];
      try {
        await invoke("update_intent", {
          recordingName,
          sequence,
          newIntent,
        });
      } catch (e: unknown) {
        const errorMsg = e instanceof Error ? e.message : String(e);
        setError(`Failed to save intent: ${errorMsg}`);
      }
    }, 500);
  }

  function handleCloseReview() {
    // Clear any pending debounced IPC calls
    for (const timer of Object.values(intentDebounceRef.current)) {
      clearTimeout(timer);
    }
    intentDebounceRef.current = {};
    setReviewingRecording(null);
    setAnalysisData(null);
    setSelectedScreenshot(null);
  }

  function formatSkillSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  function formatDuration(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  function formatDurationMs(ms: number): string {
    if (ms < 1000) return "< 1s";
    const secs = Math.round(ms / 1000);
    if (secs < 60) return `${secs}s`;
    const mins = Math.floor(secs / 60);
    const remainder = secs % 60;
    return `${mins}m ${remainder}s`;
  }

  function formatDate(isoString: string): string {
    return new Date(isoString).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  }

  return (
    <main className="app">
      {/* Status Bar */}
      <header className="status-bar">
        <div className="status-badges">
          <span className={`badge ${hasAccessibility ? "badge-ok" : "badge-error"}`}>
            {hasAccessibility ? "Accessibility" : "No Accessibility"}
          </span>
          <span className={`badge ${hasScreenRecording ? "badge-ok" : "badge-warn"}`}>
            {hasScreenRecording ? "Screen Recording" : "No Screen Rec"}
          </span>
          <span className={`badge ${maskedApiKey ? "badge-ok" : "badge-warn"}`}>
            {maskedApiKey ? `API: ${maskedApiKey}` : "No API Key"}
          </span>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="tab-bar" role="tablist" aria-label="Main navigation">
        <button
          role="tab"
          aria-selected={activeTab === "record"}
          className={`tab ${activeTab === "record" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("record")}
        >
          Record
        </button>
        <button
          role="tab"
          aria-selected={activeTab === "recordings"}
          className={`tab ${activeTab === "recordings" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("recordings")}
        >
          Recordings{recordings.length > 0 ? ` (${recordings.length})` : ""}
        </button>
        <button
          role="tab"
          aria-selected={activeTab === "skills"}
          className={`tab ${activeTab === "skills" ? "tab-active" : ""}`}
          onClick={() => { setActiveTab("skills"); loadSkills(); }}
        >
          Skills{skills.length > 0 ? ` (${skills.length})` : ""}
        </button>
        <button
          role="tab"
          aria-selected={activeTab === "settings"}
          className={`tab ${activeTab === "settings" ? "tab-active" : ""}`}
          onClick={() => setActiveTab("settings")}
        >
          Settings
        </button>
      </nav>

      {/* Loading Overlay */}
      {(analyzing || generating) && (
        <div className="loading-overlay" role="status" aria-live="assertive">
          <div className="loading-card">
            <div className="loading-spinner" />
            <div className="loading-text">
              {analyzing ? "Analyzing screenshots..." : "Generating skill..."}
            </div>
            <div className="loading-hint">
              {analyzing ? "Running OCR and AI analysis" : "Processing events and building SKILL.md"}
            </div>
          </div>
        </div>
      )}

      {/* Alerts */}
      <div className="alerts" aria-live="polite">
        {error && (
          <div className="alert error" role="alert" onClick={() => setError(null)}>
            {error}
          </div>
        )}
        {success && (
          <div className="alert success" role="status" onClick={() => setSuccess(null)}>
            {success}
          </div>
        )}
      </div>

      {/* Recording state announcements for screen readers */}
      <div className="sr-only" aria-live="assertive" role="status">
        {isRecording ? `Recording in progress. ${eventCount} events captured.` : ""}
      </div>

      {/* Record Tab */}
      {activeTab === "record" && (
        <div className="tab-content" role="tabpanel" aria-label="Record">
          {/* Permission Warnings */}
          {!hasAccessibility && (
            <div className="alert warning" role="alert">
              Accessibility permission required. Enable in System Settings &gt; Privacy &amp; Security &gt; Accessibility.
            </div>
          )}

          <section className="panel recording-panel">
            {isRecording ? (
              <>
                <div className="recording-header">
                  <span className="recording-indicator">
                    <span className="recording-dot" aria-hidden="true" />
                    Recording
                  </span>
                  <span className="recording-rate" aria-label={`${eventsPerSecond} events per second`}>{eventsPerSecond} evt/s</span>
                </div>

                <div className="recording-stats" aria-label="Recording statistics">
                  <div className="stat stat-large">
                    <span className="stat-value" aria-label={`Duration: ${formatDuration(duration)}`}>{formatDuration(duration)}</span>
                    <span className="stat-label">Duration</span>
                  </div>
                  <div className="stat stat-large">
                    <span className="stat-value" aria-label={`${eventCount} events`}>{eventCount.toLocaleString()}</span>
                    <span className="stat-label">Events</span>
                  </div>
                </div>

                <button className="btn btn-stop btn-full" onClick={handleStopRecording}>
                  Stop Recording
                  <span className="shortcut-hint">&#8963;&#8997;&#8984;S</span>
                </button>
              </>
            ) : (
              <>
                <h2 className="panel-title">New Recording</h2>
                <div className="recording-form">
                  <input
                    type="text"
                    placeholder="Recording name (auto-generated if empty)"
                    value={recordingName}
                    onChange={(e) => setRecordingName(e.target.value)}
                    aria-label="Recording name"
                  />
                  <textarea
                    placeholder="Describe the task you want to demonstrate..."
                    value={goal}
                    onChange={(e) => setGoal(e.target.value)}
                    aria-label="Task description"
                    rows={3}
                  />
                  <button
                    className="btn btn-start btn-full"
                    onClick={handleStartRecording}
                    disabled={!hasAccessibility}
                  >
                    Start Recording
                    <span className="shortcut-hint">Cmd+R</span>
                  </button>
                </div>
              </>
            )}
          </section>
        </div>
      )}

      {/* Recordings Tab */}
      {activeTab === "recordings" && (
        <div className="tab-content" role="tabpanel" aria-label="Recordings">
          {/* Skill Preview */}
          {previewContent && (
            <section className="panel preview-panel">
              <div className="preview-header">
                <h2 className="panel-title">Generated: {previewName}</h2>
                <div className="preview-actions">
                  <button
                    className="btn btn-small"
                    onClick={() => copyToClipboard(previewContent)}
                    aria-label="Copy generated skill to clipboard"
                  >
                    Copy
                  </button>
                  {previewName && (
                    <button
                      className="btn btn-small"
                      onClick={() => {
                        const skillPath = Object.values(generatedSkills).find(p => p.includes(previewName!));
                        if (skillPath) handleDownloadSkill(skillPath, previewName!);
                      }}
                      aria-label="Download generated skill"
                    >
                      Download
                    </button>
                  )}
                  <button
                    className="btn btn-small"
                    onClick={() => { setPreviewContent(null); setPreviewName(null); }}
                    aria-label="Close preview"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
              <MarkdownPreview content={previewContent} />
            </section>
          )}

          {/* Pipeline Stats — compact inline summary */}
          {pipelineStats && (
            <div className="stats-bar">
              <span className="stats-bar-summary">
                {pipelineStats.generated_steps_count} steps
                {pipelineStats.variables_count > 0 && ` / ${pipelineStats.variables_count} vars`}
                {pipelineStats.screenshot_enhanced_count > 0 && ` / ${pipelineStats.screenshot_enhanced_count} enhanced`}
                {pipelineStats.state_diff_enriched_count > 0 && ` / ${pipelineStats.state_diff_enriched_count} diffs`}
                {pipelineStats.error_correction_count > 0 && ` / ${pipelineStats.error_correction_count} filtered`}
                {pipelineStats.warnings.length > 0 && ` / ${pipelineStats.warnings.length} warnings`}
              </span>
              <button className="btn btn-tiny" onClick={() => setPipelineStats(null)}>Dismiss</button>
            </div>
          )}

          {/* Screenshot Review */}
          {reviewingRecording && analysisData && (
            <section className="panel screenshot-review-panel">
              <div className="preview-header">
                <h2 className="panel-title">Review: {reviewingRecording}</h2>
                <div className="preview-actions">
                  <span className="review-stats">
                    {analysisData.total_screenshots} screenshots
                    {analysisData.ocr_count > 0 && ` / ${analysisData.ocr_count} OCR`}
                    {analysisData.vision_count > 0 && ` / ${analysisData.vision_count} Vision`}
                    {analysisData.edited_count > 0 && ` / ${analysisData.edited_count} edited`}
                  </span>
                  <button
                    className="btn btn-generate"
                    onClick={() => {
                      const rec = recordings.find(r => r.name === reviewingRecording);
                      if (rec) {
                        handleCloseReview();
                        handleGenerateSkill(rec.path);
                      }
                    }}
                  >
                    Accept All & Generate
                  </button>
                  <button className="btn btn-small" onClick={handleCloseReview} aria-label="Close review">
                    Close
                  </button>
                </div>
              </div>
              <div className="screenshot-grid" role="list" aria-label="Screenshot analysis">
                {analysisData.analyses.map((item) => (
                  <div
                    key={item.sequence}
                    className="screenshot-row"
                    role="listitem"
                  >
                    <button
                      className="screenshot-thumb-btn"
                      onClick={() => setSelectedScreenshot(item)}
                      aria-label={`Screenshot ${item.sequence}: ${item.intent}`}
                    >
                      <img
                        className="screenshot-thumb"
                        src={convertFileSrc(item.image_path)}
                        alt={`Screenshot ${item.sequence}`}
                        loading="lazy"
                      />
                    </button>
                    <div className="screenshot-row-content">
                      <div className="screenshot-card-meta">
                        <span className="screenshot-seq">#{item.sequence}</span>
                        <span className={`tier-badge tier-${item.tier}`}>
                          {item.tier === "ocr" ? "OCR" : item.tier === "vision" ? "Vision" : "Edited"}
                        </span>
                        {item.user_edited && <span className="edited-badge">Edited</span>}
                      </div>
                      <textarea
                        className="intent-editor"
                        value={item.intent}
                        onChange={(e) => handleUpdateIntent(item.sequence, e.target.value)}
                        aria-label={`Intent for screenshot ${item.sequence}`}
                        rows={3}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Screenshot Modal */}
          {selectedScreenshot && (
            <div
              className="screenshot-modal"
              onClick={() => setSelectedScreenshot(null)}
              onKeyDown={(e) => { if (e.key === "Escape") setSelectedScreenshot(null); }}
              role="dialog"
              aria-label="Screenshot preview"
              tabIndex={-1}
            >
              <div className="screenshot-modal-content" onClick={(e) => e.stopPropagation()}>
                <img
                  src={convertFileSrc(selectedScreenshot.image_path)}
                  alt={`Screenshot ${selectedScreenshot.sequence}`}
                />
                <div className="screenshot-modal-info">
                  <strong>#{selectedScreenshot.sequence}</strong>
                  <span className={`tier-badge tier-${selectedScreenshot.tier}`}>
                    {selectedScreenshot.tier === "ocr" ? "OCR" : selectedScreenshot.tier === "vision" ? "Vision" : "Edited"}
                  </span>
                  <p>{selectedScreenshot.intent}</p>
                </div>
                <button
                  className="btn btn-small screenshot-modal-close"
                  onClick={() => setSelectedScreenshot(null)}
                  aria-label="Close screenshot preview"
                >
                  Close
                </button>
              </div>
            </div>
          )}

          {/* Recordings List */}
          <section className="panel">
            <h2 className="panel-title">
              Recordings
              <button className="btn btn-small" onClick={() => loadRecordings()} disabled={loadingRecordings}>
                {loadingRecordings ? "..." : "Refresh"}
              </button>
            </h2>
            {loadingRecordings && recordings.length === 0 ? (
              <div className="empty-state"><p>Loading recordings...</p></div>
            ) : recordings.length === 0 ? (
              <div className="empty-state">
                <p>No recordings yet</p>
                <p className="empty-hint">
                  Go to the Record tab to create your first recording
                </p>
              </div>
            ) : (
              <ul className="recordings-list">
                {recordings.map((rec) => (
                  <li key={rec.path} className="recording-item">
                    <div className="recording-info">
                      <div className="recording-name-row">
                        <strong>{rec.name}</strong>
                        <span className="recording-date">{formatDate(rec.created_at)}</span>
                      </div>
                      <div className="recording-meta">
                        <span className="meta-tag">{rec.event_count.toLocaleString()} events</span>
                        <span className="meta-tag">{formatDurationMs(rec.duration_ms)}</span>
                        {rec.has_screenshots && <span className="meta-tag meta-tag-screenshot">Screenshots</span>}
                        {rec.has_analysis && <span className="meta-tag meta-tag-analysis">Analyzed</span>}
                      </div>
                      {rec.goal && <p className="recording-goal">{rec.goal}</p>}
                    </div>
                    <div className="recording-actions">
                      {rec.has_screenshots && !rec.has_analysis && (
                        <button
                          className="btn btn-analyze"
                          onClick={() => handleAnalyzeRecording(rec.name)}
                          disabled={analyzing === rec.name}
                        >
                          {analyzing === rec.name ? "Analyzing..." : "Analyze"}
                        </button>
                      )}
                      {rec.has_analysis && (
                        <button
                          className="btn btn-review"
                          onClick={() => handleOpenReview(rec.name)}
                        >
                          Review
                        </button>
                      )}
                      <button
                        className="btn btn-generate"
                        onClick={() => handleGenerateSkill(rec.path)}
                        disabled={generating === rec.path}
                      >
                        {generating === rec.path ? "Generating..." : "Generate"}
                      </button>
                      {generatedSkills[rec.path] && (
                        <button
                          className="btn btn-small"
                          onClick={async () => {
                            const content = await readSkillFile(generatedSkills[rec.path]);
                            if (content) { setPreviewContent(content); setPreviewName(rec.name); }
                          }}
                        >
                          Preview
                        </button>
                      )}
                      {generatedSkills[rec.path] && (
                        <button
                          className="btn btn-small"
                          onClick={async () => {
                            const content = await readSkillFile(generatedSkills[rec.path]);
                            if (content) await copyToClipboard(content, "SKILL.md copied to clipboard");
                          }}
                        >
                          Copy
                        </button>
                      )}
                      <button
                        className="btn btn-small"
                        onClick={() => handleExportRecording(rec.path, rec.name)}
                        disabled={exportingRecording === rec.path}
                      >
                        {exportingRecording === rec.path ? "..." : "Export"}
                      </button>
                      <button
                        className="btn btn-small"
                        onClick={() => handleOpenInFinder(rec.path)}
                      >
                        Finder
                      </button>
                      <button
                        className={`btn btn-small btn-danger${confirmDelete === rec.name ? " confirming" : ""}`}
                        onClick={() => handleDeleteRecording(rec.name)}
                        onBlur={() => setConfirmDelete(null)}
                      >
                        {confirmDelete === rec.name ? "Confirm?" : "Delete"}
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      )}

      {/* Skills Tab */}
      {activeTab === "skills" && (
        <div className="tab-content" role="tabpanel" aria-label="Skills">
          {/* Skill Preview */}
          {skillPreviewContent && (
            <section className="panel preview-panel">
              <div className="preview-header">
                <h2 className="panel-title">Skill: {skillPreviewName}</h2>
                <div className="preview-actions">
                  <button
                    className="btn btn-small"
                    onClick={() => copyToClipboard(skillPreviewContent!)}
                    aria-label="Copy skill to clipboard"
                  >
                    Copy
                  </button>
                  <button
                    className="btn btn-small"
                    onClick={() => { setSkillPreviewContent(null); setSkillPreviewName(null); }}
                    aria-label="Close skill preview"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
              <MarkdownPreview content={skillPreviewContent} />
            </section>
          )}

          <section className="panel">
            <h2 className="panel-title">
              Generated Skills
              <button className="btn btn-small" onClick={() => loadSkills()} disabled={loadingSkills}>
                {loadingSkills ? "..." : "Refresh"}
              </button>
            </h2>
            {loadingSkills && skills.length === 0 ? (
              <div className="empty-state"><p>Loading skills...</p></div>
            ) : skills.length === 0 ? (
              <div className="empty-state">
                <p>No skills generated yet</p>
                <p className="empty-hint">
                  Generate skills from recordings to see them here
                </p>
              </div>
            ) : (
              <ul className="recordings-list">
                {skills.map((skill) => (
                  <li key={skill.path} className="recording-item">
                    <div className="recording-info">
                      <div className="recording-name-row">
                        <strong>{skill.name}</strong>
                        <span className="recording-date">{formatDate(skill.modified_at)}</span>
                      </div>
                      <div className="recording-meta">
                        <span className="meta-tag">{skill.category}</span>
                        <span className="meta-tag">{formatSkillSize(skill.size_bytes)}</span>
                      </div>
                    </div>
                    <div className="recording-actions">
                      <button
                        className="btn btn-small"
                        onClick={async () => {
                          const content = await readSkillFile(skill.path);
                          if (content) { setSkillPreviewContent(content); setSkillPreviewName(skill.name); }
                        }}
                      >
                        Preview
                      </button>
                      <button
                        className="btn btn-small"
                        onClick={async () => {
                          const content = await readSkillFile(skill.path);
                          if (content) await copyToClipboard(content);
                        }}
                      >
                        Copy
                      </button>
                      <button
                        className="btn btn-small"
                        onClick={() => handleOpenInFinder(skill.path)}
                      >
                        Finder
                      </button>
                      <button
                        className={`btn btn-small btn-danger${confirmDeleteSkill === skill.path ? " confirming" : ""}`}
                        onClick={() => handleDeleteSkill(skill.path, skill.name)}
                        onBlur={() => setConfirmDeleteSkill(null)}
                      >
                        {confirmDeleteSkill === skill.path ? "Confirm?" : "Delete"}
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      )}

      {/* Settings Tab */}
      {activeTab === "settings" && (
        <div className="tab-content" role="tabpanel" aria-label="Settings">
          {/* API Key Section */}
          <section className="panel">
            <h2 className="panel-title">Anthropic API Key</h2>
            {maskedApiKey && (
              <p className="current-key">Current: <code>{maskedApiKey}</code></p>
            )}
            <div className="input-row">
              <input
                type="password"
                placeholder="sk-ant-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                aria-label="Anthropic API key"
                onKeyDown={(e) => { if (e.key === "Enter" && apiKey) handleSaveApiKey(); }}
              />
              <button className="btn btn-primary" onClick={handleSaveApiKey} disabled={!apiKey}>
                Save
              </button>
            </div>
            <p className="help-text">Required for AI-powered skill naming and variable inference.</p>
          </section>

          {/* Generator Config */}
          {config && (
            <section className="panel">
              <h2 className="panel-title">Generator Configuration</h2>
              <div className="config-form">
                <label>
                  <span className="config-label">Trajectory Simplification (RDP epsilon, px)</span>
                  <input
                    type="number"
                    step="0.5"
                    min="0.5"
                    max="20"
                    value={config.rdp_epsilon_px}
                    onChange={(e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, rdp_epsilon_px: v }); }}
                  />
                </label>
                <label>
                  <span className="config-label">Hesitation Threshold (0-1)</span>
                  <input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    value={config.hesitation_threshold}
                    onChange={(e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, hesitation_threshold: v }); }}
                  />
                </label>
                <label>
                  <span className="config-label">Min Pause for Chunk Boundary (ms)</span>
                  <input
                    type="number"
                    step="50"
                    min="100"
                    max="2000"
                    value={config.min_pause_ms}
                    onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, min_pause_ms: v }); }}
                  />
                </label>
                <label>
                  <span className="config-label">Inference Model</span>
                  <select
                    value={config.model}
                    onChange={(e) => setConfig({ ...config, model: e.target.value })}
                  >
                    {MODEL_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span className="config-label">Vision Model</span>
                  <select
                    value={config.vision_model}
                    onChange={(e) => setConfig({ ...config, vision_model: e.target.value })}
                  >
                    {MODEL_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span className="config-label">Temperature (0-2)</span>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="2"
                    value={config.temperature}
                    onChange={(e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, temperature: v }); }}
                  />
                </label>

                <h3 className="config-section-title">Pipeline Features</h3>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={config.use_action_clustering}
                    onChange={(e) => setConfig({ ...config, use_action_clustering: e.target.checked })}
                  />
                  <span>Action Clustering</span>
                  <span className="config-hint">Group events into UnitTasks before step generation</span>
                </label>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={config.use_local_recovery}
                    onChange={(e) => setConfig({ ...config, use_local_recovery: e.target.checked })}
                  />
                  <span>Local Recovery</span>
                  <span className="config-hint">AX retry + spiral search for missing semantic data</span>
                </label>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={config.use_vision_ocr}
                    onChange={(e) => setConfig({ ...config, use_vision_ocr: e.target.checked })}
                  />
                  <span>Vision OCR</span>
                  <span className="config-hint">Apple Vision framework fallback for text recognition</span>
                </label>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={config.use_trajectory_analysis}
                    onChange={(e) => setConfig({ ...config, use_trajectory_analysis: e.target.checked })}
                  />
                  <span>Trajectory Analysis</span>
                  <span className="config-hint">RDP simplification + Fitts' Law kinematic profiling</span>
                </label>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={config.use_goms_detection}
                    onChange={(e) => setConfig({ ...config, use_goms_detection: e.target.checked })}
                  />
                  <span>GOMS Detection</span>
                  <span className="config-hint">Cognitive boundary analysis via mental operator pauses</span>
                </label>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={config.use_context_tracking}
                    onChange={(e) => setConfig({ ...config, use_context_tracking: e.target.checked })}
                  />
                  <span>Context Tracking</span>
                  <span className="config-hint">Track window/app focus changes during generation</span>
                </label>
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

                <h3 className="config-section-title">Recording Features</h3>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={config.capture_screenshots}
                    onChange={(e) => setConfig({ ...config, capture_screenshots: e.target.checked })}
                  />
                  <span>Screenshot Capture</span>
                  <span className="config-hint">Capture screenshots on significant actions for AI analysis</span>
                </label>

                <h3
                  className="config-section-title config-section-collapsible"
                  onClick={() => setShowAnnotationSettings(!showAnnotationSettings)}
                >
                  {showAnnotationSettings ? "▾" : "▸"} Visual Annotation
                </h3>
                {showAnnotationSettings && (
                  <div className="annotation-settings">
                    <label>
                      <span className="config-label">Dot Radius ({config.dot_radius}px)</span>
                      <input
                        type="range"
                        min="4"
                        max="24"
                        value={config.dot_radius}
                        onChange={(e) => setConfig({ ...config, dot_radius: parseInt(e.target.value) })}
                      />
                    </label>
                    <label>
                      <span className="config-label">Dot Color</span>
                      <div className="color-inputs">
                        <input
                          type="number"
                          min="0"
                          max="255"
                          value={config.dot_color[0]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, dot_color: [v, config.dot_color[1], config.dot_color[2]] }); }}
                          aria-label="Red"
                        />
                        <input
                          type="number"
                          min="0"
                          max="255"
                          value={config.dot_color[1]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, dot_color: [config.dot_color[0], v, config.dot_color[2]] }); }}
                          aria-label="Green"
                        />
                        <input
                          type="number"
                          min="0"
                          max="255"
                          value={config.dot_color[2]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, dot_color: [config.dot_color[0], config.dot_color[1], v] }); }}
                          aria-label="Blue"
                        />
                        <span className="color-preview" style={{ backgroundColor: `rgb(${config.dot_color[0]}, ${config.dot_color[1]}, ${config.dot_color[2]})` }} />
                      </div>
                    </label>
                    <label>
                      <span className="config-label">Ballistic Trajectory Color</span>
                      <div className="color-inputs">
                        <input
                          type="number"
                          min="0"
                          max="255"
                          value={config.trajectory_ballistic_color[0]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, trajectory_ballistic_color: [v, config.trajectory_ballistic_color[1], config.trajectory_ballistic_color[2]] }); }}
                          aria-label="Red"
                        />
                        <input
                          type="number"
                          min="0"
                          max="255"
                          value={config.trajectory_ballistic_color[1]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, trajectory_ballistic_color: [config.trajectory_ballistic_color[0], v, config.trajectory_ballistic_color[2]] }); }}
                          aria-label="Green"
                        />
                        <input
                          type="number"
                          min="0"
                          max="255"
                          value={config.trajectory_ballistic_color[2]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, trajectory_ballistic_color: [config.trajectory_ballistic_color[0], config.trajectory_ballistic_color[1], v] }); }}
                          aria-label="Blue"
                        />
                        <span className="color-preview" style={{ backgroundColor: `rgb(${config.trajectory_ballistic_color[0]}, ${config.trajectory_ballistic_color[1]}, ${config.trajectory_ballistic_color[2]})` }} />
                      </div>
                    </label>
                    <label>
                      <span className="config-label">Searching Trajectory Color</span>
                      <div className="color-inputs">
                        <input
                          type="number"
                          min="0"
                          max="255"
                          value={config.trajectory_searching_color[0]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, trajectory_searching_color: [v, config.trajectory_searching_color[1], config.trajectory_searching_color[2]] }); }}
                          aria-label="Red"
                        />
                        <input
                          type="number"
                          min="0"
                          max="255"
                          value={config.trajectory_searching_color[1]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, trajectory_searching_color: [config.trajectory_searching_color[0], v, config.trajectory_searching_color[2]] }); }}
                          aria-label="Green"
                        />
                        <input
                          type="number"
                          min="0"
                          max="255"
                          value={config.trajectory_searching_color[2]}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!Number.isNaN(v)) setConfig({ ...config, trajectory_searching_color: [config.trajectory_searching_color[0], config.trajectory_searching_color[1], v] }); }}
                          aria-label="Blue"
                        />
                        <span className="color-preview" style={{ backgroundColor: `rgb(${config.trajectory_searching_color[0]}, ${config.trajectory_searching_color[1]}, ${config.trajectory_searching_color[2]})` }} />
                      </div>
                    </label>
                    <label>
                      <span className="config-label">Trajectory Thickness ({config.trajectory_thickness}px)</span>
                      <input
                        type="range"
                        min="1"
                        max="5"
                        value={config.trajectory_thickness}
                        onChange={(e) => setConfig({ ...config, trajectory_thickness: parseInt(e.target.value) })}
                      />
                    </label>
                    <label>
                      <span className="config-label">Crop Size ({config.crop_size}px)</span>
                      <input
                        type="range"
                        min="256"
                        max="1024"
                        step="64"
                        value={config.crop_size}
                        onChange={(e) => setConfig({ ...config, crop_size: parseInt(e.target.value) })}
                      />
                    </label>
                  </div>
                )}

                <button className="btn btn-primary" onClick={handleSaveConfig} disabled={savingConfig}>
                  {savingConfig ? "Saving..." : "Save Configuration"}
                </button>

                <div className="config-export-row">
                  <button className="btn btn-small" onClick={handleExportConfig} disabled={exportingConfig}>
                    {exportingConfig ? "Exporting..." : "Export Config"}
                  </button>
                  <button className="btn btn-small" onClick={handleImportConfig}>
                    Import Config
                  </button>
                </div>
              </div>
            </section>
          )}

          {/* Keyboard Shortcuts */}
          <section className="panel">
            <h2 className="panel-title">Keyboard Shortcuts</h2>
            <dl className="shortcuts-list">
              <div className="shortcut-row">
                <dt><kbd>Cmd</kbd> + <kbd>R</kbd></dt>
                <dd>Start recording</dd>
              </div>
              <div className="shortcut-row">
                <dt><kbd>Esc</kbd></dt>
                <dd>Stop recording</dd>
              </div>
              <div className="shortcut-row">
                <dt><kbd>Enter</kbd></dt>
                <dd>Save API key (when focused)</dd>
              </div>
            </dl>
          </section>

          {/* About */}
          <section className="panel panel-muted">
            <h2 className="panel-title">About</h2>
            <p className="about-text">
              Skill Generator records macOS user interactions and generates Claude Code SKILL.md files.
              Uses GOMS cognitive model for action chunking, Fitts' Law for kinematic analysis,
              and Hoare triple verification for deterministic postconditions.
            </p>
            <div className="about-meta">
              <span>v0.1.0</span>
              <span>748 tests passing</span>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}

function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  );
}

export default AppWithErrorBoundary;
