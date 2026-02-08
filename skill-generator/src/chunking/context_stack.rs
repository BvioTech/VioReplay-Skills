//! OS Context Tracking
//!
//! Monitors window focus and UI context changes to help identify task boundaries.

use std::collections::VecDeque;

/// Window context information
#[derive(Debug, Clone)]
pub struct WindowContext {
    /// Window ID
    pub window_id: u32,
    /// Window title
    pub title: String,
    /// Application bundle ID
    pub app_bundle_id: String,
    /// Application name
    pub app_name: String,
    /// Z-index (stacking order)
    pub z_index: u32,
    /// Timestamp when this became active
    pub activated_at: u64,
}

/// Context transition event
#[derive(Debug, Clone)]
pub enum ContextTransition {
    /// New window focused
    WindowFocused(WindowContext),
    /// New window created
    WindowCreated(WindowContext),
    /// Window closed
    WindowClosed(u32),
    /// Modal dialog opened
    ModalOpened(WindowContext),
    /// Modal dialog closed
    ModalClosed(u32),
    /// Application switched
    AppSwitched {
        from: String,
        to: String,
    },
}

/// Context stack for tracking UI hierarchy
pub struct ContextStack {
    /// Stack of active window contexts
    stack: VecDeque<WindowContext>,
    /// History of transitions
    history: Vec<(u64, ContextTransition)>,
    /// Maximum stack depth
    max_depth: usize,
    /// Maximum history length
    max_history: usize,
}

impl ContextStack {
    /// Create a new context stack
    pub fn new() -> Self {
        Self {
            stack: VecDeque::new(),
            history: Vec::new(),
            max_depth: 10,
            max_history: 1000,
        }
    }

    /// Push a new window context
    pub fn push(&mut self, context: WindowContext, timestamp: u64) {
        // Record transition
        self.record_transition(timestamp, ContextTransition::WindowFocused(context.clone()));

        // Manage stack depth
        if self.stack.len() >= self.max_depth {
            self.stack.pop_front();
        }

        self.stack.push_back(context);
    }

    /// Pop the current window context
    pub fn pop(&mut self, timestamp: u64) -> Option<WindowContext> {
        if let Some(context) = self.stack.pop_back() {
            self.record_transition(timestamp, ContextTransition::WindowClosed(context.window_id));
            Some(context)
        } else {
            None
        }
    }

    /// Get current (top) context
    pub fn current(&self) -> Option<&WindowContext> {
        self.stack.back()
    }

    /// Get current depth (nesting level)
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Check if we're in a modal context
    pub fn is_modal(&self) -> bool {
        self.stack.len() > 1
    }

    /// Handle a window focus change
    pub fn on_window_focused(&mut self, context: WindowContext, timestamp: u64) {
        // Check if this is a new window or refocus
        let is_new = !self.stack.iter().any(|w| w.window_id == context.window_id);

        if is_new {
            self.push(context, timestamp);
        } else {
            // Reorder stack to bring this window to top
            let idx = self.stack.iter().position(|w| w.window_id == context.window_id);
            if let Some(i) = idx {
                if let Some(mut ctx) = self.stack.remove(i) {
                    ctx.activated_at = timestamp;
                    self.stack.push_back(ctx);
                }
            }

            if let Some(top) = self.stack.back().cloned() {
                self.record_transition(timestamp, ContextTransition::WindowFocused(top));
            }
        }
    }

    /// Handle a window close
    pub fn on_window_closed(&mut self, window_id: u32, timestamp: u64) {
        if let Some(idx) = self.stack.iter().position(|w| w.window_id == window_id) {
            self.stack.remove(idx);
            self.record_transition(timestamp, ContextTransition::WindowClosed(window_id));
        }
    }

    /// Handle app switch
    pub fn on_app_switched(&mut self, from_bundle: &str, to_bundle: &str, timestamp: u64) {
        self.record_transition(timestamp, ContextTransition::AppSwitched {
            from: from_bundle.to_string(),
            to: to_bundle.to_string(),
        });
    }

    /// Get transitions in a time range
    pub fn get_transitions(&self, start: u64, end: u64) -> Vec<(u64, ContextTransition)> {
        self.history
            .iter()
            .filter(|(t, _)| *t >= start && *t <= end)
            .cloned()
            .collect()
    }

    /// Get all window contexts
    pub fn all_contexts(&self) -> impl Iterator<Item = &WindowContext> {
        self.stack.iter()
    }

    /// Record a transition in history
    fn record_transition(&mut self, timestamp: u64, transition: ContextTransition) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push((timestamp, transition));
    }

    /// Clear the stack
    pub fn clear(&mut self) {
        self.stack.clear();
    }
}

impl Default for ContextStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context(id: u32, title: &str) -> WindowContext {
        WindowContext {
            window_id: id,
            title: title.to_string(),
            app_bundle_id: "com.test".to_string(),
            app_name: "Test".to_string(),
            z_index: 0,
            activated_at: 0,
        }
    }

    #[test]
    fn test_push_pop() {
        let mut stack = ContextStack::new();

        stack.push(make_context(1, "Window 1"), 100);
        assert_eq!(stack.depth(), 1);

        stack.push(make_context(2, "Window 2"), 200);
        assert_eq!(stack.depth(), 2);

        let popped = stack.pop(300);
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().window_id, 2);
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn test_current() {
        let mut stack = ContextStack::new();

        assert!(stack.current().is_none());

        stack.push(make_context(1, "Window 1"), 100);
        assert_eq!(stack.current().unwrap().window_id, 1);
    }

    #[test]
    fn test_is_modal() {
        let mut stack = ContextStack::new();

        assert!(!stack.is_modal());

        stack.push(make_context(1, "Main"), 100);
        assert!(!stack.is_modal());

        stack.push(make_context(2, "Dialog"), 200);
        assert!(stack.is_modal());
    }

    #[test]
    fn test_max_depth_limit() {
        let mut stack = ContextStack::new();

        // Push more than max_depth windows
        for i in 0..15 {
            stack.push(make_context(i, &format!("Window {}", i)), i as u64 * 100);
        }

        // Should be limited to max_depth (10)
        assert_eq!(stack.depth(), 10);

        // Most recent window should be on top
        assert_eq!(stack.current().unwrap().window_id, 14);
    }

    #[test]
    fn test_pop_empty_stack() {
        let mut stack = ContextStack::new();

        let popped = stack.pop(100);
        assert!(popped.is_none());
    }

    #[test]
    fn test_on_window_focused_new_window() {
        let mut stack = ContextStack::new();

        stack.on_window_focused(make_context(1, "Window 1"), 100);
        assert_eq!(stack.depth(), 1);
        assert_eq!(stack.current().unwrap().window_id, 1);

        // Focus a new window
        stack.on_window_focused(make_context(2, "Window 2"), 200);
        assert_eq!(stack.depth(), 2);
        assert_eq!(stack.current().unwrap().window_id, 2);
    }

    #[test]
    fn test_on_window_focused_existing_window() {
        let mut stack = ContextStack::new();

        stack.push(make_context(1, "Window 1"), 100);
        stack.push(make_context(2, "Window 2"), 200);
        stack.push(make_context(3, "Window 3"), 300);

        assert_eq!(stack.depth(), 3);
        assert_eq!(stack.current().unwrap().window_id, 3);

        // Refocus window 1
        stack.on_window_focused(make_context(1, "Window 1"), 400);

        // Should still have 3 windows
        assert_eq!(stack.depth(), 3);

        // Window 1 should be on top now
        assert_eq!(stack.current().unwrap().window_id, 1);
        assert_eq!(stack.current().unwrap().activated_at, 400);
    }

    #[test]
    fn test_on_window_closed() {
        let mut stack = ContextStack::new();

        stack.push(make_context(1, "Window 1"), 100);
        stack.push(make_context(2, "Window 2"), 200);
        stack.push(make_context(3, "Window 3"), 300);

        assert_eq!(stack.depth(), 3);

        // Close middle window
        stack.on_window_closed(2, 400);
        assert_eq!(stack.depth(), 2);

        // Verify window 2 is gone
        assert!(!stack.all_contexts().any(|w| w.window_id == 2));
    }

    #[test]
    fn test_on_window_closed_nonexistent() {
        let mut stack = ContextStack::new();

        stack.push(make_context(1, "Window 1"), 100);
        assert_eq!(stack.depth(), 1);

        // Try to close nonexistent window
        stack.on_window_closed(999, 200);

        // Stack should be unchanged
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn test_on_app_switched() {
        let mut stack = ContextStack::new();

        stack.on_app_switched("com.apple.Safari", "com.apple.TextEdit", 100);

        let transitions = stack.get_transitions(0, 1000);
        assert_eq!(transitions.len(), 1);

        match &transitions[0].1 {
            ContextTransition::AppSwitched { from, to } => {
                assert_eq!(from, "com.apple.Safari");
                assert_eq!(to, "com.apple.TextEdit");
            }
            _ => panic!("Expected AppSwitched transition"),
        }
    }

    #[test]
    fn test_get_transitions_in_range() {
        let mut stack = ContextStack::new();

        stack.push(make_context(1, "Window 1"), 100);
        stack.push(make_context(2, "Window 2"), 200);
        stack.push(make_context(3, "Window 3"), 300);

        let transitions = stack.get_transitions(150, 250);
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].0, 200);
    }

    #[test]
    fn test_get_transitions_all() {
        let mut stack = ContextStack::new();

        stack.push(make_context(1, "Window 1"), 100);
        stack.push(make_context(2, "Window 2"), 200);

        let transitions = stack.get_transitions(0, u64::MAX);
        assert_eq!(transitions.len(), 2);
    }

    #[test]
    fn test_all_contexts_iterator() {
        let mut stack = ContextStack::new();

        stack.push(make_context(1, "Window 1"), 100);
        stack.push(make_context(2, "Window 2"), 200);
        stack.push(make_context(3, "Window 3"), 300);

        let window_ids: Vec<u32> = stack.all_contexts().map(|w| w.window_id).collect();
        assert_eq!(window_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_clear_stack() {
        let mut stack = ContextStack::new();

        stack.push(make_context(1, "Window 1"), 100);
        stack.push(make_context(2, "Window 2"), 200);

        assert_eq!(stack.depth(), 2);

        stack.clear();

        assert_eq!(stack.depth(), 0);
        assert!(stack.current().is_none());
    }

    #[test]
    fn test_history_limit() {
        let mut stack = ContextStack::new();

        // Push more transitions than max_history
        for i in 0..1500 {
            stack.push(make_context(i, &format!("Window {}", i)), i as u64);
        }

        // History should be limited
        assert!(stack.history.len() <= stack.max_history);
    }

    #[test]
    fn test_default_context_stack() {
        let stack = ContextStack::default();
        assert_eq!(stack.depth(), 0);
        assert_eq!(stack.max_depth, 10);
        assert_eq!(stack.max_history, 1000);
    }
}
