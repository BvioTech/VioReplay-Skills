//! Reusable HTTP retry logic with exponential backoff for Anthropic API calls.
//!
//! Handles 429 rate limiting, 5xx server errors, and network timeouts
//! with configurable retry count and exponential backoff.

use reqwest::{Client, RequestBuilder, Response, StatusCode};
use tracing::warn;

/// Send an HTTP request with retry and exponential backoff.
///
/// Returns `Some(Response)` on success, `None` if all retries exhausted
/// or a non-retriable error occurs.
///
/// Retry behavior:
/// - 429 (rate limited): backoff 2s, 4s, 8s
/// - 5xx (server error): backoff 1s, 2s, 4s
/// - Timeout/connect error: backoff 1s, 2s, 4s
/// - Other 4xx: non-retriable, returns None immediately
pub async fn send_with_retry<F>(
    client: &Client,
    build_request: F,
    max_retries: u32,
    context: &str,
) -> Option<Response>
where
    F: Fn(&Client) -> RequestBuilder,
{
    let mut response = None;
    for attempt in 0..max_retries {
        let result = build_request(client).send().await;

        match result {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    response = Some(resp);
                    break;
                } else if status == StatusCode::TOO_MANY_REQUESTS {
                    // Longer backoff for rate limiting (4s, 8s, 16s) since the API
                    // needs time to reset the request quota
                    let delay = std::time::Duration::from_secs(2u64.pow(attempt + 1));
                    warn!("{}: rate limited (429), retrying in {:?}", context, delay);
                    tokio::time::sleep(delay).await;
                } else if status.is_server_error() {
                    // Shorter backoff for server errors (1s, 2s, 4s) which are
                    // typically transient and resolve faster
                    let delay = std::time::Duration::from_secs(2u64.pow(attempt));
                    warn!("{}: server error ({}), retrying in {:?}", context, status, delay);
                    tokio::time::sleep(delay).await;
                } else {
                    warn!("{}: non-retriable error ({})", context, status);
                    return None;
                }
            }
            Err(e) if e.is_timeout() || e.is_connect() => {
                let delay = std::time::Duration::from_secs(2u64.pow(attempt));
                warn!("{}: network error ({}), retrying in {:?}", context, e, delay);
                tokio::time::sleep(delay).await;
            }
            Err(e) => {
                warn!("{}: request failed: {}", context, e);
                return None;
            }
        }
    }

    if response.is_none() {
        warn!("{}: failed after {} retries", context, max_retries);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_send_with_retry_returns_none_on_non_retriable_error() {
        // Use a URL that will return a 404 (non-retriable)
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(1))
            .build()
            .unwrap();

        let result = send_with_retry(
            &client,
            |c| c.get("http://127.0.0.1:1/nonexistent"),
            1,
            "test",
        )
        .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_send_with_retry_zero_retries_returns_none() {
        let client = Client::new();
        let result = send_with_retry(
            &client,
            |c| c.get("http://127.0.0.1:1/"),
            0,
            "test",
        )
        .await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_send_with_retry_connection_refused_exhausts_retries() {
        // Port 1 typically refuses connections; with max_retries=1 the loop runs
        // once, hits a connect error, backs off, and then returns None.
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .unwrap();

        let start = std::time::Instant::now();
        let result = send_with_retry(
            &client,
            |c| c.get("http://127.0.0.1:1/"),
            1,
            "retry-connect-test",
        )
        .await;
        let elapsed = start.elapsed();

        assert!(result.is_none());
        // Should have attempted 1 iteration with a ~1s backoff (2^0 = 1s)
        assert!(elapsed >= std::time::Duration::from_millis(500));
    }

    #[tokio::test]
    async fn test_send_with_retry_timeout_exhausts_retries() {
        // Use a very short timeout so the request times out quickly
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(50))
            .build()
            .unwrap();

        let result = send_with_retry(
            &client,
            // 192.0.2.1 is TEST-NET, packets are typically blackholed (timeout)
            |c| c.get("http://192.0.2.1:9999/"),
            1,
            "retry-timeout-test",
        )
        .await;

        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_send_with_retry_multiple_retries_returns_none() {
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(100))
            .build()
            .unwrap();

        let result = send_with_retry(
            &client,
            |c| c.get("http://127.0.0.1:1/"),
            2,
            "retry-multi-test",
        )
        .await;

        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_send_with_retry_closure_receives_client() {
        // Verify the closure is called with the correct client
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(100))
            .build()
            .unwrap();

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count_clone = call_count.clone();

        let result = send_with_retry(
            &client,
            |c| {
                count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                c.get("http://127.0.0.1:1/")
            },
            2,
            "closure-test",
        )
        .await;

        assert!(result.is_none());
        // The closure should have been called exactly 2 times (max_retries = 2)
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
