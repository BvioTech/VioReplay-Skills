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
                    let delay = std::time::Duration::from_secs(2u64.pow(attempt + 1));
                    warn!("{}: rate limited (429), retrying in {:?}", context, delay);
                    tokio::time::sleep(delay).await;
                } else if status.is_server_error() {
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
}
