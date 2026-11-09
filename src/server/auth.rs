use std::sync::Arc;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use super::openai;

/// Environment variable (or `.env` entry) that turns API key checking on.
pub const API_KEY_ENV: &str = "QWEN_API_KEY";

/// The configured key. `None` leaves the API open.
pub fn load_api_key() -> Option<String> {
    let value = std::env::var(API_KEY_ENV).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

/// Reject requests whose `Authorization: Bearer` token does not match the key.
pub async fn require_api_key(
    State(key): State<Arc<String>>,
    request: Request,
    next: Next,
) -> Response {
    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(bearer_token);
    match token {
        Some(token) if same_secret(token.as_bytes(), key.as_bytes()) => next.run(request).await,
        _ => unauthorized(),
    }
}

fn bearer_token(header: &str) -> Option<&str> {
    let (scheme, token) = header.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

/// Compare without stopping at the first differing byte.
fn same_secret(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |diff, (a, b)| diff | (a ^ b))
        == 0
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        Json(openai::ErrorResponse {
            error: openai::ApiErrorBody {
                message: "missing or invalid API key".into(),
                error_type: "invalid_request_error".into(),
                code: "invalid_api_key".into(),
            },
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bearer_header() {
        assert_eq!(bearer_token("Bearer abc"), Some("abc"));
        assert_eq!(bearer_token("bearer abc"), Some("abc"));
        assert_eq!(bearer_token("Basic abc"), None);
        assert_eq!(bearer_token("abc"), None);
    }

    #[test]
    fn compares_secrets() {
        assert!(same_secret(b"key", b"key"));
        assert!(!same_secret(b"key", b"kex"));
        assert!(!same_secret(b"key", b"keys"));
    }
}
