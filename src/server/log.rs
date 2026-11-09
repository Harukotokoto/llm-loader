use std::net::SocketAddr;
use std::time::Instant;

use axum::extract::{ConnectInfo, Request};
use axum::middleware::Next;
use axum::response::Response;

/// Log each request on arrival and its status when the response head is ready.
///
/// Headers and bodies are never logged, so API keys and prompts stay out of the log.
/// For streamed replies the elapsed time covers only the wait for the first byte.
pub async fn log_request(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    tracing::info!(%peer, %method, %path, "request");

    let started = Instant::now();
    let response = next.run(request).await;
    tracing::info!(
        %peer,
        %method,
        %path,
        status = response.status().as_u16(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "response"
    );
    response
}
