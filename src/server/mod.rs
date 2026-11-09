mod auth;
mod openai;
mod routes;

use std::net::{IpAddr, SocketAddr};

use tokio::net::TcpListener;

use crate::error::ExitError;
use crate::inference::Engine;

pub use auth::load_api_key;

pub async fn listen(
    engine: Engine,
    model_id: String,
    host: IpAddr,
    port: u16,
    api_key: Option<String>,
) -> Result<(), ExitError> {
    let addr = SocketAddr::new(host, port);
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|err| ExitError::bind(format!("failed to bind {addr}: {err}")))?;
    let bound = listener.local_addr().unwrap_or(addr);
    eprintln!("listening: http://{bound}");
    eprintln!("docs: http://{bound}/docs");
    eprintln!("openapi: http://{bound}/openapi.json");

    let auth_state = if api_key.is_some() { "required" } else { "off" };
    eprintln!("api key: {auth_state}");

    let app = routes::router(routes::AppState { engine, model_id }, api_key);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("shutting down");
        })
        .await
        .map_err(|err| ExitError::usage(format!("http server stopped: {err}")))?;
    Ok(())
}
