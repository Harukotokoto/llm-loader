mod openai;
mod routes;

use std::net::{IpAddr, SocketAddr};

use tokio::net::TcpListener;

use crate::error::ExitError;
use crate::inference::Engine;

pub async fn listen(
    engine: Engine,
    model_id: String,
    host: IpAddr,
    port: u16,
) -> Result<(), ExitError> {
    let addr = SocketAddr::new(host, port);
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|err| ExitError::bind(format!("failed to bind {addr}: {err}")))?;
    let bound = listener.local_addr().unwrap_or(addr);
    eprintln!("listening: http://{bound}");
    eprintln!("docs: http://{bound}/docs");
    eprintln!("openapi: http://{bound}/openapi.json");

    let app = routes::router(routes::AppState { engine, model_id });
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("shutting down");
        })
        .await
        .map_err(|err| ExitError::usage(format!("http server stopped: {err}")))?;
    Ok(())
}
