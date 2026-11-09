use std::net::IpAddr;
use std::path::PathBuf;
use std::thread::available_parallelism;

use tracing_subscriber::EnvFilter;

use crate::error::ExitError;
use crate::inference::{self, LoadOptions};
use crate::paths::{self, CurrentModel};
use crate::server;

pub struct Request {
    pub model: Option<PathBuf>,
    pub host: IpAddr,
    pub port: u16,
    pub ctx: u32,
    pub threads: Option<u16>,
    pub gpu_layers: u32,
    pub vulkan_device: Option<u32>,
}

pub async fn run(request: Request) -> Result<(), ExitError> {
    if request.ctx == 0 {
        return Err(ExitError::usage("ctx must be at least 1"));
    }
    if request.threads == Some(0) {
        return Err(ExitError::usage("threads must be at least 1"));
    }

    let dir = paths::models_dir(None);
    let current = paths::read_current(&dir).map_err(ExitError::usage)?;
    let model_path = resolve_model(request.model, current.as_ref())?;
    let model_id = paths::model_id(&model_path, current.as_ref());
    let threads = request
        .threads
        .map(i32::from)
        .unwrap_or_else(default_threads);

    init_tracing();
    let (engine, loaded) = inference::start(LoadOptions {
        model_path,
        model_id,
        context_tokens: request.ctx,
        threads,
        gpu_layers: request.gpu_layers,
        vulkan_device: request.vulkan_device,
    })
    .map_err(|err| ExitError::model(err.to_string()))?;

    eprintln!("model: {}", loaded.path.display());
    eprintln!("device: {}", loaded.device);
    server::listen(engine, loaded.id, request.host, request.port).await
}

fn resolve_model(
    flag: Option<PathBuf>,
    current: Option<&CurrentModel>,
) -> Result<PathBuf, ExitError> {
    let path = match flag {
        Some(path) => path,
        None => current.map(|current| current.path.clone()).ok_or_else(|| {
            ExitError::usage(
                "no model is selected. Run download, or pass --model with a .gguf path.",
            )
        })?,
    };
    if !path.is_file() {
        return Err(ExitError::usage(format!(
            "model file not found: {}. Run download, or pass --model with a .gguf path.",
            path.display()
        )));
    }
    Ok(path)
}

fn default_threads() -> i32 {
    available_parallelism()
        .map(|count| count.get() as i32)
        .unwrap_or(4)
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();
}
