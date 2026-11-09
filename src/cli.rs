use std::net::IpAddr;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;

use clap::{Parser, Subcommand, error::ErrorKind};

use crate::commands::{download, models, serve};
use crate::error::ExitError;
use crate::inference::ALL_GPU_LAYERS;

const EXAMPLES: &str = "\
Examples:
  llm download Qwen/Qwen3-8B-GGUF
  llm download Qwen/Qwen3-8B-GGUF --file Q4_K_M
  llm serve
  llm models";

#[derive(Debug, Parser)]
#[command(
    name = "llm",
    version,
    about = "Download a GGUF model and serve an OpenAI-compatible API",
    arg_required_else_help = true,
    after_help = EXAMPLES
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Download one GGUF file from a Hugging Face repository.
    Download {
        /// Repository id (`owner/name`) or a huggingface.co URL.
        repo: String,
        /// File name ending in `.gguf`, or a quant token such as `Q4_K_M`.
        ///
        /// When omitted, a single `Q4_K_M` file is chosen.
        #[arg(short, long)]
        file: Option<String>,
        /// Git revision. Branch, tag, or commit.
        #[arg(long, default_value = "main")]
        revision: String,
        /// Directory that stores models and `current.json`.
        ///
        /// Defaults to `QWEN_MODELS_DIR`, or `./models`.
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Load the current model and serve the OpenAI-compatible API.
    Serve {
        /// GGUF file to load. Defaults to the model recorded by `download`.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Address to listen on.
        #[arg(long, default_value = "127.0.0.1")]
        host: IpAddr,
        /// Port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Context length, in tokens.
        #[arg(long, default_value_t = 4096)]
        ctx: u32,
        /// Compute threads. Defaults to the available CPU count.
        #[arg(long)]
        threads: Option<u16>,
        /// Layers to place on the Vulkan device. `all` offloads every layer.
        #[arg(long, default_value = "all")]
        gpu_layers: GpuLayers,
        /// Vulkan device index reported by `vulkaninfo`.
        ///
        /// Sets `GGML_VK_VISIBLE_DEVICES` before the backend starts.
        #[arg(long)]
        vulkan_device: Option<u32>,
    },
    /// List GGUF files under the models directory.
    Models,
}

/// Layer count passed to llama.cpp. `all` becomes a count that covers every layer.
#[derive(Debug, Clone, Copy)]
pub struct GpuLayers {
    count: u32,
}

impl FromStr for GpuLayers {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("all") {
            return Ok(Self {
                count: ALL_GPU_LAYERS,
            });
        }
        let count: u32 = value
            .parse()
            .map_err(|_| format!("expected a layer count or all, got {value}"))?;
        Ok(Self { count })
    }
}

pub fn run() -> Result<(), ExitError> {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(err) => {
            let _ = err.print();
            let code = match err.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => 0,
                _ => 1,
            };
            process::exit(code);
        }
    };

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|err| ExitError::usage(format!("failed to start the async runtime: {err}")))?;

    match cli.command {
        Command::Download {
            repo,
            file,
            revision,
            dir,
        } => runtime.block_on(download::run(download::Request {
            repo,
            file,
            revision,
            dir,
        })),
        Command::Serve {
            model,
            host,
            port,
            ctx,
            threads,
            gpu_layers,
            vulkan_device,
        } => runtime.block_on(serve::run(serve::Request {
            model,
            host,
            port,
            ctx,
            threads,
            gpu_layers: gpu_layers.count,
            vulkan_device,
        })),
        Command::Models => models::run(),
    }
}
