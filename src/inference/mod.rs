mod generate;
mod llama;
mod stop;

use std::path::PathBuf;
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::thread;

/// Layer count that llama.cpp treats as "every layer".
pub const ALL_GPU_LAYERS: u32 = 1000;

/// One chat message, before the model's template is applied.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// What the caller wants the model to continue.
#[derive(Debug)]
pub enum Prompt {
    Chat(Vec<ChatTurn>),
    Text(String),
}

#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub stop: Vec<String>,
    pub seed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug)]
pub enum GenerateEvent {
    Delta(String),
    Finished {
        finish: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    Failed {
        message: String,
        kind: FailKind,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailKind {
    /// The prompt or template cannot be used.
    Request,
    /// Decoding or the runtime failed.
    Internal,
}

#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub id: String,
    pub path: PathBuf,
    pub device: String,
}

pub struct LoadOptions {
    pub model_path: PathBuf,
    pub model_id: String,
    pub context_tokens: u32,
    pub threads: i32,
    pub gpu_layers: u32,
    pub vulkan_device: Option<u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("{0}")]
    Message(String),
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("the model is already generating a response")]
    Busy,
    #[error("the inference thread has stopped")]
    Stopped,
}

struct Job {
    prompt: Prompt,
    params: GenerateParams,
    events: tokio::sync::mpsc::Sender<GenerateEvent>,
}

/// Handle to the single inference thread. One generation runs at a time.
#[derive(Clone)]
pub struct Engine {
    jobs: SyncSender<Job>,
}

impl Engine {
    pub fn generate(
        &self,
        prompt: Prompt,
        params: GenerateParams,
    ) -> Result<tokio::sync::mpsc::Receiver<GenerateEvent>, EngineError> {
        let (events, receiver) = tokio::sync::mpsc::channel(32);
        match self.jobs.try_send(Job {
            prompt,
            params,
            events,
        }) {
            Ok(()) => Ok(receiver),
            Err(TrySendError::Full(_)) => Err(EngineError::Busy),
            Err(TrySendError::Disconnected(_)) => Err(EngineError::Stopped),
        }
    }
}

/// Load the model on a background thread and return once the device is known.
///
/// The thread owns the llama.cpp context. Callers only see [`Engine`].
pub fn start(options: LoadOptions) -> Result<(Engine, LoadedModel), LoadError> {
    if let Some(index) = options.vulkan_device {
        // Read by the Vulkan backend during LlamaBackend::init. The HTTP
        // runtime has not started yet, so this is the only thread.
        unsafe {
            std::env::set_var("GGML_VK_VISIBLE_DEVICES", index.to_string());
        }
    }

    let (jobs_tx, jobs_rx) = sync_channel(1);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    thread::Builder::new()
        .name("llm-inference".into())
        .spawn(move || llama::serve(options, jobs_rx, ready_tx))
        .map_err(|err| LoadError::Message(format!("failed to start inference thread: {err}")))?;

    let loaded = ready_rx.recv().map_err(|_| {
        LoadError::Message("inference thread stopped before the model loaded".into())
    })??;
    Ok((Engine { jobs: jobs_tx }, loaded))
}
