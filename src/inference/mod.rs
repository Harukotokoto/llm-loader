mod generate;
mod llama;
mod stop;
mod think;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
    Chat {
        turns: Vec<ChatTurn>,
        /// Lets a thinking model reason before it answers.
        enable_thinking: bool,
    },
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
    /// Text inside `<think>`. Only chat prompts produce it.
    Reasoning(String),
    /// Answer text.
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

/// Marks the model as taken. Releasing it lets the next request start.
struct BusyPermit(Arc<AtomicBool>);

impl Drop for BusyPermit {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Events of one generation. The model stays reserved until the last event is read
/// or this value is dropped.
pub struct GenerateStream {
    receiver: tokio::sync::mpsc::Receiver<GenerateEvent>,
    permit: Option<BusyPermit>,
}

impl GenerateStream {
    pub async fn recv(&mut self) -> Option<GenerateEvent> {
        let event = self.receiver.recv().await;
        if matches!(
            event,
            None | Some(GenerateEvent::Finished { .. } | GenerateEvent::Failed { .. })
        ) {
            self.permit = None;
        }
        event
    }
}

/// Handle to the single inference thread. One generation runs at a time.
#[derive(Clone)]
pub struct Engine {
    jobs: SyncSender<Job>,
    busy: Arc<AtomicBool>,
}

impl Engine {
    /// Start a generation, or fail with [`EngineError::Busy`] while another one is running.
    pub fn generate(
        &self,
        prompt: Prompt,
        params: GenerateParams,
    ) -> Result<GenerateStream, EngineError> {
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(EngineError::Busy);
        }
        let permit = BusyPermit(self.busy.clone());
        let (events, receiver) = tokio::sync::mpsc::channel(32);
        match self.jobs.try_send(Job {
            prompt,
            params,
            events,
        }) {
            Ok(()) => Ok(GenerateStream {
                receiver,
                permit: Some(permit),
            }),
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
    Ok((
        Engine {
            jobs: jobs_tx,
            busy: Arc::new(AtomicBool::new(false)),
        },
        loaded,
    ))
}
