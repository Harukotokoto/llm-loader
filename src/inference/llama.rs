use std::num::NonZeroU32;
use std::sync::mpsc::{Receiver, Sender};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::{
    LlamaBackendDevice, LlamaBackendDeviceType, LogOptions, list_llama_ggml_backend_devices,
    send_logs_to_tracing,
};

use super::generate;
use super::{Job, LoadError, LoadOptions, LoadedModel};

const PREFILL_BATCH: u32 = 512;

pub(super) fn serve(
    options: LoadOptions,
    jobs: Receiver<Job>,
    ready: Sender<Result<LoadedModel, LoadError>>,
) {
    if let Err(err) = run(options, jobs, &ready) {
        let _ = ready.send(Err(err));
    }
}

fn run(
    options: LoadOptions,
    jobs: Receiver<Job>,
    ready: &Sender<Result<LoadedModel, LoadError>>,
) -> Result<(), LoadError> {
    send_logs_to_tracing(LogOptions::default());
    let backend = LlamaBackend::init().map_err(|err| LoadError::Message(err.to_string()))?;
    if !backend.supports_gpu_offload() {
        return Err(LoadError::Message(
            "this build cannot offload layers to a GPU".into(),
        ));
    }

    let devices = list_llama_ggml_backend_devices();
    let device = select_vulkan_gpu(&devices)?;
    let params = LlamaModelParams::default()
        .with_n_gpu_layers(options.gpu_layers)
        .with_use_mlock(false)
        .with_devices(&[device.index])
        .map_err(|err| LoadError::Message(format!("failed to select the Vulkan device: {err}")))?;
    let model =
        llama_cpp_2::model::LlamaModel::load_from_file(&backend, &options.model_path, &params)
            .map_err(|err| LoadError::Message(format!("failed to load the model: {err}")))?;

    let context_tokens = NonZeroU32::new(options.context_tokens)
        .ok_or_else(|| LoadError::Message("context length must be at least 1".into()))?;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(context_tokens))
        .with_n_batch(PREFILL_BATCH)
        .with_n_threads(options.threads)
        .with_n_threads_batch(options.threads);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .map_err(|err| LoadError::Message(format!("failed to create the context: {err}")))?;

    let loaded = LoadedModel {
        id: options.model_id,
        path: options.model_path,
        device: format!("{} ({})", device.description, device.name),
    };
    ready
        .send(Ok(loaded))
        .map_err(|_| LoadError::Message("the server stopped before the model was ready".into()))?;

    while let Ok(job) = jobs.recv() {
        generate::run(&model, &mut ctx, job);
    }
    Ok(())
}

fn select_vulkan_gpu(devices: &[LlamaBackendDevice]) -> Result<&LlamaBackendDevice, LoadError> {
    let listing = devices
        .iter()
        .map(device_line)
        .collect::<Vec<_>>()
        .join("\n");
    devices
        .iter()
        .find(|device| is_vulkan_gpu(device))
        .ok_or_else(|| {
            LoadError::Message(format!(
                "no Vulkan GPU is available, so the server will not start.\n{listing}"
            ))
        })
}

fn is_vulkan_gpu(device: &LlamaBackendDevice) -> bool {
    let label = format!("{} {}", device.name, device.description).to_ascii_lowercase();
    if label.contains("llvmpipe") {
        return false;
    }
    device.backend.eq_ignore_ascii_case("vulkan")
        && matches!(
            device.device_type,
            LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
        )
}

fn device_line(device: &LlamaBackendDevice) -> String {
    format!(
        "- {} | {} | backend={} | {:?}",
        device.name, device.description, device.backend, device.device_type
    )
}
