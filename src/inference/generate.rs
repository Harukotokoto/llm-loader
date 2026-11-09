use encoding_rs::Decoder;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use super::stop::StopBuffer;
use super::{FailKind, FinishReason, GenerateEvent, GenerateParams, Job, Prompt};

const BATCH_TOKENS: usize = 512;

enum Emitted {
    Continue,
    Stopped,
    Closed,
}

pub(super) fn run(model: &LlamaModel, ctx: &mut LlamaContext<'_>, job: Job) {
    ctx.clear_kv_cache();
    let rendered = match render(model, &job.prompt) {
        Ok(prompt) => prompt,
        Err((kind, message)) => {
            let _ = job
                .events
                .blocking_send(GenerateEvent::Failed { message, kind });
            return;
        }
    };

    let tokens = model.vocab().tokenize(rendered.as_bytes(), false, true);
    if tokens.is_empty() {
        fail(&job, FailKind::Request, "the prompt produced no tokens");
        return;
    }

    let n_ctx = ctx.n_ctx() as usize;
    if tokens.len() >= n_ctx {
        fail(
            &job,
            FailKind::Request,
            format!(
                "prompt is {} tokens and the context holds {n_ctx}",
                tokens.len()
            ),
        );
        return;
    }

    let mut batch = LlamaBatch::new(BATCH_TOKENS.min(n_ctx), 1);
    if let Err(message) = prefill(ctx, &mut batch, &tokens) {
        fail(&job, FailKind::Internal, message);
        return;
    }

    let prompt_tokens = tokens.len() as u32;
    let limit = job.params.max_tokens.min((n_ctx - tokens.len()) as u32);
    let mut sampler = sampler_for(&job.params);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut stops = StopBuffer::new(job.params.stop.clone());
    let mut completion_tokens = 0u32;
    let mut reason = None;

    while completion_tokens < limit {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.vocab().is_eog(token) {
            reason = Some(FinishReason::Stop);
            break;
        }

        let bytes = model.vocab().token_to_piece(token, false, None);
        let piece = decode_piece(&mut decoder, &bytes, false);
        match emit_piece(&job, &mut stops, &piece) {
            Emitted::Closed => return,
            Emitted::Stopped => {
                reason = Some(FinishReason::Stop);
                completion_tokens += 1;
                break;
            }
            Emitted::Continue => {}
        }

        let position = tokens.len() as i32 + completion_tokens as i32;
        completion_tokens += 1;
        batch.clear();
        if let Err(err) = batch.add(token, position, &[0], true) {
            fail(
                &job,
                FailKind::Internal,
                format!("failed to queue the next token: {err}"),
            );
            return;
        }
        if let Err(err) = ctx.decode(&mut batch) {
            fail(
                &job,
                FailKind::Internal,
                format!("failed to decode a token: {err}"),
            );
            return;
        }
    }

    if reason != Some(FinishReason::Stop) {
        let tail = decode_piece(&mut decoder, &[], true);
        match emit_piece(&job, &mut stops, &tail) {
            Emitted::Closed => return,
            Emitted::Stopped => reason = Some(FinishReason::Stop),
            Emitted::Continue => {
                let rest = stops.flush();
                if !rest.is_empty()
                    && job
                        .events
                        .blocking_send(GenerateEvent::Delta(rest))
                        .is_err()
                {
                    return;
                }
            }
        }
    }

    let _ = job.events.blocking_send(GenerateEvent::Finished {
        finish: reason.unwrap_or(FinishReason::Length),
        prompt_tokens,
        completion_tokens,
    });
}

fn fail(job: &Job, kind: FailKind, message: impl Into<String>) {
    let _ = job.events.blocking_send(GenerateEvent::Failed {
        message: message.into(),
        kind,
    });
}

fn emit_piece(job: &Job, stops: &mut StopBuffer, piece: &str) -> Emitted {
    let (visible, stopped) = stops.push(piece);
    if !visible.is_empty()
        && job
            .events
            .blocking_send(GenerateEvent::Delta(visible))
            .is_err()
    {
        return Emitted::Closed;
    }
    if stopped {
        Emitted::Stopped
    } else {
        Emitted::Continue
    }
}

fn decode_piece(decoder: &mut Decoder, bytes: &[u8], last: bool) -> String {
    let mut piece = String::new();
    let _ = decoder.decode_to_string(bytes, &mut piece, last);
    piece
}

fn prefill(
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    tokens: &[LlamaToken],
) -> Result<(), String> {
    let last = tokens.len() - 1;
    let mut start = 0;
    while start < tokens.len() {
        batch.clear();
        let end = (start + BATCH_TOKENS).min(tokens.len());
        for (offset, token) in tokens[start..end].iter().copied().enumerate() {
            let position = (start + offset) as i32;
            batch
                .add(token, position, &[0], start + offset == last)
                .map_err(|err| format!("failed to queue prompt tokens: {err}"))?;
        }
        ctx.decode(batch)
            .map_err(|err| format!("failed to decode the prompt: {err}"))?;
        start = end;
    }
    Ok(())
}

fn sampler_for(params: &GenerateParams) -> LlamaSampler {
    if params.temperature <= 0.0 {
        return LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    }
    LlamaSampler::chain_simple([
        LlamaSampler::temp(params.temperature),
        LlamaSampler::top_p(params.top_p, 1),
        LlamaSampler::dist(params.seed),
    ])
}

fn render(model: &LlamaModel, prompt: &Prompt) -> Result<String, (FailKind, String)> {
    match prompt {
        Prompt::Text(text) => Ok(text.clone()),
        Prompt::Chat(turns) => {
            let template = model.chat_template(None).map_err(|err| {
                (
                    FailKind::Request,
                    format!("this model has no usable chat template: {err}"),
                )
            })?;
            let messages = turns
                .iter()
                .map(|turn| LlamaChatMessage::new(turn.role.clone(), turn.content.clone()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| (FailKind::Request, format!("invalid chat message: {err}")))?;
            model
                .apply_chat_template(&template, &messages, true)
                .map_err(|err| {
                    (
                        FailKind::Request,
                        format!("failed to apply the chat template: {err}"),
                    )
                })
        }
    }
}
