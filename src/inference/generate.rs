use encoding_rs::Decoder;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use super::stop::StopBuffer;
use super::think::{Segment, ThinkSplitter};
use super::{FailKind, FinishReason, GenerateEvent, GenerateParams, Job, Prompt};

const BATCH_TOKENS: usize = 512;

enum Emitted {
    Continue,
    Stopped,
    Closed,
}

pub(super) fn run(model: &LlamaModel, ctx: &mut LlamaContext<'_>, job: Job) {
    ctx.clear_kv_cache();
    let (rendered, think) = match render(model, &job.prompt) {
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
    let mut output = Output {
        stops: StopBuffer::new(job.params.stop.clone()),
        think,
    };
    let mut completion_tokens = 0u32;
    let mut reason = None;
    let mut stopped_by_text = false;

    while completion_tokens < limit {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.vocab().is_eog(token) {
            reason = Some(FinishReason::Stop);
            break;
        }

        let bytes = model.vocab().token_to_piece(token, false, None);
        let piece = decode_piece(&mut decoder, &bytes, false);
        match output.push(&job, &piece) {
            Emitted::Closed => return,
            Emitted::Stopped => {
                reason = Some(FinishReason::Stop);
                completion_tokens += 1;
                stopped_by_text = true;
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

    if !stopped_by_text {
        let tail = decode_piece(&mut decoder, &[], true);
        match output.push(&job, &tail) {
            Emitted::Closed => return,
            Emitted::Stopped => reason = Some(FinishReason::Stop),
            Emitted::Continue => {
                if let Emitted::Closed = output.finish(&job) {
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

/// Routes decoded text to reasoning or answer events, applying stop strings to the answer.
struct Output {
    stops: StopBuffer,
    /// Present for chat prompts only.
    think: Option<ThinkSplitter>,
}

impl Output {
    fn push(&mut self, job: &Job, piece: &str) -> Emitted {
        let segments = match &mut self.think {
            Some(think) => think.push(piece),
            None if piece.is_empty() => Vec::new(),
            None => vec![Segment::Content(piece.to_string())],
        };
        self.emit(job, segments)
    }

    /// Release text held back for tag or stop-string matching.
    fn finish(&mut self, job: &Job) -> Emitted {
        let segments = self
            .think
            .as_mut()
            .map(ThinkSplitter::finish)
            .unwrap_or_default();
        match self.emit(job, segments) {
            Emitted::Continue => {}
            other => return other,
        }
        let rest = self.stops.flush();
        if !rest.is_empty()
            && job
                .events
                .blocking_send(GenerateEvent::Delta(rest))
                .is_err()
        {
            return Emitted::Closed;
        }
        Emitted::Continue
    }

    fn emit(&mut self, job: &Job, segments: Vec<Segment>) -> Emitted {
        for segment in segments {
            let sent = match segment {
                Segment::Reasoning(text) => job
                    .events
                    .blocking_send(GenerateEvent::Reasoning(text))
                    .is_ok(),
                Segment::Content(text) => match self.emit_content(job, &text) {
                    Emitted::Continue => true,
                    other => return other,
                },
            };
            if !sent {
                return Emitted::Closed;
            }
        }
        Emitted::Continue
    }

    fn emit_content(&mut self, job: &Job, text: &str) -> Emitted {
        let (visible, stopped) = self.stops.push(text);
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
}

/// Decode bytes that may end inside a multi-byte character.
///
/// `decode_to_string` never grows the string, so the capacity is reserved first.
fn decode_piece(decoder: &mut Decoder, bytes: &[u8], last: bool) -> String {
    let capacity = decoder
        .max_utf8_buffer_length(bytes.len())
        .unwrap_or(bytes.len() * 3 + 4);
    let mut piece = String::with_capacity(capacity);
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

/// Marker a thinking template leaves open at the end of the prompt.
const THINK_OPEN: &str = "<think>";
/// What the template adds after the assistant header when thinking is off.
const THINK_SKIPPED: &str = "<think>\n\n</think>\n\n";

/// Build the prompt text. Chat prompts also get a splitter for `<think>` blocks.
fn render(
    model: &LlamaModel,
    prompt: &Prompt,
) -> Result<(String, Option<ThinkSplitter>), (FailKind, String)> {
    match prompt {
        Prompt::Text(text) => Ok((text.clone(), None)),
        Prompt::Chat {
            turns,
            enable_thinking,
        } => {
            let template = model.chat_template(None).map_err(|err| {
                (
                    FailKind::Request,
                    format!("this model has no usable chat template: {err}"),
                )
            })?;
            let supports_thinking = template
                .to_str()
                .is_ok_and(|source| source.contains(THINK_OPEN));
            let messages = turns
                .iter()
                .map(|turn| LlamaChatMessage::new(turn.role.clone(), turn.content.clone()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| (FailKind::Request, format!("invalid chat message: {err}")))?;
            let mut text = model
                .apply_chat_template(&template, &messages, true)
                .map_err(|err| {
                    (
                        FailKind::Request,
                        format!("failed to apply the chat template: {err}"),
                    )
                })?;
            if supports_thinking && !enable_thinking {
                text.push_str(THINK_SKIPPED);
            }
            let reasoning_open = text.trim_end().ends_with(THINK_OPEN);
            Ok((text, Some(ThinkSplitter::new(reasoning_open))))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_pieces_split_inside_a_character() {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let bytes = "こんにちは".as_bytes();
        let (head, tail) = bytes.split_at(4);
        let mut text = decode_piece(&mut decoder, b"Hi ", false);
        text.push_str(&decode_piece(&mut decoder, head, false));
        text.push_str(&decode_piece(&mut decoder, tail, false));
        text.push_str(&decode_piece(&mut decoder, &[], true));
        assert_eq!(text, "Hi こんにちは");
    }
}
