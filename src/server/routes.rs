use std::convert::Infallible;

use std::sync::Arc;

use axum::Router;
use axum::extract::rejection::JsonRejection;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::middleware;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::{Modify, OpenApi};
use utoipa_swagger_ui::SwaggerUi;

use super::auth;
use super::openai::{self, FinishReason};
use crate::inference::{
    ChatTurn, Engine, EngineError, FailKind, FinishReason as ModelFinish, GenerateEvent,
    GenerateStream, Prompt,
};

#[derive(Clone)]
pub struct AppState {
    pub engine: Engine,
    pub model_id: String,
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "llm",
        version = env!("CARGO_PKG_VERSION"),
        description = "OpenAI-compatible API for one local GGUF model. The server loads a single model and ignores the request model name."
    ),
    modifiers(&BearerAuth),
    paths(health, list_models, chat_completions, completions),
    components(schemas(
        openai::HealthBody,
        openai::ModelList,
        openai::ModelCard,
        openai::ChatCompletionRequest,
        openai::ChatMessage,
        openai::ChatTemplateKwargs,
        openai::StreamOptions,
        openai::ChatCompletionResponse,
        openai::ChatChoice,
        openai::AssistantMessage,
        openai::CompletionRequest,
        openai::CompletionResponse,
        openai::CompletionChoice,
        openai::Usage,
        openai::FinishReason,
        openai::ErrorResponse,
        openai::ApiErrorBody
    ))
)]
struct ApiDoc;

/// Declares the bearer scheme used when an API key is configured.
struct BearerAuth;

impl Modify for BearerAuth {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi.components.get_or_insert_with(Default::default);
        components.add_security_scheme(
            "api_key",
            SecurityScheme::Http(HttpBuilder::new().scheme(HttpAuthScheme::Bearer).build()),
        );
    }
}

/// `api_key` set: every `/v1` route requires it. `/health` and `/docs` stay open.
pub fn router(state: AppState, api_key: Option<String>) -> Router {
    let mut v1 = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);
    if let Some(key) = api_key {
        v1 = v1.layer(middleware::from_fn_with_state(
            Arc::new(key),
            auth::require_api_key,
        ));
    }
    Router::new()
        .route("/health", get(health))
        .merge(v1)
        .merge(SwaggerUi::new("/docs").url("/openapi.json", ApiDoc::openapi()))
}

#[utoipa::path(
    get,
    path = "/health",
    responses((status = 200, description = "The process has loaded a model.", body = openai::HealthBody))
)]
async fn health() -> Json<openai::HealthBody> {
    Json(openai::HealthBody {
        status: "ok".into(),
    })
}

#[utoipa::path(
    get,
    path = "/v1/models",
    security(("api_key" = [])),
    responses((status = 200, description = "The loaded model.", body = openai::ModelList))
)]
async fn list_models(State(state): State<AppState>) -> Json<openai::ModelList> {
    Json(openai::ModelList {
        object: "list".into(),
        data: vec![openai::ModelCard {
            id: state.model_id,
            object: "model".into(),
            owned_by: "local".into(),
        }],
    })
}

#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    security(("api_key" = [])),
    request_body = openai::ChatCompletionRequest,
    responses(
        (status = 200, description = "Generated assistant message. stream=true switches the body to server-sent events and ends with [DONE].", body = openai::ChatCompletionResponse),
        (status = 400, description = "The request cannot be run.", body = openai::ErrorResponse),
        (status = 401, description = "The API key is missing or wrong.", body = openai::ErrorResponse),
        (status = 503, description = "Another request is already generating. Retry later.", body = openai::ErrorResponse)
    )
)]
async fn chat_completions(
    State(state): State<AppState>,
    body: Result<Json<openai::ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match body {
        Ok(body) => body,
        Err(err) => return invalid_json(err),
    };
    if request.messages.is_empty() {
        return invalid("messages must not be empty");
    }
    let prepared = match openai::prepare_sampling(
        request.temperature,
        request.top_p,
        request.max_tokens,
        request.stop,
        request.seed,
        request.stream,
        request.n,
    ) {
        Ok(prepared) => prepared,
        Err(message) => return invalid(message),
    };
    let enable_thinking = request
        .chat_template_kwargs
        .and_then(|kwargs| kwargs.enable_thinking)
        .unwrap_or(true);
    let include_usage = wants_usage(request.stream_options);
    let prompt = Prompt::Chat {
        turns: request
            .messages
            .into_iter()
            .map(|message| ChatTurn {
                role: message.role,
                content: message.content,
            })
            .collect(),
        enable_thinking,
    };
    let id = openai::new_id("chatcmpl");
    dispatch(
        state,
        prompt,
        prepared,
        include_usage,
        id,
        "chat.completion",
        StreamShape::Chat,
    )
    .await
}

#[utoipa::path(
    post,
    path = "/v1/completions",
    security(("api_key" = [])),
    request_body = openai::CompletionRequest,
    responses(
        (status = 200, description = "Generated text. stream=true switches the body to server-sent events and ends with [DONE].", body = openai::CompletionResponse),
        (status = 400, description = "The request cannot be run.", body = openai::ErrorResponse),
        (status = 401, description = "The API key is missing or wrong.", body = openai::ErrorResponse),
        (status = 503, description = "Another request is already generating. Retry later.", body = openai::ErrorResponse)
    )
)]
async fn completions(
    State(state): State<AppState>,
    body: Result<Json<openai::CompletionRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match body {
        Ok(body) => body,
        Err(err) => return invalid_json(err),
    };
    if request.prompt.is_empty() {
        return invalid("prompt must not be empty");
    }
    let prepared = match openai::prepare_sampling(
        request.temperature,
        request.top_p,
        request.max_tokens,
        request.stop,
        request.seed,
        request.stream,
        request.n,
    ) {
        Ok(prepared) => prepared,
        Err(message) => return invalid(message),
    };
    let id = openai::new_id("cmpl");
    let include_usage = wants_usage(request.stream_options);
    dispatch(
        state,
        Prompt::Text(request.prompt),
        prepared,
        include_usage,
        id,
        "text_completion",
        StreamShape::Text,
    )
    .await
}

#[derive(Clone, Copy)]
enum StreamShape {
    Chat,
    Text,
}

fn wants_usage(options: Option<openai::StreamOptions>) -> bool {
    options
        .and_then(|options| options.include_usage)
        .unwrap_or(false)
}

struct Collected {
    text: String,
    reasoning: String,
    finish: FinishReason,
    prompt_tokens: u32,
    completion_tokens: u32,
}

async fn dispatch(
    state: AppState,
    prompt: Prompt,
    prepared: openai::Prepared,
    include_usage: bool,
    id: String,
    object: &'static str,
    shape: StreamShape,
) -> Response {
    let mut events = match state.engine.generate(prompt, prepared.params) {
        Ok(events) => events,
        Err(err) => return engine_error(err),
    };
    let created = openai::created_at();
    let model = state.model_id;
    if prepared.stream {
        return stream_response(events, id, created, model, shape, include_usage);
    }
    match collect(&mut events).await {
        Ok(collected) => json_completion(id, object, created, model, shape, collected),
        Err(response) => response,
    }
}

async fn collect(events: &mut GenerateStream) -> Result<Collected, Response> {
    let mut text = String::new();
    let mut reasoning = String::new();
    while let Some(event) = events.recv().await {
        match event {
            GenerateEvent::Reasoning(piece) => reasoning.push_str(&piece),
            GenerateEvent::Delta(piece) => text.push_str(&piece),
            GenerateEvent::Finished {
                finish,
                prompt_tokens,
                completion_tokens,
            } => {
                return Ok(Collected {
                    text,
                    reasoning,
                    finish: wire_finish(finish),
                    prompt_tokens,
                    completion_tokens,
                });
            }
            GenerateEvent::Failed { message, kind } => return Err(generation_error(message, kind)),
        }
    }
    Err(server_error(
        "engine_stopped",
        "the model stopped before finishing",
    ))
}

fn json_completion(
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    shape: StreamShape,
    collected: Collected,
) -> Response {
    let usage = openai::usage(collected.prompt_tokens, collected.completion_tokens);
    let body = match shape {
        StreamShape::Chat => serde_json::to_value(openai::ChatCompletionResponse {
            id,
            object: object.into(),
            created,
            model,
            choices: vec![openai::ChatChoice {
                index: 0,
                message: openai::AssistantMessage {
                    role: "assistant".into(),
                    content: collected.text,
                    reasoning_content: (!collected.reasoning.is_empty())
                        .then_some(collected.reasoning),
                },
                finish_reason: collected.finish,
            }],
            usage,
        }),
        StreamShape::Text => serde_json::to_value(openai::CompletionResponse {
            id,
            object: object.into(),
            created,
            model,
            choices: vec![openai::CompletionChoice {
                index: 0,
                text: collected.text,
                finish_reason: collected.finish,
            }],
            usage,
        }),
    };
    Json(body.unwrap_or(serde_json::Value::Null)).into_response()
}

fn stream_response(
    mut events: GenerateStream,
    id: String,
    created: i64,
    model: String,
    shape: StreamShape,
    include_usage: bool,
) -> Response {
    let stream = async_stream::stream! {
        let mut started = false;
        while let Some(event) = events.recv().await {
            match event {
                GenerateEvent::Reasoning(text) => {
                    let delta = Delta::Reasoning(text);
                    let data = chunk_json(&id, created, &model, shape, delta, started, None);
                    started = true;
                    yield Ok::<Event, Infallible>(Event::default().data(data));
                }
                GenerateEvent::Delta(text) => {
                    let delta = Delta::Content(text);
                    let data = chunk_json(&id, created, &model, shape, delta, started, None);
                    started = true;
                    yield Ok(Event::default().data(data));
                }
                GenerateEvent::Finished { finish, prompt_tokens, completion_tokens } => {
                    let data = chunk_json(
                        &id,
                        created,
                        &model,
                        shape,
                        Delta::Empty,
                        started,
                        Some(wire_finish(finish)),
                    );
                    yield Ok(Event::default().data(data));
                    if include_usage {
                        let data = usage_chunk_json(&id, created, &model, shape, prompt_tokens, completion_tokens);
                        yield Ok(Event::default().data(data));
                    }
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                GenerateEvent::Failed { message, kind } => {
                    let body = openai::ErrorResponse {
                        error: openai::ApiErrorBody {
                            message,
                            error_type: fail_type(kind).into(),
                            code: fail_code(kind).into(),
                        },
                    };
                    if let Ok(data) = serde_json::to_string(&body) {
                        yield Ok(Event::default().data(data));
                    }
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// What one streamed chunk carries.
enum Delta {
    Reasoning(String),
    Content(String),
    Empty,
}

fn chunk_json(
    id: &str,
    created: i64,
    model: &str,
    shape: StreamShape,
    delta: Delta,
    started: bool,
    finish: Option<FinishReason>,
) -> String {
    let payload = match shape {
        StreamShape::Chat => {
            let mut body = serde_json::Map::new();
            if !started {
                body.insert("role".into(), "assistant".into());
            }
            match delta {
                Delta::Reasoning(text) => {
                    body.insert("reasoning_content".into(), text.into());
                }
                Delta::Content(text) => {
                    body.insert("content".into(), text.into());
                }
                Delta::Empty => {}
            }
            serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": body,
                    "finish_reason": finish,
                }],
            })
        }
        StreamShape::Text => serde_json::json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "text": match delta {
                    Delta::Content(text) => text,
                    _ => String::new(),
                },
                "finish_reason": finish,
            }],
        }),
    };
    serde_json::to_string(&payload).unwrap_or_else(|_| "{}".into())
}

/// Last chunk before `[DONE]` when the client asked for usage. It has no choices.
fn usage_chunk_json(
    id: &str,
    created: i64,
    model: &str,
    shape: StreamShape,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> String {
    let object = match shape {
        StreamShape::Chat => "chat.completion.chunk",
        StreamShape::Text => "text_completion",
    };
    let payload = serde_json::json!({
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "choices": [],
        "usage": openai::usage(prompt_tokens, completion_tokens),
    });
    serde_json::to_string(&payload).unwrap_or_else(|_| "{}".into())
}

fn wire_finish(finish: ModelFinish) -> FinishReason {
    match finish {
        ModelFinish::Stop => FinishReason::Stop,
        ModelFinish::Length => FinishReason::Length,
    }
}

fn invalid_json(err: JsonRejection) -> Response {
    invalid(err.body_text())
}

fn invalid(message: impl Into<String>) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "invalid_request",
        message,
    )
}

fn generation_error(message: String, kind: FailKind) -> Response {
    let status = match kind {
        FailKind::Request => StatusCode::BAD_REQUEST,
        FailKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error_response(status, fail_type(kind), fail_code(kind), message)
}

fn engine_error(err: EngineError) -> Response {
    match err {
        EngineError::Busy => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "model_busy",
            err.to_string(),
        ),
        EngineError::Stopped => server_error("engine_stopped", err.to_string()),
    }
}

fn server_error(code: &'static str, message: impl Into<String>) -> Response {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        code,
        message,
    )
}

fn fail_type(kind: FailKind) -> &'static str {
    match kind {
        FailKind::Request => "invalid_request_error",
        FailKind::Internal => "server_error",
    }
}

fn fail_code(kind: FailKind) -> &'static str {
    match kind {
        FailKind::Request => "invalid_prompt",
        FailKind::Internal => "inference_error",
    }
}

fn error_response(
    status: StatusCode,
    error_type: &str,
    code: &str,
    message: impl Into<String>,
) -> Response {
    (
        status,
        Json(openai::ErrorResponse {
            error: openai::ApiErrorBody {
                message: message.into(),
                error_type: error_type.into(),
                code: code.into(),
            },
        }),
    )
        .into_response()
}
