use std::time::{SystemTime, UNIX_EPOCH};

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

use crate::inference::GenerateParams;

pub const DEFAULT_MAX_TOKENS: u32 = 512;

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthBody {
    pub status: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatCompletionRequest {
    /// Sent by OpenAI clients. The server always uses the loaded model.
    #[allow(dead_code)]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    /// One string, or an array of strings.
    #[serde(default, deserialize_with = "deserialize_stop")]
    #[schema(value_type = Vec<String>)]
    pub stop: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub seed: Option<u32>,
    pub n: Option<u32>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ChatChoice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AssistantMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CompletionRequest {
    /// Sent by OpenAI clients. The server always uses the loaded model.
    #[allow(dead_code)]
    pub model: String,
    pub prompt: String,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    /// One string, or an array of strings.
    #[serde(default, deserialize_with = "deserialize_stop")]
    #[schema(value_type = Vec<String>)]
    pub stop: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub seed: Option<u32>,
    pub n: Option<u32>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Copy, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    pub error: ApiErrorBody,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    #[schema(rename = "type")]
    pub error_type: String,
    pub code: String,
}

pub struct Prepared {
    pub params: GenerateParams,
    pub stream: bool,
}

pub fn prepare_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<Vec<String>>,
    seed: Option<u32>,
    stream: Option<bool>,
    n: Option<u32>,
) -> Result<Prepared, String> {
    if n.is_some_and(|count| count != 1) {
        return Err("n must be 1".into());
    }
    let temperature = temperature.unwrap_or(1.0);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("temperature must be zero or a positive finite number".into());
    }
    let top_p = top_p.unwrap_or(1.0);
    if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
        return Err("top_p must be greater than 0 and at most 1".into());
    }
    let stop = stop.unwrap_or_default();
    if stop.iter().any(|item| item.is_empty()) {
        return Err("stop strings must not be empty".into());
    }
    Ok(Prepared {
        params: GenerateParams {
            temperature,
            top_p,
            max_tokens: max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            stop,
            seed: seed.unwrap_or_else(random_seed),
        },
        stream: stream.unwrap_or(false),
    })
}

pub fn usage(prompt_tokens: u32, completion_tokens: u32) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    }
}

pub fn created_at() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0)
}

pub fn new_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:x}{n:x}", created_at())
}

fn random_seed() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u32)
        .unwrap_or(1)
}

fn deserialize_stop<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(vec![text])),
        Some(Value::Array(items)) => {
            let mut stops = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::String(text) => stops.push(text),
                    _ => return Err(serde::de::Error::custom("stop entries must be strings")),
                }
            }
            Ok(Some(stops))
        }
        Some(_) => Err(serde::de::Error::custom(
            "stop must be a string or an array of strings",
        )),
    }
}
