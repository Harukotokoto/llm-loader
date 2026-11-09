use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Record of the model selected by the latest successful `download`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentModel {
    pub repo: String,
    pub file: String,
    pub path: PathBuf,
    pub revision: String,
}

/// `QWEN_MODELS_DIR`, or `./models` when the variable is unset or empty.
pub fn models_dir(flag: Option<&Path>) -> PathBuf {
    if let Some(path) = flag {
        return path.to_path_buf();
    }
    match std::env::var("QWEN_MODELS_DIR") {
        Ok(value) if !value.is_empty() => PathBuf::from(value),
        _ => PathBuf::from("models"),
    }
}

pub fn current_path(dir: &Path) -> PathBuf {
    dir.join("current.json")
}

pub fn read_current(dir: &Path) -> Result<Option<CurrentModel>, String> {
    let path = current_path(dir);
    if !path.exists() {
        return Ok(None);
    }
    let body =
        fs::read_to_string(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
    serde_json::from_str(&body).map_err(|err| format!("parse {}: {err}", path.display()))
}

pub fn write_current(dir: &Path, current: &CurrentModel) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|err| format!("create {}: {err}", dir.display()))?;
    let path = current_path(dir);
    let tmp = dir.join("current.json.tmp");
    let body =
        serde_json::to_vec_pretty(current).map_err(|err| format!("encode current model: {err}"))?;
    fs::write(&tmp, body).map_err(|err| format!("write {}: {err}", tmp.display()))?;
    fs::rename(&tmp, &path).map_err(|err| format!("replace {}: {err}", path.display()))
}

/// Model id shown by `/v1/models`. A downloaded model keeps its repo id.
pub fn model_id(path: &Path, current: Option<&CurrentModel>) -> String {
    if let Some(current) = current
        && same_file(path, &current.path)
    {
        return current.repo.clone();
    }
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("local")
        .to_string()
}

pub fn same_file(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

pub fn gguf_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut found = Vec::new();
    if dir.exists() {
        collect_gguf(dir, &mut found)?;
        found.sort();
    }
    Ok(found)
}

fn collect_gguf(dir: &Path, found: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|err| format!("read {}: {err}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| format!("read {}: {err}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_gguf(&path, found)?;
            continue;
        }
        let is_gguf = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
        if is_gguf {
            found.push(path);
        }
    }
    Ok(())
}
