use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

use hf_hub::HFClient;
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};

use crate::paths::{self, CurrentModel};

#[derive(Debug)]
pub enum HubError {
    Usage(String),
    Download(String),
}

pub struct DownloadRequest {
    pub repo: String,
    pub file: Option<String>,
    pub revision: String,
    pub dir: PathBuf,
}

pub fn parse_repo(input: &str) -> Result<(String, String), HubError> {
    let input = input.trim().trim_end_matches('/');
    let path = strip_host(input).ok_or_else(|| {
        HubError::Usage("repo must be owner/name or a https://huggingface.co/owner/name URL".into())
    })?;
    let path = path.trim_end_matches('/').trim_end_matches(".git");
    let mut parts = path.split('/').filter(|part| !part.is_empty());
    let owner = parts.next().unwrap_or("");
    let name = parts.next().unwrap_or("");
    if owner.is_empty() || name.is_empty() || parts.next().is_some() {
        return Err(HubError::Usage(
            "repo must be owner/name, with no extra path".into(),
        ));
    }
    Ok((owner.to_string(), name.to_string()))
}

fn strip_host(input: &str) -> Option<&str> {
    for prefix in [
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://hf.co/",
        "http://hf.co/",
    ] {
        if let Some(rest) = input.strip_prefix(prefix) {
            return Some(rest);
        }
    }
    if input.contains("://") {
        return None;
    }
    Some(input)
}

/// Pick the one GGUF file a download should fetch.
pub fn choose_file(names: &[String], requested: Option<&str>) -> Result<String, HubError> {
    let gguf = gguf_names(names);
    match requested {
        Some(file) if looks_like_file(file) => {
            check_repo_path(file)?;
            Ok(file.to_string())
        }
        Some(quant) => match_token(&gguf, quant),
        None => prefer_default(&gguf),
    }
}

pub async fn download(request: DownloadRequest) -> Result<PathBuf, HubError> {
    if request.revision.is_empty() {
        return Err(HubError::Usage("revision must not be empty".into()));
    }
    let (owner, name) = parse_repo(&request.repo)?;
    let client = HFClient::builder()
        .build()
        .map_err(|err| HubError::Download(err.to_string()))?;
    let repo = client.model(owner.clone(), name.clone());
    let filename = match request.file.as_deref() {
        Some(file) if looks_like_file(file) => {
            check_repo_path(file)?;
            file.to_string()
        }
        other => {
            let info = repo
                .info()
                .revision(request.revision.clone())
                .send()
                .await
                .map_err(|err| HubError::Download(err.to_string()))?;
            let names = info
                .siblings
                .unwrap_or_default()
                .into_iter()
                .map(|sibling| sibling.rfilename)
                .collect::<Vec<_>>();
            choose_file(&names, other)?
        }
    };

    let dest_dir = request.dir.join(&owner).join(&name);
    let dest = dest_dir.join(&filename);
    let path = if dest.is_file() && file_size(&dest)? > 0 {
        eprintln!("already downloaded");
        dest
    } else {
        eprintln!("downloading {owner}/{name} {filename}");
        repo.download_file()
            .filename(filename.clone())
            .local_dir(dest_dir)
            .revision(request.revision.clone())
            .progress(Progress::new(StderrProgress))
            .send()
            .await
            .map_err(|err| HubError::Download(err.to_string()))?
    };
    let path = fs_canonicalize(&path);
    paths::write_current(
        &request.dir,
        &CurrentModel {
            repo: format!("{owner}/{name}"),
            file: filename,
            path: path.clone(),
            revision: request.revision,
        },
    )
    .map_err(HubError::Download)?;
    Ok(path)
}

fn looks_like_file(value: &str) -> bool {
    value.to_ascii_lowercase().ends_with(".gguf")
}

fn check_repo_path(file: &str) -> Result<(), HubError> {
    let path = Path::new(file);
    if path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        return Err(HubError::Usage(
            "file must be a path inside the repository".into(),
        ));
    }
    Ok(())
}

fn gguf_names(names: &[String]) -> Vec<&str> {
    names
        .iter()
        .map(String::as_str)
        .filter(|name| name.to_ascii_lowercase().ends_with(".gguf"))
        .collect()
}

fn prefer_default(gguf: &[&str]) -> Result<String, HubError> {
    let matches = names_containing(gguf, "q4_k_m");
    if matches.len() == 1 {
        return Ok(matches[0].to_string());
    }
    if matches.len() > 1 {
        return Err(ambiguous("Q4_K_M", &matches));
    }
    if gguf.len() == 1 {
        return Ok(gguf[0].to_string());
    }
    if gguf.is_empty() {
        return Err(HubError::Usage(
            "repository has no .gguf file. Pass --file with a file name.".into(),
        ));
    }
    Err(ambiguous("a single GGUF", gguf))
}

fn match_token(gguf: &[&str], token: &str) -> Result<String, HubError> {
    let matches = names_containing(gguf, &token.to_ascii_lowercase());
    if matches.len() == 1 {
        return Ok(matches[0].to_string());
    }
    if matches.is_empty() {
        return Err(HubError::Usage(format!(
            "no .gguf file contains {token}.{}",
            list_files(gguf)
        )));
    }
    Err(ambiguous(token, &matches))
}

fn names_containing<'a>(names: &[&'a str], needle: &str) -> Vec<&'a str> {
    names
        .iter()
        .copied()
        .filter(|name| name.to_ascii_lowercase().contains(needle))
        .collect()
}

fn ambiguous(label: &str, names: &[&str]) -> HubError {
    HubError::Usage(format!(
        "multiple files match {label}. Pass --file with one of them.{}",
        list_files(names)
    ))
}

fn list_files(names: &[&str]) -> String {
    if names.is_empty() {
        return String::new();
    }
    let mut body = String::from("\n");
    for name in names {
        body.push_str("- ");
        body.push_str(name);
        body.push('\n');
    }
    body
}

fn file_size(path: &Path) -> Result<u64, HubError> {
    std::fs::metadata(path)
        .map(|meta| meta.len())
        .map_err(|err| HubError::Download(format!("stat {}: {err}", path.display())))
}

fn fs_canonicalize(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

struct StderrProgress;

impl ProgressHandler for StderrProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                eprintln!("download: {total_bytes} bytes");
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            } => {
                eprint!("\rdownload: {bytes_completed}/{total_bytes} bytes");
                let _ = io::stderr().flush();
            }
            DownloadEvent::Progress { files } => {
                for file in files {
                    eprint!(
                        "\rdownload: {} {}/{} bytes",
                        file.filename, file.bytes_completed, file.total_bytes
                    );
                }
                let _ = io::stderr().flush();
            }
            DownloadEvent::Complete => {
                eprintln!("\ndownload finished");
            }
        }
    }
}
