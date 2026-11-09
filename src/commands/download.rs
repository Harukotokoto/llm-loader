use std::path::PathBuf;

use crate::error::ExitError;
use crate::hub::{self, HubError};
use crate::paths;

pub struct Request {
    pub repo: String,
    pub file: Option<String>,
    pub revision: String,
    pub dir: Option<PathBuf>,
}

pub async fn run(request: Request) -> Result<(), ExitError> {
    let path = hub::download(hub::DownloadRequest {
        repo: request.repo,
        file: request.file,
        revision: request.revision,
        dir: paths::models_dir(request.dir.as_deref()),
    })
    .await
    .map_err(into_exit)?;
    println!("{}", path.display());
    Ok(())
}

fn into_exit(err: HubError) -> ExitError {
    match err {
        HubError::Usage(message) => ExitError::usage(message),
        HubError::Download(message) => ExitError::download(message),
    }
}
