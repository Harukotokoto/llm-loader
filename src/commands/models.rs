use crate::error::ExitError;
use crate::paths::{self, CurrentModel};

pub fn run() -> Result<(), ExitError> {
    let dir = paths::models_dir(None);
    let current = paths::read_current(&dir).map_err(ExitError::usage)?;
    let files = paths::gguf_files(&dir).map_err(ExitError::usage)?;

    if let Some(current) = &current {
        println!("current: {}", current.repo);
        if !files
            .iter()
            .any(|path| paths::same_file(path, &current.path))
        {
            println!("missing: {}", current.path.display());
        }
    }

    if files.is_empty() && current.is_none() {
        println!("models: {}", dir.display());
        return Ok(());
    }

    for path in files {
        let marker = match &current {
            Some(current) if is_current(&path, current) => "*",
            _ => " ",
        };
        println!("{marker} {}", path.display());
    }
    Ok(())
}

fn is_current(path: &std::path::Path, current: &CurrentModel) -> bool {
    paths::same_file(path, &current.path)
}
