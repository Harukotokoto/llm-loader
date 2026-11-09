mod cli;
mod commands;
mod error;
mod hub;
mod inference;
mod paths;
mod server;

use std::process::ExitCode;

fn main() -> ExitCode {
    // Optional settings such as QWEN_API_KEY. Real environment variables win.
    let _ = dotenvy::dotenv();
    match cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(err.code())
        }
    }
}
