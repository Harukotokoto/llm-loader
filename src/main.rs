mod cli;
mod commands;
mod error;
mod hub;
mod inference;
mod paths;
mod server;

use std::process::ExitCode;

fn main() -> ExitCode {
    match cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(err.code())
        }
    }
}
