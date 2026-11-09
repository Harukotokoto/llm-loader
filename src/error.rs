/// Process exit codes shared by every command.
#[derive(Debug, thiserror::Error)]
pub enum ExitError {
    /// The arguments or the local catalog cannot be resolved.
    #[error("{0}")]
    Usage(String),
    /// Hugging Face could not be reached, or the file could not be stored.
    #[error("{0}")]
    Download(String),
    /// The model or the Vulkan device could not be used.
    #[error("{0}")]
    Model(String),
    /// The HTTP listener could not bind.
    #[error("{0}")]
    Bind(String),
}

impl ExitError {
    pub fn usage(message: impl Into<String>) -> Self {
        Self::Usage(message.into())
    }

    pub fn download(message: impl Into<String>) -> Self {
        Self::Download(message.into())
    }

    pub fn model(message: impl Into<String>) -> Self {
        Self::Model(message.into())
    }

    pub fn bind(message: impl Into<String>) -> Self {
        Self::Bind(message.into())
    }

    pub fn code(&self) -> u8 {
        match self {
            Self::Usage(_) => 1,
            Self::Download(_) => 2,
            Self::Model(_) => 3,
            Self::Bind(_) => 4,
        }
    }
}
