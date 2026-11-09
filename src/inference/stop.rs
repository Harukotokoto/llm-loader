/// Holds back the suffix of generated text that might still grow into a stop string.
#[derive(Debug)]
pub struct StopBuffer {
    stops: Vec<String>,
    pending: String,
}

impl StopBuffer {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
        }
    }

    /// Append decoded text. The returned string is safe to emit.
    ///
    /// `true` means a stop string was found and generation should end.
    pub fn push(&mut self, piece: &str) -> (String, bool) {
        self.pending.push_str(piece);
        if self.stops.is_empty() {
            return (std::mem::take(&mut self.pending), false);
        }
        if let Some(index) = self.first_stop() {
            let visible = self.pending[..index].to_string();
            self.pending.clear();
            return (visible, true);
        }
        let hold = held_suffix(&self.pending, &self.stops);
        let emit_until = self.pending.len() - hold;
        let visible = self.pending[..emit_until].to_string();
        self.pending = self.pending[emit_until..].to_string();
        (visible, false)
    }

    /// Emit text that was held because it was a prefix of a stop string.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn first_stop(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter(|stop| !stop.is_empty())
            .filter_map(|stop| self.pending.find(stop.as_str()))
            .min()
    }
}

fn held_suffix(text: &str, stops: &[String]) -> usize {
    text.char_indices()
        .map(|(index, _)| &text[index..])
        .filter(|suffix| {
            stops
                .iter()
                .any(|stop| suffix.len() < stop.len() && stop.starts_with(suffix))
        })
        .map(|suffix| suffix.len())
        .max()
        .unwrap_or(0)
}
