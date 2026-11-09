const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Which part of the answer a piece of text belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    Reasoning(String),
    Content(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Waiting to see whether the output opens with `<think>`.
    Start,
    Reasoning,
    Content,
}

/// Splits `<think>...</think>` out of generated text, across arbitrary piece boundaries.
///
/// Whitespace around the reasoning block is dropped.
#[derive(Debug)]
pub struct ThinkSplitter {
    state: State,
    pending: String,
    trim_start: bool,
}

impl ThinkSplitter {
    /// `reasoning_open` is true when the prompt already ends inside `<think>`.
    pub fn new(reasoning_open: bool) -> Self {
        Self {
            state: if reasoning_open {
                State::Reasoning
            } else {
                State::Start
            },
            pending: String::new(),
            trim_start: reasoning_open,
        }
    }

    pub fn push(&mut self, piece: &str) -> Vec<Segment> {
        self.pending.push_str(piece);
        let mut out = Vec::new();
        loop {
            match self.state {
                State::Start => {
                    let head = self.pending.trim_start();
                    if let Some(rest) = head.strip_prefix(OPEN) {
                        self.pending = rest.to_string();
                        self.state = State::Reasoning;
                        self.trim_start = true;
                    } else if head.is_empty() || OPEN.starts_with(head) {
                        return out;
                    } else {
                        self.state = State::Content;
                        self.trim_start = false;
                    }
                }
                State::Reasoning => {
                    if let Some(index) = self.pending.find(CLOSE) {
                        let before = self.pending[..index].trim_end().to_string();
                        self.pending = self.pending[index + CLOSE.len()..].to_string();
                        self.emit_reasoning(&before, &mut out);
                        self.state = State::Content;
                        self.trim_start = true;
                    } else {
                        // Keep a possible partial `</think>` and trailing whitespace back.
                        let hold = partial_suffix(&self.pending, CLOSE);
                        let body_end = self.pending.len() - hold;
                        let emit_end = self.pending[..body_end].trim_end().len();
                        let text = self.pending[..emit_end].to_string();
                        self.pending = self.pending[emit_end..].to_string();
                        self.emit_reasoning(&text, &mut out);
                        return out;
                    }
                }
                State::Content => {
                    if self.trim_start {
                        self.pending = self.pending.trim_start().to_string();
                        if self.pending.is_empty() {
                            return out;
                        }
                        self.trim_start = false;
                    }
                    if !self.pending.is_empty() {
                        out.push(Segment::Content(std::mem::take(&mut self.pending)));
                    }
                    return out;
                }
            }
        }
    }

    /// Release anything still held when generation ends.
    pub fn finish(&mut self) -> Vec<Segment> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.pending);
        match self.state {
            State::Start | State::Content => {
                if !rest.is_empty() {
                    out.push(Segment::Content(rest));
                }
            }
            State::Reasoning => {
                let text = rest.trim_end().to_string();
                self.emit_reasoning(&text, &mut out);
            }
        }
        out
    }

    fn emit_reasoning(&mut self, text: &str, out: &mut Vec<Segment>) {
        let text = if self.trim_start {
            text.trim_start()
        } else {
            text
        };
        if !text.is_empty() {
            self.trim_start = false;
            out.push(Segment::Reasoning(text.to_string()));
        }
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `tag`.
fn partial_suffix(text: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&len| {
            len <= text.len()
                && text.is_char_boundary(text.len() - len)
                && tag.starts_with(&text[text.len() - len..])
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pieces: &[&str], reasoning_open: bool) -> (String, String) {
        let mut splitter = ThinkSplitter::new(reasoning_open);
        let mut segments = Vec::new();
        for piece in pieces {
            segments.extend(splitter.push(piece));
        }
        segments.extend(splitter.finish());
        let (mut reasoning, mut content) = (String::new(), String::new());
        for segment in segments {
            match segment {
                Segment::Reasoning(text) => reasoning.push_str(&text),
                Segment::Content(text) => content.push_str(&text),
            }
        }
        (reasoning, content)
    }

    #[test]
    fn splits_reasoning_and_content() {
        let (reasoning, content) = run(&["<think>\nthin", "king\n</think>\n\nhello"], false);
        assert_eq!(reasoning, "thinking");
        assert_eq!(content, "hello");
    }

    #[test]
    fn handles_tags_split_across_pieces() {
        let (reasoning, content) = run(&["<th", "ink>a ", "b</thi", "nk>c"], false);
        assert_eq!(reasoning, "a b");
        assert_eq!(content, "c");
    }

    #[test]
    fn plain_answer_is_content() {
        let (reasoning, content) = run(&["Hello", " world"], false);
        assert_eq!(reasoning, "");
        assert_eq!(content, "Hello world");
    }

    #[test]
    fn prompt_already_inside_think() {
        let (reasoning, content) = run(&["step 1\n", "step 2\n</think>\n\nanswer"], true);
        assert_eq!(reasoning, "step 1\nstep 2");
        assert_eq!(content, "answer");
    }

    #[test]
    fn unfinished_reasoning_is_not_lost() {
        let (reasoning, content) = run(&["<think>\nstill going </thi"], false);
        assert_eq!(reasoning, "still going </thi");
        assert_eq!(content, "");
    }
}
