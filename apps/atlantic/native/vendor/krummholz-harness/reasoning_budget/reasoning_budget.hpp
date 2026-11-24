// Krummholz harness: a hard limit on hidden reasoning. The model's reply arrives as a stream: reasoning, the close of
// the reasoning ("</think>"), then the visible answer. If the reasoning is still open after `budget` generated tokens,
// the harness stops generation and shows a fixed "I don't know" instead of waiting or showing nothing.
//
// Why: Atlantic Lite v8 leaves its reasoning open on about 16 percent of fact questions asked without tools (it
// second-guesses itself until the token limit). Those replies are not answers; under this limit they become honest
// abstentions, which is what the model does on its own for half of such questions.
//
// Use, once per reply:
//   kz::ReasoningBudget b(1024);
//   for each generated piece:  if (b.feed(piece, tokens_in_piece) == kz::ReasoningBudget::Stop) { stop the model; break; }
//   show b.visible()           // the answer after the close of the reasoning, or the fixed reply if the limit was hit
// Reasoning text is never shown. No model, no network, no allocation beyond the answer itself.
#pragma once
#include <string>

namespace kz {

class ReasoningBudget {
  public:
    enum Action { Continue, Stop };

    explicit ReasoningBudget(int budget_tokens = 1024,
                             std::string fallback = "I don't know. I could not reach an answer I can stand behind, so I would rather say so than guess.")
        : budget_(budget_tokens), fallback_(std::move(fallback)) {}

    // One generated piece of text and how many tokens it was. Returns Stop when generation must be cut off.
    Action feed(const std::string& piece, int tokens = 1) {
        if (exceeded_) return Stop;
        if (closed_) { answer_ += piece; return Continue; }
        tail_ += piece;
        reasoning_tokens_ += tokens;
        const auto at = tail_.find(kClose);
        if (at != std::string::npos) {  // the close may arrive split across pieces, so it is searched in the kept tail
            closed_ = true;
            answer_ = tail_.substr(at + kCloseLen);
            tail_.clear();
            return Continue;
        }
        if (tail_.size() > 64) tail_.erase(0, tail_.size() - 64);  // only enough to catch a split close
        if (reasoning_tokens_ >= budget_) { exceeded_ = true; return Stop; }
        return Continue;
    }

    bool closed() const { return closed_; }
    bool exceeded() const { return exceeded_; }
    int reasoning_tokens() const { return reasoning_tokens_; }

    // What the user sees. If the stream ended with the reasoning still open (limit hit, or the model simply stopped),
    // that is the fixed reply: an unfinished thought is never shown as an answer.
    std::string visible() const {
        if (!closed_) return fallback_;
        size_t a = 0, b = answer_.size();
        while (a < b && (answer_[a] == '\n' || answer_[a] == ' ' || answer_[a] == '\r')) ++a;
        while (b > a && (answer_[b - 1] == '\n' || answer_[b - 1] == ' ' || answer_[b - 1] == '\r')) --b;
        return a == b ? fallback_ : answer_.substr(a, b - a);  // a closed reasoning followed by nothing is no answer either
    }

  private:
    static constexpr const char* kClose = "</think>";
    static constexpr size_t kCloseLen = 8;
    int budget_, reasoning_tokens_ = 0;
    bool closed_ = false, exceeded_ = false;
    std::string fallback_, tail_, answer_;
};

}  // namespace kz
