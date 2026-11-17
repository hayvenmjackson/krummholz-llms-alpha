// Copyright 2026 Krummholz AI LLC. Licensed under Apache-2.0.
#include "../include/atlantic_host.hpp"

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "../vendor/krummholz-harness/judge/judge.hpp"
#include "../vendor/krummholz-harness/reasoning_budget/reasoning_budget.hpp"
#include "../vendor/krummholz-harness/sampling/sampling.hpp"

namespace atlantic {
namespace {
constexpr const char* kUnavailable = "The safety check is unavailable, so I can't show a reply right now.";
constexpr const char* kCancelled = "Reply cancelled.";
constexpr const char* kFailure = "The reply could not be completed. Please try again.";
constexpr const char* kInvalid = "This request could not be processed. Please shorten it and try again.";
constexpr const char* kRefusal = "I can't help with that.";

// Check UTF-8 before the byte-windowing upstream judge sees it. Reject NUL and
// invalid/overlong sequences rather than treating an incomplete input as safe.
bool valid_utf8(std::string_view s) {
  for (std::size_t i = 0; i < s.size();) {
    auto c = static_cast<unsigned char>(s[i]);
    if (c == 0) return false;
    if (c < 0x80) { ++i; continue; }
    std::size_t n = 0;
    std::uint32_t cp = 0;
    if (c >= 0xC2 && c <= 0xDF) { n = 2; cp = c & 0x1F; }
    else if (c >= 0xE0 && c <= 0xEF) { n = 3; cp = c & 0x0F; }
    else if (c >= 0xF0 && c <= 0xF4) { n = 4; cp = c & 0x07; }
    else return false;
    if (i + n > s.size()) return false;
    for (std::size_t j = 1; j < n; ++j) {
      auto next = static_cast<unsigned char>(s[i + j]);
      if ((next & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (next & 0x3F);
    }
    if ((n == 3 && cp < 0x800) || (n == 4 && cp < 0x10000) ||
        (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) return false;
    i += n;
  }
  return true;
}

bool blank(std::string_view s) {
  return s.find_first_not_of(" \r\n\t") == std::string_view::npos;
}

void fixed(Result& result, Status status, Stage stage, const char* text) {
  result.status = status;
  result.stage = stage;
  result.text = text;
  result.model_text_released = false;
}
}  // namespace

Assets Assets::in_directory(const std::string& directory) {
  const auto prefix = directory + "/";
  return {prefix + "safety_screen/rules.tsv", prefix + "judge/models/request.kzj",
          prefix + "judge/models/reply.kzj", prefix + "judge/models/request.kze",
          prefix + "judge/models/reply.kze"};
}

class Host::Impl {
 public:
  Impl(Assets assets, Generate generate, Embed embed, Limits limits)
      : generate_(std::move(generate)), embed_(std::move(embed)), limits_(limits),
        judge_(assets.request_linear, assets.reply_linear,
               assets.request_embedding_head, assets.reply_embedding_head,
               [this](const std::string& window) {
                 if (!active_ || active_->cancelled() || !embed_) {
                   throw std::runtime_error("embedding unavailable");
                 }
                 const CancelProbe cancelled = [this] { return !active_ || active_->cancelled(); };
                 return embed_(window, cancelled);
               }),
        gate_(assets.rules, [this](const std::string& request, const std::string& reply) {
          return judge_(request, reply);
        }) {
    if (!generate_ || !embed_) throw std::runtime_error("model and encoder callbacks are required");
    if (limits_.reasoning_tokens < 1 || limits_.reasoning_tokens > 1000000 ||
        limits_.generated_tokens < limits_.reasoning_tokens || limits_.generated_tokens > 1000000 ||
        limits_.input_bytes < 1 || limits_.input_bytes > 1024 * 1024 ||
        limits_.generated_bytes < 1 || limits_.generated_bytes > 16 * 1024 * 1024 ||
        limits_.token_piece_bytes < 1 || limits_.token_piece_bytes > limits_.generated_bytes) {
      throw std::runtime_error("invalid host limits");
    }
  }

  Result run(const std::string& input, Cancellation& cancellation, double temperature) noexcept {
    Result result;
    try {
      std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
      if (!lock.owns_lock()) {
        fixed(result, Status::Busy, Stage::None, "Another reply is in progress.");
        return result;
      }
      active_ = &cancellation;
      struct Reset { Cancellation*& active; ~Reset() { active = nullptr; } } reset{active_};
      const auto cancelled = [&] { return cancellation.cancelled(); };
      if (cancelled()) {
        fixed(result, Status::Cancelled, Stage::None, kCancelled);
        return result;
      }
      if (input.size() > limits_.input_bytes || blank(input) || !valid_utf8(input)) {
        fixed(result, Status::InvalidRequest, Stage::InputRules, kInvalid);
        return result;
      }
      result.stage = Stage::InputRules;
      const auto before = gate_.before(input);
      if (!before.allowed) {
        fixed(result, Status::Blocked, Stage::InputRules, kRefusal);
        result.text = before.text;
        result.rule_id = before.rule;
        return result;
      }
      result.input_rules_passed = true;
      if (cancelled()) {
        fixed(result, Status::Cancelled, result.stage, kCancelled);
        return result;
      }
      result.stage = Stage::InputJudge;
      const auto input_verdict = judge_.request_verdict(input);
      if (cancelled()) {
        fixed(result, Status::Cancelled, result.stage, kCancelled);
        return result;
      }
      if (input_verdict != kz::Judged::Safe) {
        fixed(result, input_verdict == kz::Judged::Harmful ? Status::Blocked : Status::Unavailable,
              Stage::InputJudge, input_verdict == kz::Judged::Harmful ? kRefusal : kUnavailable);
        return result;
      }
      result.input_judge_passed = true;
      result.stage = Stage::Generation;
      kz::ReasoningBudget budget(limits_.reasoning_tokens);
      std::size_t bytes = 0;
      bool reasoning_stop = false;
      bool limit_stop = false;
      bool stopped = false;
      const TokenSink sink = [&](std::string_view piece) {
        if (stopped || cancelled()) { stopped = true; return false; }
        if (piece.size() > limits_.token_piece_bytes ||
            piece.size() > limits_.generated_bytes - bytes ||
            result.generated_tokens >= limits_.generated_tokens) {
          limit_stop = true;
          stopped = true;
          return false;
        }
        bytes += piece.size();
        ++result.generated_tokens;
        if (budget.feed(std::string(piece), 1) == kz::ReasoningBudget::Stop) {
          reasoning_stop = true;
          stopped = true;
          return false;
        }
        return true;
      };
      const GenerationInput generation{input, kz::Sampling::resolve(temperature).temperature,
                                       limits_.generated_tokens};
      const auto end = generate_(generation, sink, cancelled);
      if (cancelled()) {
        fixed(result, Status::Cancelled, Stage::Generation, kCancelled);
        return result;
      }
      if (limit_stop || end == GenerationEnd::Failed ||
          (!reasoning_stop && end != GenerationEnd::Complete)) {
        fixed(result, Status::GenerationFailed, Stage::Generation, kFailure);
        return result;
      }
      const auto visible = budget.visible();
      // A token may contain only part of a UTF-8 character, so validate only
      // after buffering. Reopened reasoning is withheld rather than exposed.
      if (!valid_utf8(visible) || visible.find("<think>") != std::string::npos ||
          visible.find("</think>") != std::string::npos) {
        fixed(result, Status::GenerationFailed, Stage::Generation, kFailure);
        return result;
      }
      result.stage = Stage::OutputRules;
      const auto after = gate_.after(input, visible);
      if (cancelled()) {
        fixed(result, Status::Cancelled, Stage::OutputJudge, kCancelled);
        return result;
      }
      result.output_rules_passed = after.stopped_by != "output-rule";
      if (!after.allowed) {
        const bool unavailable = after.stopped_by == "judge-unavailable";
        fixed(result, unavailable ? Status::Unavailable : Status::Blocked,
              after.stopped_by == "output-rule" ? Stage::OutputRules : Stage::OutputJudge,
              unavailable ? kUnavailable : kRefusal);
        result.text = after.text;
        result.rule_id = after.rule;
        return result;
      }
      result.output_judge_passed = true;
      result.stage = Stage::Finished;
      result.text = after.text;
      // The upstream budget also substitutes a fixed abstention for a closed
      // reasoning segment with an empty answer. Recognize that case by text.
      const kz::ReasoningBudget empty_budget(limits_.reasoning_tokens);
      const bool abstained = !budget.closed() || visible == empty_budget.visible();
      result.status = abstained ? Status::Abstained : Status::Answer;
      result.model_text_released = !abstained;
      return result;
    } catch (...) {
      // Callback exception strings can contain raw prompt/model text. Never
      // propagate them to the UI, logs, or FFI result.
      fixed(result, result.stage == Stage::Generation ? Status::GenerationFailed : Status::Unavailable,
            result.stage, result.stage == Stage::Generation ? kFailure : kUnavailable);
      return result;
    }
  }

 private:
  Generate generate_;
  Embed embed_;
  Limits limits_;
  Cancellation* active_ = nullptr;
  kz::EmbeddingJudge judge_;
  kz::SafetyGate gate_;
  std::mutex mutex_;
};

Host::Host(Assets assets, Generate generate, Embed embed, Limits limits)
    : impl_(std::make_unique<Impl>(std::move(assets), std::move(generate), std::move(embed), limits)) {}
Host::~Host() = default;
Result Host::run(const std::string& input, Cancellation& cancellation, double temperature) noexcept {
  return impl_->run(input, cancellation, temperature);
}

const char* status_name(Status status) noexcept {
  switch (status) {
    case Status::Answer: return "answer";
    case Status::Abstained: return "abstained";
    case Status::Blocked: return "blocked";
    case Status::Unavailable: return "unavailable";
    case Status::Cancelled: return "cancelled";
    case Status::InvalidRequest: return "invalid-request";
    case Status::GenerationFailed: return "generation-failed";
    case Status::Busy: return "busy";
  }
  return "unavailable";
}

const char* stage_name(Stage stage) noexcept {
  switch (stage) {
    case Stage::None: return "none";
    case Stage::InputRules: return "input-rules";
    case Stage::InputJudge: return "input-judge";
    case Stage::Generation: return "generation";
    case Stage::OutputRules: return "output-rules";
    case Stage::OutputJudge: return "output-judge";
    case Stage::Finished: return "finished";
  }
  return "none";
}
}  // namespace atlantic
