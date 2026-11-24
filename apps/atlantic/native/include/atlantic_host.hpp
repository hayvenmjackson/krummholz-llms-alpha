// Copyright 2026 Krummholz AI LLC. Licensed under Apache-2.0.
#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace atlantic {

// One cancellation object per request. It must outlive the synchronous run().
class Cancellation final {
 public:
  void cancel() noexcept { cancelled_.store(true, std::memory_order_relaxed); }
  bool cancelled() const noexcept { return cancelled_.load(std::memory_order_relaxed); }

 private:
  std::atomic<bool> cancelled_{false};
};

struct Assets {
  std::string rules;
  std::string request_linear;
  std::string reply_linear;
  std::string request_embedding_head;
  std::string reply_embedding_head;

  // directory is the installed, verified krummholz-harness bundle root.
  static Assets in_directory(const std::string& directory);
};

struct Limits {
  int reasoning_tokens = 1024;
  int generated_tokens = 4096;
  std::size_t input_bytes = 64 * 1024;
  std::size_t generated_bytes = 1024 * 1024;
  std::size_t token_piece_bytes = 16 * 1024;
};

struct GenerationInput {
  // The complete, already checked input. The adapter must not append unchecked
  // messages, retrieved text, or memory after this boundary.
  std::string input;
  double temperature = 0.7;
  int max_generated_tokens = 4096;
};

enum class GenerationEnd { Complete, Stopped, Failed };

// Called once per generated tokenizer token, including empty text pieces.
// false means stop promptly. Never forward these pieces to the UI or logs.
using TokenSink = std::function<bool(std::string_view)>;
using CancelProbe = std::function<bool()>;
using Generate = std::function<GenerationEnd(const GenerationInput&, const TokenSink&, const CancelProbe&)>;

// Must provide exact bge-small-en-v1.5 first-token-pool vectors, 384 dimensions.
// The pinned judge performs L2 normalization and rejects invalid vectors.
using Embed = std::function<std::vector<float>(const std::string&, const CancelProbe&)>;

enum class Status {
  Answer,
  Abstained,
  Blocked,
  Unavailable,
  Cancelled,
  InvalidRequest,
  GenerationFailed,
  Busy,
};

enum class Stage { None, InputRules, InputJudge, Generation, OutputRules, OutputJudge, Finished };

struct Result {
  Status status = Status::Unavailable;
  Stage stage = Stage::None;
  // This is the only user-visible text. It is a fixed status/refusal or a
  // completed reply accepted by both the output rules and embedding judge.
  std::string text;
  std::string rule_id;
  bool model_text_released = false;
  bool input_rules_passed = false;
  bool input_judge_passed = false;
  bool output_rules_passed = false;
  bool output_judge_passed = false;
  int generated_tokens = 0;
};

class Host final {
 public:
  // Throws if assets are unreadable or either required native callback is
  // missing. There is no rules-only, raw-generation, or permissive mode.
  Host(Assets assets, Generate generate, Embed embed, Limits limits = {});
  ~Host();
  Host(const Host&) = delete;
  Host& operator=(const Host&) = delete;
  Host(Host&&) = delete;
  Host& operator=(Host&&) = delete;

  // Synchronous. Run on a native worker, never the UI thread. Concurrent calls
  // to one Host return Busy. Cancel from another thread using cancellation.
  // A model callback must return Complete only for a normal end-of-generation.
  Result run(const std::string& complete_input, Cancellation& cancellation,
             double requested_temperature = -1.0) noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

const char* status_name(Status status) noexcept;
const char* stage_name(Stage stage) noexcept;

}  // namespace atlantic
