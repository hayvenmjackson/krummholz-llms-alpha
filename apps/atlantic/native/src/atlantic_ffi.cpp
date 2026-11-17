// Copyright 2026 Krummholz AI LLC. Licensed under Apache-2.0.
#include "../include/atlantic_host.h"
#include "../include/atlantic_host.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

struct atlantic_host { std::unique_ptr<atlantic::Host> value; };
struct atlantic_cancellation { atlantic::Cancellation value; };
struct atlantic_result { atlantic::Result value; };

namespace {
struct SinkContext {
  const atlantic::TokenSink& sink;
  const atlantic::CancelProbe& cancelled;
  bool failed = false;
};
int emit_piece(void* opaque, const char* piece, size_t size) {
  auto& context = *static_cast<SinkContext*>(opaque);
  if (!piece && size != 0) { context.failed = true; return 0; }
  try { return context.sink(std::string_view(piece ? piece : "", size)) ? 1 : 0; }
  catch (...) { context.failed = true; return 0; }
}
int generation_cancelled(void* opaque) {
  return static_cast<SinkContext*>(opaque)->cancelled() ? 1 : 0;
}
int embedding_cancelled(void* opaque) {
  return (*static_cast<const atlantic::CancelProbe*>(opaque))() ? 1 : 0;
}
}  // namespace

extern "C" {
int atlantic_host_create(const char* directory, atlantic_generate_fn generate,
                        atlantic_embed_fn embed, void* user_data, atlantic_host** output) {
  if (output) *output = nullptr;
  if (!output || !directory || !*directory || !generate || !embed) return 1;
  try {
    atlantic::Generate model = [=](const atlantic::GenerationInput& input,
                                 const atlantic::TokenSink& sink,
                                 const atlantic::CancelProbe& cancelled) {
      SinkContext context{sink, cancelled, false};
      const auto result = generate(user_data, input.input.data(), input.input.size(),
                                   input.temperature, input.max_generated_tokens,
                                   emit_piece, generation_cancelled, &context);
      if (context.failed || result < 0 || result > 1) return atlantic::GenerationEnd::Failed;
      return result == 0 ? atlantic::GenerationEnd::Complete : atlantic::GenerationEnd::Stopped;
    };
    atlantic::Embed encoder = [=](const std::string& window, const atlantic::CancelProbe& cancelled) {
      std::vector<float> output_vector(384);
      if (cancelled()) return std::vector<float>{};
      // C callback receives a non-owning pointer valid only during this call.
      if (embed(user_data, window.data(), window.size(), output_vector.data(), output_vector.size(),
                embedding_cancelled, const_cast<atlantic::CancelProbe*>(&cancelled)) != 0) {
        return std::vector<float>{};
      }
      return output_vector;
    };
    auto host = std::make_unique<atlantic_host>();
    host->value = std::make_unique<atlantic::Host>(atlantic::Assets::in_directory(directory),
                                                 std::move(model), std::move(encoder));
    *output = host.release();
    return 0;
  } catch (...) { return 2; }
}

void atlantic_host_destroy(atlantic_host* host) { delete host; }
atlantic_cancellation* atlantic_cancellation_create(void) {
  try { return new atlantic_cancellation; } catch (...) { return nullptr; }
}
void atlantic_cancellation_cancel(atlantic_cancellation* cancellation) {
  if (cancellation) cancellation->value.cancel();
}
void atlantic_cancellation_destroy(atlantic_cancellation* cancellation) { delete cancellation; }

atlantic_result* atlantic_host_run(atlantic_host* host, const char* input, size_t bytes,
                                 atlantic_cancellation* cancellation, double temperature) {
  if (!host || !cancellation || (!input && bytes != 0)) return nullptr;
  // Reject oversized foreign-language buffers before allocating/copying.
  if (bytes > 64 * 1024) {
    try {
      auto result = std::make_unique<atlantic_result>();
      result->value.status = atlantic::Status::InvalidRequest;
      result->value.stage = atlantic::Stage::InputRules;
      result->value.text = "This request could not be processed. Please shorten it and try again.";
      return result.release();
    } catch (...) { return nullptr; }
  }
  try {
    auto result = std::make_unique<atlantic_result>();
    result->value = host->value->run(std::string(input ? input : "", bytes), cancellation->value, temperature);
    return result.release();
  } catch (...) { return nullptr; }
}

void atlantic_result_destroy(atlantic_result* result) { delete result; }
const char* atlantic_result_text(const atlantic_result* result) {
  return result ? result->value.text.c_str() : "The local runtime is unavailable.";
}
const char* atlantic_result_status(const atlantic_result* result) {
  return result ? atlantic::status_name(result->value.status) : "unavailable";
}
const char* atlantic_result_stage(const atlantic_result* result) {
  return result ? atlantic::stage_name(result->value.stage) : "none";
}
const char* atlantic_result_rule(const atlantic_result* result) {
  return result ? result->value.rule_id.c_str() : "";
}
int atlantic_result_released_model_text(const atlantic_result* result) {
  return result && result->value.model_text_released ? 1 : 0;
}
int atlantic_result_generated_tokens(const atlantic_result* result) {
  return result ? result->value.generated_tokens : 0;
}
}  // extern "C"
