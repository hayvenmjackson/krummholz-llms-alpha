// Synthetic boundary tests. No Atlantic model or real encoder is run here.
#include "../include/atlantic_host.hpp"
#include "../include/atlantic_host.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using atlantic::Cancellation;
using atlantic::GenerationEnd;
using atlantic::Host;
using atlantic::Stage;
using atlantic::Status;

namespace {
int checks = 0;
void expect(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
template<class F> void expect_throw(F&& function, const char* message) {
  bool threw = false;
  try { function(); } catch (...) { threw = true; }
  expect(threw, message);
}

template<class T> void binary(std::ofstream& file, const T& value) {
  file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

struct Fixtures {
  fs::path path = fs::temp_directory_path() /
      ("atlantic-synthetic-host-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  Fixtures() {
    fs::create_directories(path / "judge/models");
    fs::create_directories(path / "safety_screen");
    std::ofstream rules(path / "safety_screen/rules.tsv");
    rules << "fixture-input\tinput\ttest\trefuse\t0\tfixture deny input\n"
          << "fixture-output\toutput\ttest\trefuse\t0\traw:fixture denied output\n";
    rules.close();
    for (const auto* name : {"request.kzj", "reply.kzj"}) {
      std::ofstream out(path / "judge/models" / name, std::ios::binary);
      out.write("KZJ1", 4);
      binary(out, std::uint32_t{8});
      binary(out, 0.0f);
      binary(out, 1.0f);
      for (int i = 0; i < 8; ++i) binary(out, 0.0f);
    }
    make_head(path / "judge/models/request.kze", 384, 0);
    make_head(path / "judge/models/reply.kze", 768, 384);
  }
  ~Fixtures() { std::error_code error; fs::remove_all(path, error); }
  static void make_head(const fs::path& path, std::uint32_t dimensions, std::uint32_t trigger) {
    std::ofstream out(path, std::ios::binary);
    out.write("KZE1", 4);
    binary(out, dimensions);
    binary(out, std::uint32_t{1});
    binary(out, 0.0f);  // No hashed-word component in this synthetic test head.
    binary(out, 0.0f);  // Threshold.
    for (std::uint32_t i = 0; i < dimensions; ++i) binary(out, i == trigger ? 1.0f : 0.0f);
    binary(out, 0.0f);  // Hidden bias.
    binary(out, 1.0f);  // Hidden-to-output weight.
    binary(out, -0.5f);
  }
  atlantic::Assets assets() const { return atlantic::Assets::in_directory(path.string()); }
};

std::vector<float> synthetic_embed(const std::string& text, const atlantic::CancelProbe&) {
  std::vector<float> result(384, 0.0f);
  result[text.find("unsafe_fixture") != std::string::npos ? 0 : 1] = 1.0f;
  return result;
}

atlantic::Generate synthetic_model(std::string reply, int* calls = nullptr) {
  return [reply = std::move(reply), calls](const atlantic::GenerationInput&,
                                        const atlantic::TokenSink& emit,
                                        const atlantic::CancelProbe& cancelled) {
    if (calls) ++*calls;
    // The test uses one character per synthetic tokenizer token. Real model
    // bindings must emit one actual tokenizer token per call instead.
    for (char c : reply) {
      if (cancelled() || !emit(std::string_view(&c, 1))) return GenerationEnd::Stopped;
    }
    return GenerationEnd::Complete;
  };
}

atlantic::Result run(Host& host, const std::string& input = "Hello") {
  Cancellation cancellation;
  return host.run(input, cancellation);
}

void test_order_and_release(Fixtures& files) {
  std::vector<std::string> order;
  Host host(files.assets(),
    [&](const atlantic::GenerationInput& request, const atlantic::TokenSink& sink, const atlantic::CancelProbe&) {
      order.push_back("generate");
      expect(request.input == "Hello", "Generator did not receive the exact checked input");
      expect(request.temperature == 0.7, "Pinned default temperature changed");
      for (const auto* piece : {"private thoughts", "</thi", "nk>", "  Hello there.  "}) {
        if (!sink(piece)) return GenerationEnd::Stopped;
      }
      return GenerationEnd::Complete;
    },
    [&](const std::string& text, const atlantic::CancelProbe& cancelled) {
      order.push_back(text == "Hello" ? "input-embed" : "output-embed");
      return synthetic_embed(text, cancelled);
    });
  const auto result = run(host);
  expect(result.status == Status::Answer && result.text == "Hello there.", "Approved visible reply not released");
  expect(result.model_text_released, "Approved model reply flag missing");
  expect(result.input_rules_passed && result.input_judge_passed &&
         result.output_rules_passed && result.output_judge_passed, "A mandatory check was skipped");
  expect(order == std::vector<std::string>({"input-embed", "generate", "input-embed", "output-embed"}),
         "Input/output judge ordering changed");
  expect(result.text.find("private") == std::string::npos, "Reasoning leaked");
}

void test_inputs(Fixtures& files) {
  int calls = 0;
  Host host(files.assets(), synthetic_model("</think>safe", &calls), synthetic_embed);
  auto result = run(host, "fixture deny input");
  expect(result.status == Status::Blocked && result.stage == Stage::InputRules && calls == 0,
         "Blocked input rule reached generator");
  result = run(host, "unsafe_fixture");
  expect(result.status == Status::Blocked && result.stage == Stage::InputJudge && calls == 0,
         "Blocked input judge reached generator");
  result = run(host, std::string(1600, 'a') + " unsafe_fixture");
  expect(result.status == Status::Blocked && calls == 0, "Late input window escaped judge");
  result = run(host, std::string(65537, 'a'));
  expect(result.status == Status::InvalidRequest && calls == 0, "Oversized input reached generator");
  result = run(host, "  \n\t");
  expect(result.status == Status::InvalidRequest && calls == 0, "Empty input reached generator");
  result = run(host, std::string("bad\0input", 9));
  expect(result.status == Status::InvalidRequest && calls == 0, "NUL input reached generator");
  result = run(host, std::string("\xc0\xaf", 2));
  expect(result.status == Status::InvalidRequest && calls == 0, "Invalid UTF-8 reached generator");
}

void test_missing_and_bad_encoder(Fixtures& files) {
  expect_throw([&] { Host h(files.assets(), {}, synthetic_embed); }, "Missing model accepted");
  expect_throw([&] { Host h(files.assets(), synthetic_model(""), {}); }, "Missing encoder accepted");
  auto missing = files.assets();
  missing.rules += ".missing";
  expect_throw([&] { Host h(missing, synthetic_model(""), synthetic_embed); }, "Missing rules accepted");
  missing = files.assets();
  missing.reply_embedding_head += ".missing";
  expect_throw([&] { Host h(missing, synthetic_model(""), synthetic_embed); }, "Missing head accepted");
  std::vector<atlantic::Embed> failures = {
    [](const std::string&, const atlantic::CancelProbe&) { return std::vector<float>{}; },
    [](const std::string&, const atlantic::CancelProbe&) { return std::vector<float>(2, 1.0f); },
    [](const std::string&, const atlantic::CancelProbe&) { return std::vector<float>(384, 0.0f); },
    [](const std::string&, const atlantic::CancelProbe&) {
      return std::vector<float>(384, std::numeric_limits<float>::quiet_NaN());
    },
    [](const std::string&, const atlantic::CancelProbe&) -> std::vector<float> {
      throw std::runtime_error("SECRET callback diagnostic");
    }
  };
  for (auto& embed : failures) {
    int calls = 0;
    Host host(files.assets(), synthetic_model("</think>SECRET", &calls), embed);
    const auto result = run(host);
    expect(result.status == Status::Unavailable && result.stage == Stage::InputJudge && calls == 0,
           "Unavailable encoder was treated as safe");
    expect(result.text.find("SECRET") == std::string::npos, "Callback diagnostic leaked");
  }
}

void test_output_checks(Fixtures& files) {
  Host rules(files.assets(), synthetic_model("</think>fixture denied output"), synthetic_embed);
  auto result = run(rules);
  expect(result.status == Status::Blocked && result.stage == Stage::OutputRules && !result.model_text_released,
         "Output rule did not stop release");
  expect(result.text.find("fixture") == std::string::npos, "Blocked output leaked");
  Host judge(files.assets(), synthetic_model("</think>unsafe_fixture"), synthetic_embed);
  result = run(judge);
  expect(result.status == Status::Blocked && result.stage == Stage::OutputJudge && !result.model_text_released,
         "Output judge did not stop release");
  Host late(files.assets(), synthetic_model("</think>" + std::string(1600, 'a') + " unsafe_fixture"), synthetic_embed);
  result = run(late);
  expect(result.status == Status::Blocked && result.stage == Stage::OutputJudge,
         "Late output window escaped judge");
  Host unavailable(files.assets(), synthetic_model("</think>SECRET unavailable-output"),
    [](const std::string& text, const atlantic::CancelProbe& cancelled) {
      return text.find("unavailable-output") != std::string::npos ? std::vector<float>{} : synthetic_embed(text, cancelled);
    });
  result = run(unavailable);
  expect(result.status == Status::Unavailable && result.stage == Stage::OutputJudge,
         "Unavailable output judge did not fail closed");
  expect(result.text.find("SECRET") == std::string::npos, "Unchecked output leaked");
}

void test_reasoning_and_errors(Fixtures& files) {
  atlantic::Limits short_budget;
  short_budget.reasoning_tokens = 3;
  Host budget(files.assets(), synthetic_model("never closed"), synthetic_embed, short_budget);
  auto result = run(budget);
  expect(result.status == Status::Abstained && result.generated_tokens == 3 && !result.model_text_released,
         "Reasoning budget did not stop with abstention");
  expect(result.output_rules_passed && result.output_judge_passed, "Fixed abstention bypassed output checks");
  Host unfinished(files.assets(), synthetic_model("not closed"), synthetic_embed);
  result = run(unfinished);
  expect(result.status == Status::Abstained && !result.model_text_released, "Unfinished reasoning leaked");
  Host empty(files.assets(), synthetic_model("</think>   "), synthetic_embed);
  expect(run(empty).status == Status::Abstained, "Empty final answer not treated as abstention");
  Host reopened(files.assets(), synthetic_model("</think>answer<think>SECRET"), synthetic_embed);
  result = run(reopened);
  expect(result.status == Status::GenerationFailed && result.text.find("SECRET") == std::string::npos,
         "Reopened reasoning leaked");
  Host throws(files.assets(),
    [](const atlantic::GenerationInput&, const atlantic::TokenSink& emit, const atlantic::CancelProbe&) -> GenerationEnd {
      emit("</think>SECRET PARTIAL");
      throw std::runtime_error("SECRET ERROR");
    }, synthetic_embed);
  result = run(throws);
  expect(result.status == Status::GenerationFailed && result.text.find("SECRET") == std::string::npos,
         "Failed generation leaked partial output");
  Host stops(files.assets(),
    [](const atlantic::GenerationInput&, const atlantic::TokenSink& emit, const atlantic::CancelProbe&) {
      emit("</think>SECRET PARTIAL");
      return GenerationEnd::Stopped;
    }, synthetic_embed);
  result = run(stops);
  expect(result.status == Status::GenerationFailed && !result.model_text_released, "Incomplete generation released");
  Host invalid(files.assets(), synthetic_model(std::string("</think>\xc0\xaf", 10)), synthetic_embed);
  expect(run(invalid).status == Status::GenerationFailed, "Invalid output UTF-8 released");
  atlantic::Limits token_limit;
  token_limit.reasoning_tokens = 1;
  token_limit.generated_tokens = 2;
  Host too_many(files.assets(),
    [](const atlantic::GenerationInput&, const atlantic::TokenSink& sink, const atlantic::CancelProbe&) {
      sink("</think>"); sink("SECRET"); sink(" TOO MUCH");
      return GenerationEnd::Complete;
    }, synthetic_embed, token_limit);
  result = run(too_many);
  expect(result.status == Status::GenerationFailed && !result.model_text_released, "Token overflow released");
}

void test_cancellation_and_busy(Fixtures& files) {
  Cancellation before;
  before.cancel();
  int calls = 0;
  Host host(files.assets(), synthetic_model("</think>SECRET", &calls), synthetic_embed);
  auto result = host.run("Hello", before);
  expect(result.status == Status::Cancelled && calls == 0, "Pre-cancelled request ran model");
  Cancellation during;
  Host cancels(files.assets(),
    [&](const atlantic::GenerationInput&, const atlantic::TokenSink& emit, const atlantic::CancelProbe& cancelled) {
      emit("</think>SECRET PARTIAL");
      during.cancel();
      expect(cancelled(), "Model cancel probe did not update");
      expect(!emit("SECRET SECOND"), "Sink accepted tokens after cancellation");
      return GenerationEnd::Stopped;
    }, synthetic_embed);
  result = cancels.run("Hello", during);
  expect(result.status == Status::Cancelled && result.text.find("SECRET") == std::string::npos,
         "Cancelled output leaked");
  Cancellation final_check;
  Host cancel_after(files.assets(), synthetic_model("</think>SECRET FINAL"),
    [&](const std::string& text, const atlantic::CancelProbe& cancelled) {
      if (text.find("SECRET") != std::string::npos) final_check.cancel();
      return synthetic_embed(text, cancelled);
    });
  result = cancel_after.run("Hello", final_check);
  expect(result.status == Status::Cancelled && !result.model_text_released, "Cancellation during final check ignored");
  Host* running = nullptr;
  Host reentrant(files.assets(),
    [&](const atlantic::GenerationInput&, const atlantic::TokenSink& emit, const atlantic::CancelProbe&) {
      expect(run(*running).status == Status::Busy, "Concurrent run did not return Busy");
      emit("</think>Done");
      return GenerationEnd::Complete;
    }, synthetic_embed);
  running = &reentrant;
  expect(run(reentrant).status == Status::Answer, "Outer request failed after Busy response");
}

int c_generate(void*, const char*, size_t, double, int, atlantic_emit_fn emit,
               atlantic_cancelled_fn, void* context) {
  const std::string text = "</think>C ABI answer";
  return emit(context, text.data(), text.size()) ? 0 : 1;
}
int c_embed(void*, const char* text, size_t bytes, float* output, size_t dimensions,
            atlantic_cancelled_fn, void*) {
  auto values = synthetic_embed(std::string(text, bytes), [] { return false; });
  if (dimensions != values.size()) return 1;
  for (size_t i = 0; i < dimensions; ++i) output[i] = values[i];
  return 0;
}

void test_c_abi(Fixtures& files) {
  atlantic_host* host = nullptr;
  expect(atlantic_host_create(files.path.string().c_str(), c_generate, c_embed, nullptr, &host) == 0,
         "C ABI initialization failed");
  auto* cancellation = atlantic_cancellation_create();
  expect(cancellation != nullptr, "C cancellation allocation failed");
  auto* result = atlantic_host_run(host, "Hello", 5, cancellation, -1.0);
  expect(result != nullptr && std::string(atlantic_result_text(result)) == "C ABI answer", "C ABI result failed");
  expect(std::string(atlantic_result_status(result)) == "answer" && atlantic_result_released_model_text(result),
         "C ABI status failed");
  atlantic_result_destroy(result);
  atlantic_cancellation_cancel(cancellation);
  result = atlantic_host_run(host, "Hello", 5, cancellation, -1.0);
  expect(std::string(atlantic_result_status(result)) == "cancelled", "C ABI cancellation failed");
  atlantic_result_destroy(result);
  atlantic_cancellation_destroy(cancellation);
  atlantic_host_destroy(host);
  expect(atlantic_host_create(files.path.string().c_str(), nullptr, c_embed, nullptr, &host) == 1 && !host,
         "C ABI accepted missing generator");
}

void test_real_bundle_loads(const std::string& directory) {
  int generation_calls = 0;
  Host pinned(atlantic::Assets::in_directory(directory), synthetic_model("SECRET", &generation_calls),
    [](const std::string&, const atlantic::CancelProbe&) { return std::vector<float>{}; });
  auto result = run(pinned, "Please tell me a little about trees.");
  expect(result.status == Status::Unavailable && generation_calls == 0,
         "Real pinned heads accepted missing embeddings");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Fixtures files;
    test_order_and_release(files);
    test_inputs(files);
    test_missing_and_bad_encoder(files);
    test_output_checks(files);
    test_reasoning_and_errors(files);
    test_cancellation_and_busy(files);
    test_c_abi(files);
    if (argc > 1) test_real_bundle_loads(argv[1]);
    std::cout << "PASS: " << checks << " synthetic boundary assertions. No actual model inference tested.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
