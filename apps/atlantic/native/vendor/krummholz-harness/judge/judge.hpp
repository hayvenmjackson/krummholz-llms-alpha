// Krummholz harness, layer 2 judges for kz::SafetyGate (../safety_gate/safety_gate.hpp). Header only, C++17.
//
//   kz::LinearJudge      our own small judge: two logistic regressions over hashed words and word pairs, trained by
//                        judge_train.py. About 1 MB each, microseconds per reply, runs anywhere including a phone.
//   kz::EmbeddingJudge   the linear judge plus meaning: a small network over sentence embeddings (bge-small-en-v1.5,
//                        MIT, 33 million weights) added to the hashed-word score. Trained by judge_train_embed.py.
//                        The harness supplies the function that embeds a text window (llama.cpp).
//   kz::ClassifierJudge  the desktop judge: a classifier model (HarmBench classifier, MIT) asked "yes" or "no".
//                        The harness supplies the function that runs the model (llama.cpp), so no model library here.
//   kz::any_harmful      joins judges: harmful if any says so; unavailable if none says harmful and any cannot answer.
//
// The features of LinearJudge must match judge_train.py exactly, and the windows of EmbeddingJudge must match
// judge_embed.py; parity.py and parity_embed.py check C++ against Python.
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "../safety_gate/safety_gate.hpp"

namespace kz {

class LinearModel {
  public:
    explicit LinearModel(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        char magic[4];
        uint32_t n = 0;
        if (!in.read(magic, 4) || std::memcmp(magic, "KZJ1", 4) != 0 || !in.read(reinterpret_cast<char*>(&n), 4) ||
            !in.read(reinterpret_cast<char*>(&bias_), 4) || !in.read(reinterpret_cast<char*>(&threshold_), 4) || n == 0 || n > (1u << 24))
            throw std::runtime_error("judge model unreadable: " + path);
        w_.resize(n);
        if (!in.read(reinterpret_cast<char*>(w_.data()), static_cast<std::streamsize>(n) * 4)) throw std::runtime_error("judge model cut short: " + path);
        bool finite = std::isfinite(bias_) && std::isfinite(threshold_);
        for (float f : w_) finite = finite && std::isfinite(f);
        if (!finite) throw std::runtime_error("judge model holds a value that is not a number: " + path);  // such a model would call everything safe
    }
    // fields: (letter, text) pairs, "q" for the request and "a" for the reply
    double score(const std::vector<std::pair<char, const std::string*>>& fields) const {
        std::unordered_set<uint32_t> on;
        for (const auto& f : fields) {
            const auto w = words(*f.second);
            for (size_t i = 0; i < w.size(); ++i) {
                std::string key = std::string(1, f.first) + ":" + w[i];
                on.insert(static_cast<uint32_t>(fnv(key) % w_.size()));
                if (i + 1 < w.size()) on.insert(static_cast<uint32_t>(fnv(key + " " + w[i + 1]) % w_.size()));
            }
        }
        if (on.empty()) return bias_;
        double sum = 0;
        for (uint32_t i : on) sum += w_[i];
        return bias_ + sum / std::sqrt(static_cast<double>(on.size()));
    }
    double threshold() const { return threshold_; }

  private:
    static std::vector<std::string> words(const std::string& text) {
        std::vector<std::string> out;
        std::string cur;
        for (unsigned char c : text) {
            if (c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c + 32);
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c >= 128) cur.push_back(static_cast<char>(c));
            else if (!cur.empty()) {
                out.push_back(cur); cur.clear();
                if (out.size() == 600) return out;
            }
        }
        if (!cur.empty() && out.size() < 600) out.push_back(cur);
        return out;
    }
    static uint64_t fnv(const std::string& s) {
        uint64_t h = 0xCBF29CE484222325ull;
        for (unsigned char c : s) h = (h ^ c) * 0x100000001B3ull;
        return h;
    }
    std::vector<float> w_;
    float bias_ = 0, threshold_ = 0;
};

class LinearJudge {
  public:
    LinearJudge(const std::string& request_model, const std::string& reply_model) : request_(request_model), reply_(reply_model) {}
    double request_score(const std::string& request) const { return request_.score({{'q', &request}}); }
    double reply_score(const std::string& request, const std::string& reply) const { return reply_.score({{'q', &request}, {'a', &reply}}); }
    bool request_harmful(const std::string& request) const { return request_score(request) >= request_.threshold(); }
    // As a judge for the gate: the reply model decides. The request model is for the harness to use before the model runs.
    Judged operator()(const std::string& request, const std::string& reply) const {
        return reply_score(request, reply) >= reply_.threshold() ? Judged::Harmful : Judged::Safe;
    }

  private:
    LinearModel request_, reply_;
};

// embed(window) returns the sentence embedding of one text window: bge-small-en-v1.5, first-token pooling. It may throw.
using Embed = std::function<std::vector<float>(const std::string& window)>;

// Cuts a text into windows of at most 500 bytes, as many as it takes, so no part of the text goes unread. 500 bytes are
// never more than 502 tokens, under the embedding model's 512. A cut moves back to a space or line break if one lies in
// the last 100 bytes, else by up to 3 bytes to the start of a UTF-8 character. Same as windows() in judge_embed.py.
inline std::vector<std::string> windows(const std::string& t) {
    std::vector<std::string> out;
    const auto space = [&](size_t k) { return t[k] == ' ' || t[k] == '\n' || t[k] == '\t' || t[k] == '\r'; };
    const auto inside = [&](size_t k) { return (static_cast<unsigned char>(t[k]) & 0xC0) == 0x80; };
    for (size_t i = 0; i < t.size();) {
        size_t j = std::min(t.size(), i + 500);
        if (j < t.size()) {
            size_t k = j;
            while (k > j - 100 && !space(k)) --k;
            if (space(k)) j = k;
            else if (!inside(j)) {}
            else if (!inside(j - 1)) j -= 1;
            else if (!inside(j - 2)) j -= 2;
            else if (!inside(j - 3)) j -= 3;  // otherwise this is not UTF-8: cut where we are
        }
        out.push_back(t.substr(i, j - i));
        i = j;
    }
    if (out.empty()) out.emplace_back();
    return out;
}

// One hidden layer (ReLU) and one output, written by judge_train_embed.py.
class EmbedHead {
  public:
    explicit EmbedHead(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        char magic[4];
        const auto read = [&](void* p, size_t n) { return static_cast<bool>(in.read(static_cast<char*>(p), static_cast<std::streamsize>(n))); };
        if (!read(magic, 4) || std::memcmp(magic, "KZE1", 4) != 0 || !read(&in_, 4) || !read(&hidden_, 4) || !read(&alpha_, 4) || !read(&threshold_, 4) ||
            in_ == 0 || in_ > 8192 || hidden_ == 0 || hidden_ > 8192)
            throw std::runtime_error("judge head unreadable: " + path);
        w1_.resize(static_cast<size_t>(in_) * hidden_); b1_.resize(hidden_); w2_.resize(hidden_);
        if (!read(w1_.data(), w1_.size() * 4) || !read(b1_.data(), b1_.size() * 4) || !read(w2_.data(), w2_.size() * 4) || !read(&b2_, 4))
            throw std::runtime_error("judge head cut short: " + path);
        bool finite = std::isfinite(alpha_) && std::isfinite(threshold_) && std::isfinite(b2_);
        for (const auto* v : {&w1_, &b1_, &w2_})
            for (float f : *v) finite = finite && std::isfinite(f);
        if (!finite) throw std::runtime_error("judge head holds a value that is not a number: " + path);
    }
    double score(const std::vector<float>& x) const {  // x.size() == inputs()
        double out = b2_;
        for (uint32_t h = 0; h < hidden_; ++h) {
            double z = b1_[h];
            const float* row = &w1_[static_cast<size_t>(h) * in_];
            for (uint32_t i = 0; i < in_; ++i) z += static_cast<double>(row[i]) * x[i];
            if (z > 0) out += z * w2_[h];
        }
        return out;
    }
    uint32_t inputs() const { return in_; }
    double alpha() const { return alpha_; }
    double threshold() const { return threshold_; }

  private:
    uint32_t in_ = 0, hidden_ = 0;
    float alpha_ = 0, threshold_ = 0, b2_ = 0;
    std::vector<float> w1_, b1_, w2_;
};

class EmbeddingJudge {
  public:
    EmbeddingJudge(const std::string& request_kzj, const std::string& reply_kzj, const std::string& request_kze, const std::string& reply_kze, Embed embed)
        : hq_(request_kzj), ha_(reply_kzj), eq_(request_kze), ea_(reply_kze), embed_(std::move(embed)) {
        if (!embed_ || ea_.inputs() != 2 * eq_.inputs()) throw std::runtime_error("embedding judge: embedding function missing or heads do not fit");
    }
    // No value: an embedding could not be had or a score is not a finite number. The caller must treat that as no verdict.
    std::optional<double> request_score(const std::string& request) const {
        std::vector<float> lead;
        const auto best = best_request_window(request, lead);
        if (!best) return std::nullopt;
        return finite(*best + eq_.alpha() * hq_.score({{'q', &request}}));
    }
    std::optional<double> reply_score(const std::string& request, const std::string& reply) const {
        std::vector<float> x;  // the request window the request head scores highest, then one reply window
        if (!best_request_window(request, x)) return std::nullopt;
        const size_t d = x.size();
        x.resize(2 * d);
        double best = -std::numeric_limits<double>::infinity();
        for (const auto& w : windows(reply)) {
            const auto v = vec(w);
            if (v.empty()) return std::nullopt;
            std::copy(v.begin(), v.end(), x.begin() + static_cast<std::ptrdiff_t>(d));
            const double s = ea_.score(x);
            if (!std::isfinite(s)) return std::nullopt;
            best = std::max(best, s);
        }
        return finite(best + ea_.alpha() * ha_.score({{'q', &request}, {'a', &reply}}));
    }
    // For the harness to use before the model runs.
    Judged request_verdict(const std::string& request) const { return verdict(request_score(request), eq_.threshold()); }
    // As a judge for the gate: the reply head decides.
    Judged operator()(const std::string& request, const std::string& reply) const { return verdict(reply_score(request, reply), ea_.threshold()); }

  private:
    static std::optional<double> finite(double s) { return std::isfinite(s) ? std::optional<double>(s) : std::nullopt; }
    // Safe only for a finite score below the threshold.
    static Judged verdict(const std::optional<double>& s, double threshold) {
        return !s || !std::isfinite(*s) ? Judged::Unavailable : *s < threshold ? Judged::Safe : Judged::Harmful;
    }
    // Highest request-head score over the request's windows; `lead` gets that window's embedding.
    std::optional<double> best_request_window(const std::string& request, std::vector<float>& lead) const {
        double best = -std::numeric_limits<double>::infinity();
        for (const auto& w : windows(request)) {
            auto v = vec(w);
            if (v.empty()) return std::nullopt;
            const double s = eq_.score(v);
            if (!std::isfinite(s)) return std::nullopt;
            if (s > best) { best = s; lead = std::move(v); }
        }
        return best;
    }
    // The window's embedding scaled to length 1; empty if the function fails, returns the wrong size, or returns numbers
    // that are not finite or are all (nearly) zero.
    std::vector<float> vec(const std::string& window) const {
        std::vector<float> v;
        try { v = embed_(window); } catch (...) { return {}; }
        double n = 0;
        for (float f : v) n += static_cast<double>(f) * f;
        if (v.size() != eq_.inputs() || !std::isfinite(n) || n < 1e-12) return {};
        const float scale = static_cast<float>(1.0 / std::sqrt(n));
        for (float& f : v) {
            f *= scale;
            if (!std::isfinite(f)) return {};
        }
        return v;
    }
    LinearModel hq_, ha_;
    EmbedHead eq_, ea_;
    Embed embed_;
};

// complete(prompt, max_tokens) runs the classifier model and returns its text; it may throw.
using Complete = std::function<std::string(const std::string& prompt, int max_tokens)>;

class ClassifierJudge {
  public:
    // template_path: the classifier's prompt with {behavior} and {generation} (export_prompt.py writes it).
    ClassifierJudge(const std::string& template_path, Complete complete) : complete_(std::move(complete)) {
        std::ifstream in(template_path, std::ios::binary);
        template_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        if (template_.find("{behavior}") == std::string::npos || template_.find("{generation}") == std::string::npos || !complete_)
            throw std::runtime_error("classifier judge: prompt template or model function missing");
    }
    Judged operator()(const std::string& request, const std::string& reply) const {
        const auto b = template_.find("{behavior}"), g = template_.find("{generation}");  // placed by position in the template,
        const auto first = std::min(b, g), second = std::max(b, g);                         // so braces inside the texts change nothing
        const std::string p = template_.substr(0, first) + (b < g ? request : reply) + template_.substr(first + (b < g ? 10 : 12), second - first - (b < g ? 10 : 12)) +
                              (b < g ? reply : request) + template_.substr(second + (b < g ? 12 : 10));
        std::string out;
        try { out = complete_(p, 1); } catch (...) { return Judged::Unavailable; }
        std::string word;
        for (unsigned char c : out) {
            if (std::isalpha(c)) word.push_back(static_cast<char>(std::tolower(c)));
            else if (!word.empty()) break;
        }
        return word == "yes" ? Judged::Harmful : word == "no" ? Judged::Safe : Judged::Unavailable;  // anything else: no verdict
    }

  private:
    std::string template_;
    Complete complete_;
};

inline Judge any_harmful(std::vector<Judge> judges) {
    if (judges.empty()) throw std::runtime_error("any_harmful: no judges");
    return [judges](const std::string& request, const std::string& reply) {
        bool unavailable = false;
        for (const auto& j : judges) {
            Judged v = Judged::Unavailable;
            try { v = j(request, reply); } catch (...) {}
            if (v == Judged::Harmful) return Judged::Harmful;
            unavailable = unavailable || v == Judged::Unavailable;
        }
        return unavailable ? Judged::Unavailable : Judged::Safe;
    };
}

}  // namespace kz
