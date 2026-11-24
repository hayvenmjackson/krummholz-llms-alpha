#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "upstream/judge/judge.hpp"

std::string hex(const std::string& text) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(text.size() * 2);
    for (unsigned char byte : text) {
        result += digits[byte >> 4];
        result += digits[byte & 15];
    }
    return result;
}

int digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    throw std::runtime_error("invalid protocol byte");
}

std::string read_text() {
    std::string line;
    if (!std::getline(std::cin, line) || line.size() > 65536 || line.size() % 2 != 0)
        throw std::runtime_error("invalid protocol line");
    std::string result;
    result.reserve(line.size() / 2);
    for (size_t i = 0; i < line.size(); i += 2)
        result += static_cast<char>((digit(line[i]) << 4) | digit(line[i + 1]));
    return result;
}

std::vector<float> embed(const std::string& text) {
    std::cout << "EMBED\t" << hex(text) << std::endl;
    std::string line;
    if (!std::getline(std::cin, line) || line.size() > 32768) return {};
    std::istringstream stream(line);
    std::vector<float> result;
    for (float number; stream >> number;) result.push_back(number);
    if (!stream.eof() || result.size() != 384) return {};
    return result;
}

double parse_threshold(const std::string& value) {
    size_t end = 0;
    const double threshold = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(threshold)) throw std::runtime_error("invalid threshold");
    return threshold;
}

void print_probe(const std::string& stopped_by, const std::optional<double>& score) {
    if (!stopped_by.empty()) {
        std::cout << "SCORE\t" << stopped_by << "\t\n";
    } else if (!score || !std::isfinite(*score)) {
        std::cout << "SCORE\tjudge-unavailable\t\n";
    } else {
        std::cout << "SCORE\tscored\t" << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << *score << '\n';
    }
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) return 2;
    const std::string stage = argv[2];
    const bool before = stage == "before" || stage == "request-probe";
    const bool probe = stage == "request-probe" || stage == "reply-probe";
    if (!before && stage != "after" && stage != "reply-probe") return 2;
    if (probe && argc == 4) return 2;
    try {
        const std::filesystem::path assets = argv[1];
        const auto models = assets / "judge" / "models";
        const auto model = [&](const char* file) { return (models / file).string(); };
        std::optional<double> threshold;
        double released_request_threshold = 0;
        if (argc == 4) {
            threshold = parse_threshold(argv[3]);
            const double released = kz::EmbedHead(model(before ? "request.kze" : "reply.kze")).threshold();
            if (before ? *threshold <= released : *threshold >= released)
                throw std::runtime_error("threshold would violate its stage constraints");
            if (before) released_request_threshold = released;
        }
        const kz::EmbeddingJudge judge(model("request.kzj"), model("reply.kzj"),
                                       model("request.kze"), model("reply.kze"), embed);
        const kz::GateTexts texts;
        std::optional<double> reply_score;
        const auto reply_judge = [&](const std::string& request, const std::string& reply) {
            if (!threshold && !probe) return judge(request, reply);
            reply_score = judge.reply_score(request, reply);
            if (!reply_score || !std::isfinite(*reply_score)) return kz::Judged::Unavailable;
            if (threshold && *reply_score >= *threshold) return kz::Judged::Harmful;
            return kz::Judged::Safe;
        };
        const kz::SafetyGate gate((assets / "safety_screen" / "rules.tsv").string(),
                                  reply_judge, texts);
        const auto request = read_text();
        kz::GateResult result;
        bool deferred = false;
        if (before) {
            result = gate.before(request);
            if (probe) {
                print_probe(result.stopped_by, result.allowed ? judge.request_score(request) : std::nullopt);
                return 0;
            }
            if (result.allowed) {
                auto verdict = kz::Judged::Unavailable;
                if (threshold) {
                    const auto score = judge.request_score(request);
                    if (score && std::isfinite(*score)) {
                        verdict = *score >= *threshold ? kz::Judged::Harmful : kz::Judged::Safe;
                        deferred = verdict == kz::Judged::Safe && *score >= released_request_threshold;
                    }
                } else {
                    verdict = judge.request_verdict(request);
                }
                if (verdict == kz::Judged::Harmful)
                    result = {false, "request-judge", "", texts.refusal};
                if (verdict == kz::Judged::Unavailable)
                    result = {false, "judge-unavailable", "", texts.unavailable};
            }
        } else {
            result = gate.after(request, read_text());
            if (probe) {
                print_probe(result.stopped_by, reply_score);
                return 0;
            }
        }
        std::cout << "RESULT\t" << result.allowed << '\t' << result.stopped_by << '\t'
                  << result.rule << '\t' << hex(result.text);
        if (deferred) std::cout << "\tdeferred";
        std::cout << std::endl;
        return 0;
    } catch (const std::exception&) {
        return 3;
    }
}
