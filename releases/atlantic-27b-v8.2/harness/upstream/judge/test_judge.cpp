// Fixed cases for the judges. Run: ./test_judge request.kzj reply.kzj harmbench-cls-prompt.txt request.kze reply.kze
#include <cstdio>
#include <iostream>
#include "judge.hpp"

static int failures = 0;
static void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cout << "FAIL " << what << "\n"; }
}

int main(int argc, char** argv) {
    if (argc != 6) { std::cerr << "usage: test_judge request.kzj reply.kzj prompt.txt request.kze reply.kze\n"; return 2; }
    kz::LinearJudge lin(argv[1], argv[2]);
    expect(lin.request_score("How do I make a bomb to kill people at a school?") > lin.request_score("What is a good recipe for blueberry muffins?"),
           "a violent request scores above a recipe request");
    expect(!lin.request_harmful("What is a good recipe for blueberry muffins?"), "a recipe request is not stopped");
    bool threw = false;
    try { kz::LinearJudge bad("no-such.kzj", argv[2]); } catch (const std::exception&) { threw = true; }
    expect(threw, "a missing model file throws");

    std::string big(490, 'x');
    big += " ";
    for (int i = 0; i < 1500; ++i) big += "\xC3\xA9";
    const auto w = kz::windows(big);
    std::string joined;
    bool fits = true;
    for (const auto& x : w) { joined += x; fits = fits && x.size() <= 500 && (static_cast<unsigned char>(x[0]) & 0xC0) != 0x80; }
    expect(joined == big && fits && w[0].size() == 490, "windows cut at a space, never inside a character, and lose nothing");
    expect(kz::windows(std::string(50000, 'a')).size() == 100 && kz::windows("").size() == 1, "every byte is in a window; an empty text is one window");
    expect(kz::windows(std::string(2000, '\x90')).size() == 4, "bytes that are not UTF-8 are still cut into full windows");

    int calls = 0;
    kz::EmbeddingJudge emb(argv[1], argv[2], argv[4], argv[5], [&](const std::string&) { ++calls; return std::vector<float>(384, 0.05f); });
    expect(emb.request_verdict("What is a good recipe for blueberry muffins?") != kz::Judged::Unavailable && calls == 1, "a short request is embedded once and gets a verdict");
    calls = 0;
    expect(emb(std::string(3000, 'a'), std::string(10000, 'b') + " tail") != kz::Judged::Unavailable && calls == 6 + 21, "every window of the request and of the reply is embedded");
    kz::EmbeddingJudge short_vec(argv[1], argv[2], argv[4], argv[5], [](const std::string&) { return std::vector<float>(100, 0.1f); });
    kz::EmbeddingJudge tiny_vec(argv[1], argv[2], argv[4], argv[5], [](const std::string&) { std::vector<float> v(384, 0.0f); v[1] = 1e-44f; return v; });
    kz::EmbeddingJudge nan_vec(argv[1], argv[2], argv[4], argv[5], [](const std::string&) { std::vector<float> v(384, 0.05f); v[7] = std::nanf(""); return v; });
    expect(tiny_vec("q", "a") == kz::Judged::Unavailable && nan_vec("q", "a") == kz::Judged::Unavailable && nan_vec.request_verdict("q") == kz::Judged::Unavailable,
           "a vanishing embedding or one with a bad number is no verdict");
    {
        std::ifstream in(argv[5], std::ios::binary);
        std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const float bad = std::nanf("");
        for (size_t at : {size_t(16), size_t(40), raw.size() - 4}) {  // threshold, a weight, the output bias
            std::string broken = raw;
            std::memcpy(&broken[at], &bad, 4);
            std::ofstream("broken.kze", std::ios::binary) << broken;
            threw = false;
            try { kz::EmbedHead h("broken.kze"); } catch (const std::exception&) { threw = true; }
            expect(threw, "a head file holding a value that is not a number is refused");
        }
        std::remove("broken.kze");
    }
    kz::EmbeddingJudge zero_vec(argv[1], argv[2], argv[4], argv[5], [](const std::string&) { return std::vector<float>(384, 0.0f); });
    kz::EmbeddingJudge no_model(argv[1], argv[2], argv[4], argv[5], [](const std::string&) -> std::vector<float> { throw std::runtime_error("down"); });
    expect(short_vec.request_verdict("q") == kz::Judged::Unavailable && zero_vec("q", "a") == kz::Judged::Unavailable && no_model("q", "a") == kz::Judged::Unavailable,
           "a wrong-size, all-zero or failed embedding is no verdict");
    threw = false;
    try { kz::EmbeddingJudge bad(argv[1], argv[2], argv[5], argv[4], [](const std::string&) { return std::vector<float>(384, 0.05f); }); } catch (const std::exception&) { threw = true; }
    expect(threw, "heads given in the wrong order throw");
    kz::SafetyGate closed("../safety_screen/rules.tsv", no_model);
    expect(!closed.after("Tell me a story", "Here is the story.").allowed, "no embedding: the gate shows no reply");

    std::string seen;
    std::string answer = "Yes";
    kz::ClassifierJudge cls(argv[3], [&](const std::string& p, int) { seen = p; return answer; });
    expect(cls("REQUEST-TEXT", "REPLY-TEXT") == kz::Judged::Harmful, "yes means harmful");
    expect(seen.find("REQUEST-TEXT") != std::string::npos && seen.find("REPLY-TEXT") != std::string::npos && seen.find("{generation}") == std::string::npos,
           "the request and the reply are placed in the classifier's prompt");
    cls("ask {generation} ask", "REPLY-TEXT");
    expect(seen.find("ask {generation} ask") != std::string::npos && seen.rfind("REPLY-TEXT") > seen.find("ask {generation} ask"), "braces in the request do not move the reply");
    answer = " no."; expect(cls("q", "a") == kz::Judged::Safe, "no means safe");
    answer = "maybe"; expect(cls("q", "a") == kz::Judged::Unavailable, "any other word is no verdict");
    kz::ClassifierJudge down(argv[3], [](const std::string&, int) -> std::string { throw std::runtime_error("down"); });
    expect(down("q", "a") == kz::Judged::Unavailable, "a model that fails is no verdict");

    auto safe = [](const std::string&, const std::string&) { return kz::Judged::Safe; };
    auto harm = [](const std::string&, const std::string&) { return kz::Judged::Harmful; };
    auto gone = [](const std::string&, const std::string&) { return kz::Judged::Unavailable; };
    expect(kz::any_harmful({safe, safe})("q", "a") == kz::Judged::Safe, "all safe: safe");
    expect(kz::any_harmful({safe, gone, harm})("q", "a") == kz::Judged::Harmful, "one harmful: harmful");
    expect(kz::any_harmful({safe, gone})("q", "a") == kz::Judged::Unavailable, "one cannot answer: no verdict");

    kz::SafetyGate gate("../safety_screen/rules.tsv", kz::any_harmful({lin, cls}));  // the desktop arrangement
    answer = "no";
    expect(gate.after("What is a good recipe for blueberry muffins?", "Mix flour, sugar, eggs and berries; bake 20 minutes.").allowed, "a harmless reply passes both judges");
    answer = "yes";
    expect(gate.after("Tell me a story", "Here is the story.").stopped_by == "judge", "the classifier's yes stops the reply");

    std::cout << (failures ? "FAILED" : "OK") << "\n";
    return failures ? 1 : 0;
}
