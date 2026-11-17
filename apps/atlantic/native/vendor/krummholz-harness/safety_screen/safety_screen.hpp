// Krummholz harness, layer 1 of the safety screen: fixed rules checked in code before a request reaches the model
// (stage "input") and before a reply is shown (stage "output"). No model, no network; a few microseconds per rule.
// Rules live in a tab-separated file so a daily job can replace them; a file that fails to load must be treated by the
// caller as "screen unavailable" and shown to the user, never as "allow".
//
// Rule line:  id <TAB> stage <TAB> category <TAB> action <TAB> window <TAB> expression
//   stage       input | output
//   action      refuse | crisis | review      (crisis: refuse and show crisis resources; review: pass to layer 2)
//   window      0 = anywhere in the text; N = every group must match within N words of the first group
//   expression  groups joined by " & "; a group is alternatives joined by "|"; a group starting with "!" must NOT match.
//               An alternative is a phrase of words; a word ending in "*" matches any word with that beginning.
//               Output rules may also use "raw:TEXT" (substring of the lower-cased reply) and "builtin:shellcode".
//               A raw or builtin match that is already present in the user's own message is ignored, so a reply that
//               explains a line the user pasted is not blocked.
// Text is lower-cased and split into words of letters and digits; everything else separates words.
#pragma once
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kz {

struct Verdict {
    std::string action = "allow", rule, category;
    bool blocked() const { return action == "refuse" || action == "crisis"; }
};

inline std::string lower(const std::string& s) {
    std::string o(s);
    for (auto& c : o) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return o;
}

inline std::vector<std::string> words(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (unsigned char c : text) {
        if (std::isalnum(c)) cur.push_back(static_cast<char>(std::tolower(c)));
        else if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

inline std::vector<std::string> split(const std::string& s, const std::string& sep) {
    std::vector<std::string> out;
    size_t a = 0, b;
    while ((b = s.find(sep, a)) != std::string::npos) { out.push_back(s.substr(a, b - a)); a = b + sep.size(); }
    out.push_back(s.substr(a));
    return out;
}

// A run of at least 16 escaped bytes (\x41\x42...), the usual way injected machine code is written out.
inline bool has_shellcode(const std::string& t) {
    int run = 0;
    for (size_t i = 0; i + 3 < t.size();) {
        if (t[i] == '\\' && (t[i + 1] == 'x' || t[i + 1] == 'X') && std::isxdigit(static_cast<unsigned char>(t[i + 2])) &&
            std::isxdigit(static_cast<unsigned char>(t[i + 3]))) {
            if (++run >= 16) return true;
            i += 4;
        } else {
            if (!(t[i] == '"' || t[i] == '\'' || t[i] == ' ' || t[i] == '\n' || t[i] == '+' || t[i] == ',')) run = 0;
            ++i;
        }
    }
    return false;
}

struct Alt {
    std::vector<std::string> phrase;  // words; a trailing '*' on a word means "begins with"
    std::string raw;                  // output rules: substring of the lower-cased text
    bool shellcode = false;
};

struct Group {
    bool negative = false;
    std::vector<Alt> alts;
};

struct Rule {
    std::string id, stage, category, action;
    int window = 0;
    std::vector<Group> groups;
};

class Screen {
  public:
    explicit Screen(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("safety screen: cannot read rules file " + path);
        std::string line;
        int n = 0;
        while (std::getline(in, line)) {
            ++n;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            auto f = split(line, "\t");
            if (f.size() != 6) throw std::runtime_error("safety screen: line " + std::to_string(n) + " needs 6 fields");
            Rule r{f[0], f[1], f[2], f[3], std::stoi(f[4]), {}};
            if (r.stage != "input" && r.stage != "output") throw std::runtime_error("safety screen: bad stage, line " + std::to_string(n));
            if (r.action != "refuse" && r.action != "crisis" && r.action != "review") throw std::runtime_error("safety screen: bad action, line " + std::to_string(n));
            for (auto g : split(f[5], " & ")) {
                Group grp;
                if (!g.empty() && g[0] == '!') { grp.negative = true; g = g.substr(1); }
                for (const auto& a : split(g, "|")) {
                    Alt alt;
                    if (a.rfind("raw:", 0) == 0) alt.raw = lower(a.substr(4));
                    else if (a == "builtin:shellcode") alt.shellcode = true;
                    else {
                        std::istringstream ss(lower(a));
                        std::string w;
                        while (ss >> w) alt.phrase.push_back(w);
                    }
                    if (alt.phrase.empty() && alt.raw.empty() && !alt.shellcode) throw std::runtime_error("safety screen: empty alternative, line " + std::to_string(n));
                    grp.alts.push_back(alt);
                }
                r.groups.push_back(grp);
            }
            if (r.groups.empty() || r.groups[0].negative) throw std::runtime_error("safety screen: first group must be positive, line " + std::to_string(n));
            rules_.push_back(r);
        }
        if (rules_.empty()) throw std::runtime_error("safety screen: no rules in " + path);
    }

    size_t size() const { return rules_.size(); }

    // The user's message, before it reaches the model.
    Verdict check_input(const std::string& text) const { return check("input", text, ""); }
    // The model's reply, before it is shown. `input` is the user's message the reply answers.
    Verdict check_output(const std::string& reply, const std::string& input) const { return check("output", reply, input); }

  private:
    std::vector<Rule> rules_;

    static bool word_matches(const std::string& pattern, const std::string& w) {
        if (!pattern.empty() && pattern.back() == '*') return w.compare(0, pattern.size() - 1, pattern, 0, pattern.size() - 1) == 0;
        return pattern == w;
    }

    // Word positions where the group matches; raw and builtin matches count as position 0.
    static std::vector<int> positions(const Group& g, const std::vector<std::string>& w, const std::string& raw, const std::string& raw_input) {
        std::vector<int> out;
        for (const auto& a : g.alts) {
            if (a.shellcode) { if (has_shellcode(raw) && !has_shellcode(raw_input)) out.push_back(0); continue; }
            if (!a.raw.empty()) { if (raw.find(a.raw) != std::string::npos && raw_input.find(a.raw) == std::string::npos) out.push_back(0); continue; }
            const size_t k = a.phrase.size();
            for (size_t i = 0; i + k <= w.size(); ++i) {
                bool ok = true;
                for (size_t j = 0; j < k && ok; ++j) ok = word_matches(a.phrase[j], w[i + j]);
                if (ok) out.push_back(static_cast<int>(i));
            }
        }
        return out;
    }

    static bool near(const std::vector<int>& ps, int anchor, int window) {
        return std::any_of(ps.begin(), ps.end(), [&](int p) { return std::abs(p - anchor) <= window; });
    }

    Verdict check(const std::string& stage, const std::string& text, const std::string& input) const {
        const auto w = words(text);
        const std::string raw = lower(text), raw_input = lower(input);
        Verdict review;
        for (const auto& r : rules_) {
            if (r.stage != stage) continue;
            std::vector<std::vector<int>> pos;
            bool possible = true;
            for (const auto& g : r.groups) {
                pos.push_back(positions(g, w, raw, raw_input));
                if (!g.negative && pos.back().empty()) { possible = false; break; }
            }
            if (!possible) continue;
            bool hit = false;
            if (r.window == 0) {
                hit = true;
                for (size_t i = 0; i < r.groups.size(); ++i) if (r.groups[i].negative && !pos[i].empty()) hit = false;
            } else {
                for (int anchor : pos[0]) {
                    bool ok = true;
                    for (size_t i = 1; i < r.groups.size() && ok; ++i) {
                        const bool n = near(pos[i], anchor, r.window);
                        ok = r.groups[i].negative ? !n : n;
                    }
                    if (ok) { hit = true; break; }
                }
            }
            if (!hit) continue;
            if (r.action != "review") return Verdict{r.action, r.id, r.category};  // the first blocking rule decides
            if (review.rule.empty()) review = Verdict{r.action, r.id, r.category};
        }
        return review;
    }
};

}  // namespace kz
