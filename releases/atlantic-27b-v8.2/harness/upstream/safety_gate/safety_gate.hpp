// Krummholz harness: the safety gate every model sits behind (Atlantic Lite and Atlantic 27B use this same code).
// It joins the two checks that need no change to the weights:
//   layer 1  fixed rules (../safety_screen), on the request before the model runs and on the reply before it is shown
//   layer 2  a judge: a classifier that reads the request and the finished reply and says whether the reply is harmful
// The model's reply is shown only if both layers pass it. Anything else gives a fixed refusal written here, in code.
// Fail closed: no rules file, or a judge that cannot answer, means no reply is shown.
//
// The judge is a function the harness supplies, so this header needs no model library. On a server or desktop it can
// run the HarmBench classifier (cais/HarmBench-Mistral-7b-val-cls, MIT) through llama.cpp with the prompt in
// config/harmbench-cls-prompt.json and return Harmful on "yes". A phone needs a smaller classifier.
//
//   kz::SafetyGate gate("rules.tsv", judge);
//   auto a = gate.before(request);            if (!a.allowed) show(a.text); else {
//   auto reply = model(request);              // reasoning already removed
//   auto b = gate.after(request, reply);      show(b.text); }   // b.text is the reply or a fixed refusal
#pragma once
#include <functional>
#include <string>
#include <utility>
#include "../safety_screen/safety_screen.hpp"

namespace kz {

enum class Judged { Safe, Harmful, Unavailable };
using Judge = std::function<Judged(const std::string& request, const std::string& reply)>;

struct GateResult {
    bool allowed = false;
    std::string stopped_by;  // "", "input-rule", "output-rule", "judge", "judge-unavailable"
    std::string rule;        // the rule id when a rule stopped it
    std::string text;        // what the user sees
};

struct GateTexts {
    std::string refusal = "I can't help with that.";
    std::string crisis = "I can't help with that. If you are thinking about harming yourself, please reach out now: in the "
                         "United States call or text 988 (Suicide and Crisis Lifeline); elsewhere, contact local emergency services.";
    std::string unavailable = "The safety check is unavailable, so I can't show a reply right now.";
};

class SafetyGate {
  public:
    SafetyGate(const std::string& rules_path, Judge judge, GateTexts texts = {})
        : screen_(rules_path), judge_(std::move(judge)), texts_(std::move(texts)) {
        if (!judge_) throw std::runtime_error("safety gate: no judge supplied");  // never run with layer 2 missing
    }

    // Before the model runs. A stopped request never reaches the model.
    GateResult before(const std::string& request) const { return from_rule(screen_.check_input(request), "input-rule", ""); }

    // After the model has finished. `reply` is the text meant for the user, without reasoning.
    GateResult after(const std::string& request, const std::string& reply) const {
        GateResult r = from_rule(screen_.check_output(reply, request), "output-rule", reply);
        if (!r.allowed) return r;
        Judged j = Judged::Unavailable;
        try { j = judge_(request, reply); } catch (...) { j = Judged::Unavailable; }
        if (j == Judged::Safe) return r;
        r.allowed = false;
        r.stopped_by = j == Judged::Harmful ? "judge" : "judge-unavailable";
        r.text = j == Judged::Harmful ? texts_.refusal : texts_.unavailable;
        return r;
    }

  private:
    GateResult from_rule(const Verdict& v, const char* stage, const std::string& pass_text) const {
        GateResult r;
        if (!v.blocked()) { r.allowed = true; r.text = pass_text; return r; }
        r.stopped_by = stage; r.rule = v.rule;
        r.text = v.action == "crisis" ? texts_.crisis : texts_.refusal;
        return r;
    }
    Screen screen_;
    Judge judge_;
    GateTexts texts_;
};

}  // namespace kz
