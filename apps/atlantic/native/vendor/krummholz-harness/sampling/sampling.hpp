// Krummholz harness: the sampling settings the harness passes to the language model when the caller gives none.
// The language model is one component; the harness owns how it is sampled.
//
// Default temperature 0.7 (Hayven, November 16, 2026). Evidence so far, Atlantic Lite 4B v8.1 Q4_K_M, 229 fact
// questions without tools: wrong answers 22 at temperature 0, 13 and 19 in two samples at 0.7; right 82 against 90
// and 83. That is within noise for 229 questions, so the default is provisional until the temperature probe in
// results/bof reports. The release measurements on the model cards were taken at temperature 0.
//
// Use:  kz::Sampling s = kz::Sampling::resolve(requested_temperature);   // pass a negative value for "not given"
#pragma once

namespace kz {

struct Sampling {
    static constexpr double kDefaultTemperature = 0.7;
    static constexpr double kMaxTemperature = 2.0;

    double temperature = kDefaultTemperature;

    // The caller's temperature if it gave a usable one (0 is allowed and means greedy), else the default.
    static Sampling resolve(double requested) {
        Sampling s;
        if (requested >= 0.0 && requested == requested) s.temperature = requested > kMaxTemperature ? kMaxTemperature : requested;
        return s;
    }
};

}  // namespace kz
