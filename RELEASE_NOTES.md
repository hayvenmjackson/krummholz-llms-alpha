# Krummholz LLMs [ALPHA] — development notes

## October 26, 2026 — building, testing, and raising the bar

The code is public, GitHub pushes are working, and the Android app builds. That's
real progress. It isn't the same as having a model we should put in your hands.

We caught a frustrating problem in the first model candidates: sometimes the
answer contains the right fact, but the citation points to something else. In
other cases, the model adds details that were never in the source. We fixed the
tokenizer and prompt-handling issues we found, then checked the original weights
too. Those models still haven't earned a release.

So we're testing a larger model. Useful, accurate results matter more than a tiny
download or support for every old device. We're also preparing documented
public-domain text and evaluations for using tools and looking things up instead
of confidently guessing. That training has not happened yet, and the current app
does not browse the web.

The next test gave us a clear answer: SmolLM3 3B isn't good enough for this release.
It passed 12 of 36 cases in independent agent review. The other 24 failed, with
problems ranging from invalid output to reversed times, missing conditions and
following a malicious instruction in a source. We'd already fixed the integration
issues and verified that the native conversion preserved every tensor value.
Those fixes mattered, but they didn't solve the model's answer-quality problems.

We're keeping the failed results public and preparing an unquantized GPU
comparison of Olmo 3 7B Instruct and Qwen3 8B. Neither has earned a release yet.
The Android work continues too, including importing large model downloads in
verified parts. I want something useful in people's hands, and this is the work
it takes to get there.

Our first public-domain intake is also complete: six documents downloaded, with
30 carefully reviewed passages retained from five. It's a small starting point,
with separate training and evaluation text. More data only helps if it's good
data, and none of it has been used for training yet.

Acadia, Katahdin and Atlantic are being built as distinct workflows on a shared
platform. We aren't presenting three newly trained models as a finished product.
When the first usable alpha is ready, this repo will have the APK, checksums,
model details and an honest account of what we've tested. Quality comes first.

— Hayven
