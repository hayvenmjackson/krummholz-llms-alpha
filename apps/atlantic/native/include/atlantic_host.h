/* Copyright 2026 Krummholz AI LLC. Licensed under Apache-2.0. */
#ifndef ATLANTIC_HOST_H
#define ATLANTIC_HOST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct atlantic_host atlantic_host;
typedef struct atlantic_cancellation atlantic_cancellation;
typedef struct atlantic_result atlantic_result;

/* 1 means continue, 0 means stop. One call per tokenizer token. These pieces
 * stay inside the host and must never be sent to a display or diagnostic log. */
typedef int (*atlantic_emit_fn)(void* sink_context, const char* piece, size_t bytes);
typedef int (*atlantic_cancelled_fn)(void* sink_context);

/* Return 0 only at a normal end-of-generation, 1 after an intentional stop,
 * and 2 on error. Do not retain callback pointers after this call returns. */
typedef int (*atlantic_generate_fn)(
    void* user_data, const char* checked_input, size_t input_bytes,
    double temperature, int max_tokens,
    atlantic_emit_fn emit, atlantic_cancelled_fn cancelled, void* sink_context);

/* Fill exactly output_dimensions floats with bge-small-en-v1.5 first-token
 * pooling, returning 0 on success and nonzero on failure. No local placeholder
 * embedding is acceptable in a release build. Do not retain the pointers. */
typedef int (*atlantic_embed_fn)(
    void* user_data, const char* window, size_t window_bytes,
    float* output, size_t output_dimensions,
    atlantic_cancelled_fn cancelled, void* cancel_context);

/* 0 success, 1 invalid arguments, 2 initialization/assets unavailable.
 * Call only with the installed bundle after signature/hash verification.
 * Both callbacks are required. user_data must outlive the host. */
int atlantic_host_create(const char* verified_bundle_directory,
                        atlantic_generate_fn generate, atlantic_embed_fn embed,
                        void* user_data, atlantic_host** output);
void atlantic_host_destroy(atlantic_host* host);

atlantic_cancellation* atlantic_cancellation_create(void);
void atlantic_cancellation_cancel(atlantic_cancellation* cancellation);
void atlantic_cancellation_destroy(atlantic_cancellation* cancellation);

/* Synchronous worker-thread operation. host and cancellation must remain alive
 * until it returns. nullptr indicates invalid arguments/allocation failure.
 * A negative temperature selects the pinned harness default, currently 0.7. */
atlantic_result* atlantic_host_run(atlantic_host* host, const char* complete_input,
                                 size_t input_bytes, atlantic_cancellation* cancellation,
                                 double requested_temperature);
void atlantic_result_destroy(atlantic_result* result);

/* Returned pointers live until result_destroy(). Only result_text() may be
 * displayed. There is no raw generation or streaming-text accessor. */
const char* atlantic_result_text(const atlantic_result* result);
const char* atlantic_result_status(const atlantic_result* result);
const char* atlantic_result_stage(const atlantic_result* result);
const char* atlantic_result_rule(const atlantic_result* result);
int atlantic_result_released_model_text(const atlantic_result* result);
int atlantic_result_generated_tokens(const atlantic_result* result);

#ifdef __cplusplus
}
#endif
#endif
