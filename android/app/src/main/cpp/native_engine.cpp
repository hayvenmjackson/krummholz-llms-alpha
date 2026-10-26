#include <jni.h>
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {
constexpr int kBatchSize = 16;
constexpr size_t kMaxPromptBytes = 65536;
constexpr size_t kMaxOutputBytes = 262144;
constexpr int64_t kMaxModelBytes = 2LL * 1024 * 1024 * 1024;
constexpr int64_t kOperationMillis = 180000;

int64_t nowMillis() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Cancelled final : std::runtime_error {
    Cancelled() : std::runtime_error("The native operation was cancelled.") {}
};
struct JavaException final {};

using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
using Batch = std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)>;
using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

struct Engine {
    std::mutex operation;
    std::atomic<jlong> activeOperation{0};
    std::atomic<jlong> cancelledThrough{0};
    std::atomic<bool> closed{false};
    std::atomic<int64_t> deadline{0};
    Model model{nullptr, llama_model_free};
    int contextTokens = 2048;
    int threads = 2;

    bool isCancelled() const {
        const jlong active = activeOperation.load();
        return active != 0 && cancelledThrough.load() >= active;
    }
    bool shouldStop() const {
        return isCancelled() || closed.load() || nowMillis() >= deadline.load();
    }
    void check() const {
        if (closed.load() || isCancelled()) throw Cancelled();
        if (nowMillis() >= deadline.load())
            throw std::runtime_error("Native operation exceeded its three-minute limit.");
    }
};

// A shared owner survives release/cancel racing with a worker. Java never owns a
// raw native pointer, and freed handles cannot be recycled into another engine.
std::mutex registryMutex;
std::unordered_map<jlong, std::shared_ptr<Engine>> registry;
jlong nextHandle = 1;
std::once_flag backendOnce;

std::shared_ptr<Engine> findEngine(jlong handle) {
    std::lock_guard<std::mutex> guard(registryMutex);
    auto found = registry.find(handle);
    if (found == registry.end()) throw std::runtime_error("The native engine is closed.");
    return found->second;
}

class Operation {
public:
    Operation(Engine & engine, jlong ticket) : engine_(engine), lock_(engine.operation, std::try_to_lock) {
        if (!lock_.owns_lock()) throw std::runtime_error("The native engine is already busy.");
        if (ticket <= 0) throw std::invalid_argument("Invalid native operation ticket.");
        // Java reserves this ticket before JNI entry. A cancel during that handoff
        // is retained for this call, while a delayed old cancel cannot poison a retry.
        if (engine_.cancelledThrough.load() >= ticket || engine_.closed.load()) throw Cancelled();
        engine_.activeOperation.store(ticket);
        engine_.deadline.store(nowMillis() + kOperationMillis);
    }
    ~Operation() { engine_.activeOperation.store(0); }
private:
    Engine & engine_;
    std::unique_lock<std::mutex> lock_;
};

void throwJava(JNIEnv * env, const char * type, const char * message) {
    if (env->ExceptionCheck()) return;
    jclass cls = env->FindClass(type);
    if (cls != nullptr) {
        env->ThrowNew(cls, message); // These messages are fixed ASCII, never user/model text.
        env->DeleteLocalRef(cls);
    }
}

void translateException(JNIEnv * env) {
    try { throw; }
    catch (const JavaException &) { /* Preserve the Java callback/allocation exception. */ }
    catch (const Cancelled & error) {
        throwJava(env, "java/util/concurrent/CancellationException", error.what());
    }
    catch (const std::invalid_argument & error) {
        throwJava(env, "java/lang/IllegalArgumentException", error.what());
    }
    catch (const std::bad_alloc &) {
        throwJava(env, "java/lang/OutOfMemoryError", "Insufficient memory for local inference.");
    }
    catch (const std::exception & error) {
        throwJava(env, "java/lang/IllegalStateException", error.what());
    }
    catch (...) {
        throwJava(env, "java/lang/IllegalStateException", "Native inference failed.");
    }
}

std::string readBytes(JNIEnv * env, jbyteArray bytes, size_t limit) {
    if (bytes == nullptr) throw std::invalid_argument("Input bytes must not be null.");
    const jsize length = env->GetArrayLength(bytes);
    if (length < 1 || static_cast<size_t>(length) > limit)
        throw std::invalid_argument("Input exceeds the native byte limit or is empty.");
    std::string value(static_cast<size_t>(length), '\0');
    env->GetByteArrayRegion(bytes, 0, length, reinterpret_cast<jbyte *>(value.data()));
    if (env->ExceptionCheck()) throw JavaException();
    if (value.find('\0') != std::string::npos)
        throw std::invalid_argument("Embedded null bytes are not supported.");
    return value;
}

bool abortDecode(void * data) { return static_cast<Engine *>(data)->shouldStop(); }
bool continueLoading(float, void * data) { return !abortDecode(data); }
void discardRuntimeLog(ggml_log_level, const char *, void *) {
    // Runtime logs can include model metadata; no prompt, source, path or token logging.
}

void fillBatch(llama_batch_ext * batch, const llama_token * tokens, int count, int position) {
    llama_batch_ext_clear(batch);
    for (int i = 0; i < count; ++i) {
        const int index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = position + i;
        if (index < 0 || !llama_batch_ext_set_pos(batch, index, &pos) ||
            !llama_batch_ext_set_output_logits(batch, index, i == count - 1))
            throw std::runtime_error("Unable to prepare the bounded inference batch.");
    }
}

void processBatch(Engine & engine, llama_context * context, llama_batch_ext * batch) {
    engine.check();
    const int result = llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch);
    engine.check();
    if (result != 0) throw std::runtime_error("Model decoding failed; no response was accepted.");
}

std::string tokenPiece(const llama_vocab * vocab, llama_token token) {
    char small[256];
    int size = llama_token_to_piece(vocab, token, small, sizeof(small), 0, true);
    if (size >= 0) return std::string(small, static_cast<size_t>(size));
    if (size < -65536) throw std::runtime_error("Model token exceeds the output limit.");
    std::string piece(static_cast<size_t>(-size), '\0');
    const int written = llama_token_to_piece(vocab, token, piece.data(),
                                            static_cast<int>(piece.size()), 0, true);
    if (written < 0 || static_cast<size_t>(written) > piece.size())
        throw std::runtime_error("Unable to decode a model token.");
    piece.resize(static_cast<size_t>(written));
    return piece;
}
} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_ai_krummholz_llm_NativeEngine_nativeCreate(JNIEnv * env, jclass) {
    try {
        std::call_once(backendOnce, [] {
            llama_log_set(discardRuntimeLog, nullptr);
            llama_backend_init();
        });
        auto engine = std::make_shared<Engine>();
        std::lock_guard<std::mutex> guard(registryMutex);
        if (nextHandle <= 0 || nextHandle == INT64_MAX)
            throw std::runtime_error("Native handle limit reached.");
        const jlong handle = nextHandle++;
        registry.emplace(handle, std::move(engine));
        return handle;
    } catch (...) { translateException(env); return 0; }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_krummholz_llm_NativeEngine_nativeLoadModel(
        JNIEnv * env, jclass, jlong handle, jlong ticket, jbyteArray pathBytes, jint contextTokens, jint threads) {
    try {
        if (contextTokens < 512 || contextTokens > 4096 || threads < 1 || threads > 4)
            throw std::invalid_argument("Context must be 512-4096 tokens and threads 1-4.");
        const std::string path = readBytes(env, pathBytes, 4096);
        if (path.front() != '/') throw std::invalid_argument("A private absolute model path is required.");
        auto engine = findEngine(handle);
        Operation operation(*engine, ticket);
        // Import/hash verification belongs to ModelStore. Open the verified local
        // file without following a symlink, then keep the same descriptor for load.
        const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (descriptor < 0) throw std::runtime_error("Unable to open the verified local model.");
        FILE * rawFile = fdopen(descriptor, "rb");
        if (rawFile == nullptr) {
            close(descriptor);
            throw std::runtime_error("Unable to open the local model stream.");
        }
        std::unique_ptr<FILE, decltype(&fclose)> file(rawFile, fclose);
        struct stat info{};
        char magic[4];
        if (fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 4 ||
            info.st_size > kMaxModelBytes || fread(magic, 1, sizeof(magic), file.get()) != sizeof(magic) ||
            std::memcmp(magic, "GGUF", sizeof(magic)) != 0)
            throw std::invalid_argument("The local model is not a supported bounded GGUF file.");
        std::rewind(file.get());
        engine->model.reset(); // Never keep two resident models during replacement.
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = 0;
        params.load_mode = LLAMA_LOAD_MODE_MMAP;
        params.use_extra_bufts = false;
        params.progress_callback = continueLoading;
        params.progress_callback_user_data = engine.get();
        Model loaded(llama_model_load_from_file_ptr(file.get(), params), llama_model_free);
        engine->check();
        if (!loaded) throw std::runtime_error("The verified model could not be loaded on this device.");
        if (llama_model_has_encoder(loaded.get()) || !llama_model_has_decoder(loaded.get()))
            throw std::invalid_argument("This alpha supports decoder-only text models.");
        if (llama_model_n_ctx_train(loaded.get()) < contextTokens)
            throw std::invalid_argument("Requested context exceeds the model's trained context.");
        engine->contextTokens = contextTokens;
        engine->threads = threads;
        engine->model = std::move(loaded);
    } catch (...) { translateException(env); }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_krummholz_llm_NativeEngine_nativeGenerate(
        JNIEnv * env, jclass, jlong handle, jlong ticket, jbyteArray promptBytes, jint maxTokens, jobject listener) {
    try {
        if (maxTokens < 1 || maxTokens > 1024)
            throw std::invalid_argument("Output must be bounded to 1-1024 tokens.");
        const std::string prompt = readBytes(env, promptBytes, kMaxPromptBytes);
        auto engine = findEngine(handle);
        Operation operation(*engine, ticket);
        if (!engine->model) throw std::runtime_error("Load a verified model before generation.");
        const llama_vocab * vocab = llama_model_get_vocab(engine->model.get());
        const int probe = llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()),
                                         nullptr, 0, true, true);
        if (probe >= 0 || probe == INT32_MIN)
            throw std::invalid_argument("The prompt could not be tokenized within its bounds.");
        const int needed = -probe;
        if (needed > engine->contextTokens - maxTokens - 16)
            throw std::invalid_argument("Prompt plus output exceeds the actual model token budget.");
        std::vector<llama_token> tokens(static_cast<size_t>(needed));
        if (llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), tokens.data(),
                           needed, true, true) != needed)
            throw std::runtime_error("The prompt could not be tokenized consistently.");
        jmethodID progress = nullptr;
        if (listener != nullptr) {
            jclass cls = env->GetObjectClass(listener);
            if (cls == nullptr) throw JavaException();
            progress = env->GetMethodID(cls, "onProgress", "(I)V");
            env->DeleteLocalRef(cls);
            if (progress == nullptr || env->ExceptionCheck()) throw JavaException();
        }
        llama_context_params params = llama_context_default_params();
        params.n_ctx = static_cast<uint32_t>(engine->contextTokens);
        params.n_batch = kBatchSize;
        params.n_ubatch = kBatchSize;
        params.n_seq_max = 1;
        params.n_threads = engine->threads;
        params.n_threads_batch = engine->threads;
        params.offload_kqv = false;
        params.op_offload = false;
        params.no_perf = true;
        params.abort_callback = abortDecode;
        params.abort_callback_data = engine.get();
        // Fresh context per request: no previous prompt, KV state or profile carries over.
        Context context(llama_init_from_model(engine->model.get(), params), llama_free);
        engine->check();
        if (!context) throw std::runtime_error("Insufficient memory for the requested context.");
        if (llama_n_ctx(context.get()) < static_cast<uint32_t>(needed + maxTokens))
            throw std::runtime_error("The runtime allocated less context than requested.");
        Batch batch(llama_batch_ext_init(context.get()), llama_batch_ext_free);
        Sampler sampler(llama_sampler_init_greedy(), llama_sampler_free);
        if (!batch || !sampler) throw std::bad_alloc();
        for (int position = 0; position < needed; position += kBatchSize) {
            const int count = std::min(kBatchSize, needed - position);
            fillBatch(batch.get(), tokens.data() + position, count, position);
            processBatch(*engine, context.get(), batch.get());
        }
        std::string output;
        for (int generated = 0; generated < maxTokens; ++generated) {
            engine->check();
            const llama_token token = llama_sampler_sample(sampler.get(), context.get(), -1);
            if (llama_vocab_is_eog(vocab, token)) break;
            output += tokenPiece(vocab, token); // true special-token rendering preserves Pleias markers.
            if (output.size() > kMaxOutputBytes)
                throw std::runtime_error("Model output exceeded its byte limit.");
            if (progress != nullptr) {
                env->CallVoidMethod(listener, progress, static_cast<jint>(generated + 1));
                if (env->ExceptionCheck()) throw JavaException();
            }
            engine->check();
            if (output.find("<|answer_end|>") != std::string::npos || generated + 1 == maxTokens) break;
            fillBatch(batch.get(), &token, 1, needed + generated);
            processBatch(*engine, context.get(), batch.get());
        }
        engine->check();
        jbyteArray result = env->NewByteArray(static_cast<jsize>(output.size()));
        if (result == nullptr) throw JavaException();
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(output.size()),
                               reinterpret_cast<const jbyte *>(output.data()));
        if (env->ExceptionCheck()) throw JavaException();
        return result;
    } catch (...) { translateException(env); return nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_krummholz_llm_NativeEngine_nativeCancel(JNIEnv *, jclass, jlong handle, jlong ticket) {
    if (ticket <= 0) return;
    std::lock_guard<std::mutex> guard(registryMutex);
    auto found = registry.find(handle);
    if (found != registry.end()) {
        auto & cancelled = found->second->cancelledThrough;
        jlong previous = cancelled.load();
        while (previous < ticket && !cancelled.compare_exchange_weak(previous, ticket)) {}
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_krummholz_llm_NativeEngine_nativeRelease(JNIEnv *, jclass, jlong handle) {
    std::shared_ptr<Engine> released;
    {
        std::lock_guard<std::mutex> guard(registryMutex);
        auto found = registry.find(handle);
        if (found == registry.end()) return;
        released = std::move(found->second);
        registry.erase(found);
    }
    released->closed.store(true);
    // If a worker is active, its shared_ptr frees the model after it exits.
}
