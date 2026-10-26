package ai.krummholz.llm;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicLong;

/** Local CPU inference. ModelStore must verify the private GGUF before loadModel. */
public final class NativeEngine implements AutoCloseable {
    static { System.loadLibrary("krummholz_llm"); }

    public interface ProgressListener {
        /** Called on the inference worker; contains no prompt or generated text. */
        void onProgress(int generatedTokens);
    }

    private final AtomicLong handle = new AtomicLong(nativeCreate());
    private final AtomicLong nextOperation = new AtomicLong();
    private final AtomicLong activeOperation = new AtomicLong();

    /** Blocking: invoke on the serial inference worker, never the main thread. */
    public void loadModel(String verifiedAbsolutePath, int contextTokens, int threads) {
        byte[] path = encodeUtf8(verifiedAbsolutePath);
        long operation = beginOperation();
        try { nativeLoadModel(requireHandle(), operation, path, contextTokens, threads); }
        finally { activeOperation.compareAndSet(operation, 0); }
    }

    /** Returns raw protocol output for validation, not text safe to display directly. */
    public String generate(String rawPrompt, int maxOutputTokens, ProgressListener listener) {
        byte[] prompt = encodeUtf8(rawPrompt);
        long operation = beginOperation();
        try {
            byte[] output = nativeGenerate(requireHandle(), operation, prompt, maxOutputTokens, listener);
            if (output == null) throw new IllegalStateException("Native generation returned no result.");
            return StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(output)).toString();
        } catch (CharacterCodingException error) {
            throw new IllegalStateException("The model returned incomplete or invalid UTF-8.");
        } finally { activeOperation.compareAndSet(operation, 0); }
    }

    /** Instrumentation seam: exact input IDs from the same tokenizer used by generate. */
    int[] tokenizeForTest(String rawPrompt) {
        byte[] prompt = encodeUtf8(rawPrompt);
        long operation = beginOperation();
        try {
            int[] ids = nativeTokenize(requireHandle(), operation, prompt);
            if (ids == null) throw new IllegalStateException("Native tokenization returned no result.");
            return ids;
        } finally { activeOperation.compareAndSet(operation, 0); }
    }

    /** Cancels only the active call, including its Java-to-JNI handoff; idle calls do nothing. */
    public void cancel() {
        long operation = activeOperation.get();
        long value = handle.get();
        if (value != 0 && operation != 0) nativeCancel(value, operation);
    }

    /** Invalidates the handle immediately; active native work owns its cleanup. */
    @Override public void close() {
        long value = handle.getAndSet(0);
        if (value != 0) nativeRelease(value);
    }

    private long requireHandle() {
        long value = handle.get();
        if (value == 0) throw new IllegalStateException("The native engine is closed.");
        return value;
    }

    private long beginOperation() {
        long operation = nextOperation.incrementAndGet();
        if (operation <= 0) throw new IllegalStateException("Native operation limit reached.");
        if (!activeOperation.compareAndSet(0, operation))
            throw new IllegalStateException("The native engine is already busy.");
        return operation;
    }

    private static byte[] encodeUtf8(String text) {
        if (text == null) throw new IllegalArgumentException("Text must not be null.");
        try {
            ByteBuffer encoded = StandardCharsets.UTF_8.newEncoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .encode(CharBuffer.wrap(text));
            byte[] bytes = new byte[encoded.remaining()];
            encoded.get(bytes);
            return bytes;
        } catch (CharacterCodingException error) {
            throw new IllegalArgumentException("Text contains invalid Unicode.");
        }
    }

    private static native long nativeCreate();
    private static native void nativeLoadModel(long handle, long operation, byte[] path, int contextTokens, int threads);
    private static native byte[] nativeGenerate(long handle, long operation, byte[] prompt, int maxOutputTokens,
                                               ProgressListener listener);
    private static native int[] nativeTokenize(long handle, long operation, byte[] prompt);
    private static native void nativeCancel(long handle, long operation);
    private static native void nativeRelease(long handle);
}
