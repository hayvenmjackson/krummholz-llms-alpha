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

    /** Blocking: invoke on the serial inference worker, never the main thread. */
    public void loadModel(String verifiedAbsolutePath, int contextTokens, int threads) {
        nativeLoadModel(requireHandle(), encodeUtf8(verifiedAbsolutePath), contextTokens, threads);
    }

    /** Returns raw protocol output for validation, not text safe to display directly. */
    public String generate(String rawPrompt, int maxOutputTokens, ProgressListener listener) {
        byte[] output = nativeGenerate(requireHandle(), encodeUtf8(rawPrompt), maxOutputTokens, listener);
        if (output == null) throw new IllegalStateException("Native generation returned no result.");
        try {
            return StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(output)).toString();
        } catch (CharacterCodingException error) {
            throw new IllegalStateException("The model returned incomplete or invalid UTF-8.");
        }
    }

    /** Thread-safe; cancellation is observed during loading and CPU decoding. */
    public void cancel() {
        long value = handle.get();
        if (value != 0) nativeCancel(value);
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
    private static native void nativeLoadModel(long handle, byte[] path, int contextTokens, int threads);
    private static native byte[] nativeGenerate(long handle, byte[] prompt, int maxOutputTokens,
                                               ProgressListener listener);
    private static native void nativeCancel(long handle);
    private static native void nativeRelease(long handle);
}
