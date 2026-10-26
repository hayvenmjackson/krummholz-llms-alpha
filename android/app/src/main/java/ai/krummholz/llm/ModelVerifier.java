package ai.krummholz.llm;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;

/** Verifies opaque bytes before any GGUF metadata or native model parser runs. */
public final class ModelVerifier {
    private ModelVerifier() {}
    public interface Progress { void onBytes(long bytes); }

    public static void copyVerified(InputStream input, File stage, long expectedBytes, String expectedHash,
                                    BooleanSupplier cancelled, Progress progress) throws IOException {
        checkSpec(expectedBytes, expectedHash);
        MessageDigest digest = digest();
        long total = 0;
        try (FileOutputStream output = new FileOutputStream(stage)) {
            byte[] buffer = new byte[128 * 1024];
            for (;;) {
                checkCancelled(cancelled);
                int n = input.read(buffer);
                if (n < 0) break;
                if (n == 0) continue;
                if (n > expectedBytes - total) throw new IOException("This is not the approved model file (size mismatch).");
                output.write(buffer, 0, n); digest.update(buffer, 0, n); total += n;
                progress.onBytes(total);
            }
            checkCancelled(cancelled);
            if (total != expectedBytes || !hex(digest.digest()).equals(expectedHash))
                throw new IOException("This model did not pass its size and SHA-256 checks. The previous model is unchanged.");
            output.getFD().sync();
        }
    }

    public static void verify(File file, long expectedBytes, String expectedHash,
                              BooleanSupplier cancelled) throws IOException {
        checkSpec(expectedBytes, expectedHash);
        if (!file.isFile() || file.length() != expectedBytes)
            throw new IOException("Import the approved model file to continue.");
        MessageDigest digest = digest();
        try (InputStream input = new FileInputStream(file)) {
            byte[] buffer = new byte[128 * 1024];
            long total = 0;
            for (;;) {
                checkCancelled(cancelled);
                int n = input.read(buffer);
                if (n < 0) break;
                total += n;
                if (total > expectedBytes) throw new IOException("The saved model changed. Import it again.");
                digest.update(buffer, 0, n);
            }
            checkCancelled(cancelled);
            if (total != expectedBytes || !hex(digest.digest()).equals(expectedHash))
                throw new IOException("The saved model failed verification. Import the approved file again.");
        }
    }
    private static void checkSpec(long bytes, String hash) throws IOException {
        if (bytes <= 0 || bytes > 4L * 1024 * 1024 * 1024 || hash == null || !hash.matches("[0-9a-f]{64}"))
            throw new IOException("This build does not have an approved model manifest.");
    }
    private static void checkCancelled(BooleanSupplier cancelled) {
        if (cancelled.getAsBoolean() || Thread.currentThread().isInterrupted()) throw new CancellationException();
    }
    private static MessageDigest digest() {
        try { return MessageDigest.getInstance("SHA-256"); }
        catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
    }
    private static String hex(byte[] bytes) {
        StringBuilder text = new StringBuilder();
        for (byte value : bytes) text.append(String.format(java.util.Locale.ROOT, "%02x", value & 255));
        return text.toString();
    }
}
