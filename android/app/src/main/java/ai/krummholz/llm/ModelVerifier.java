package ai.krummholz.llm;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;

/** Verifies opaque bytes before any GGUF metadata or native model parser runs. */
public final class ModelVerifier {
    private ModelVerifier() {}
    public interface Progress { void onBytes(long bytes); }
    public static final class Part {
        public final int index;
        public final String name, sha256;
        public final long offset, bytes;
        public Part(int index, String name, long offset, long bytes, String sha256) {
            this.index = index; this.name = name; this.offset = offset; this.bytes = bytes; this.sha256 = sha256;
        }
    }
    public interface PartSource { InputStream open(Part part) throws IOException; }

    public static void validateParts(String filename, long wholeBytes, long partBytes, List<Part> parts) throws IOException {
        if (filename == null || !filename.matches("[A-Za-z0-9][A-Za-z0-9_.-]{0,120}\\.gguf")
                || wholeBytes <= 0 || wholeBytes > 8L * 1024 * 1024 * 1024
                || partBytes <= 0 || partBytes >= 2L * 1024 * 1024 * 1024 || parts == null
                || parts.isEmpty() || parts.size() > 64 || parts.size() != (wholeBytes + partBytes - 1) / partBytes)
            throw new IOException("The embedded multipart manifest is invalid.");
        for (int i = 0; i < parts.size(); i++) {
            Part part = parts.get(i);
            long offset = i * partBytes;
            String expectedName = String.format(java.util.Locale.ROOT, "%s.part-%05d-of-%05d", filename, i + 1, parts.size());
            if (part == null || part.index != i + 1 || !expectedName.equals(part.name) || part.offset != offset
                    || part.bytes != Math.min(partBytes, wholeBytes - offset)
                    || part.sha256 == null || !part.sha256.matches("[0-9a-f]{64}"))
                throw new IOException("The embedded multipart order or file pin is invalid.");
        }
    }

    /** Match metadata only; content hashes and descriptor sizes are checked separately. */
    public static int matchSelectedPart(String name, long bytes, List<Part> parts, boolean[] seen) throws IOException {
        if (name == null || name.length() > 180 || seen == null || seen.length != parts.size())
            throw new IOException("Choose the original locally downloaded model parts.");
        for (int i = 0; i < parts.size(); i++) {
            Part part = parts.get(i);
            if (part.name.equals(name)) {
                if (seen[i]) throw new IOException("A model part was selected more than once.");
                if (bytes != part.bytes) throw new IOException("A model part is incomplete (size mismatch).");
                seen[i] = true;
                return i;
            }
        }
        throw new IOException("An unrecognized file was selected. Choose only the original model parts.");
    }

    public static void copyVerifiedParts(PartSource source, File stage, long expectedBytes, String expectedHash,
                                         String filename, long partBytes, List<Part> parts,
                                         BooleanSupplier cancelled, Progress progress) throws IOException {
        checkSpec(expectedBytes, expectedHash);
        validateParts(filename, expectedBytes, partBytes, parts);
        checkCancelled(cancelled);
        MessageDigest whole = digest();
        long total = 0;
        try (FileOutputStream output = new FileOutputStream(stage)) {
            byte[] buffer = new byte[128 * 1024];
            for (Part part : parts) {
                checkCancelled(cancelled);
                MessageDigest chunk = digest();
                long copied = 0;
                try (InputStream input = source.open(part)) {
                    if (input == null) throw new IOException("A model part could not be opened.");
                    for (;;) {
                        checkCancelled(cancelled);
                        int n = input.read(buffer);
                        if (n < 0) break;
                        if (n == 0) continue;
                        if (n > part.bytes - copied) throw new IOException("A model part has an unexpected size.");
                        output.write(buffer, 0, n); chunk.update(buffer, 0, n); whole.update(buffer, 0, n);
                        copied += n; total += n; progress.onBytes(total);
                    }
                }
                checkCancelled(cancelled);
                if (copied != part.bytes || !hex(chunk.digest()).equals(part.sha256))
                    throw new IOException("A model part failed its size and SHA-256 checks. The previous model is unchanged.");
            }
            checkCancelled(cancelled);
            if (total != expectedBytes || !hex(whole.digest()).equals(expectedHash))
                throw new IOException("The assembled model failed its whole-file SHA-256 check. The previous model is unchanged.");
            output.getFD().sync();
        }
    }

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
        if (bytes <= 0 || bytes > 8L * 1024 * 1024 * 1024 || hash == null || !hash.matches("[0-9a-f]{64}"))
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
