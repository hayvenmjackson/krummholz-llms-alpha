package ai.krummholz.llm;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.Assert.*;

public class ModelVerifierTest {
    @Rule public TemporaryFolder folder = new TemporaryFolder();
    private static final byte[] BYTES = new byte[]{'a', 'b', 'c'};
    private static final String HASH = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    @Test public void exactBytesAndHashAcceptedBeforeAnyParsing() throws Exception {
        File stage = folder.newFile("model.partial");
        ModelVerifier.copyVerified(new ByteArrayInputStream(BYTES), stage, 3, HASH, () -> false, n -> {});
        assertArrayEquals(BYTES, Files.readAllBytes(stage.toPath()));
        ModelVerifier.verify(stage, 3, HASH, () -> false);
    }
    @Test public void sameSizeWrongHashRejected() throws Exception {
        File stage = folder.newFile("model.partial");
        assertThrows(IOException.class, () -> ModelVerifier.copyVerified(new ByteArrayInputStream(new byte[]{'x','y','z'}), stage, 3, HASH, () -> false, n -> {}));
    }
    @Test public void truncatedAndOversizedImportsRejected() throws Exception {
        File stage = folder.newFile("model.partial");
        assertThrows(IOException.class, () -> ModelVerifier.copyVerified(new ByteArrayInputStream(new byte[]{'a'}), stage, 3, HASH, () -> false, n -> {}));
        assertThrows(IOException.class, () -> ModelVerifier.copyVerified(new ByteArrayInputStream(new byte[]{'a','b','c','d'}), stage, 3, HASH, () -> false, n -> {}));
    }
    @Test public void cancellationStopsCopyWithoutTouchingPreviousModel() throws Exception {
        File previous = folder.newFile("approved-model.gguf"); Files.write(previous.toPath(), BYTES);
        File stage = folder.newFile("import.partial");
        assertThrows(CancellationException.class, () -> ModelVerifier.copyVerified(new ByteArrayInputStream(BYTES), stage, 3, HASH, () -> true, n -> {}));
        assertArrayEquals(BYTES, Files.readAllBytes(previous.toPath()));
    }
    @Test public void wrongManifestRejectedBeforeInputIsRead() throws Exception {
        File stage = folder.newFile("model.partial");
        ByteArrayInputStream input = new ByteArrayInputStream(BYTES);
        assertThrows(IOException.class, () -> ModelVerifier.copyVerified(input, stage, 0, "", () -> false, n -> {}));
        assertEquals(3, input.available());
    }
    @Test public void existingModelIsRehashedAndCancellationHonored() throws Exception {
        File model = folder.newFile("model.gguf"); Files.write(model.toPath(), new byte[]{'x','y','z'});
        assertThrows(IOException.class, () -> ModelVerifier.verify(model, 3, HASH, () -> false));
        assertThrows(CancellationException.class, () -> ModelVerifier.verify(model, 3, HASH, () -> true));
    }
    @Test public void boundedLargeManifestPassesValidationWithoutAllocatingItsContents() throws Exception {
        File stage = folder.newFile("large-model.partial");
        // Cancellation occurs after spec validation and before the first read.
        // This exercises long-sized manifests without creating a multi-GiB file.
        assertThrows(CancellationException.class, () -> ModelVerifier.copyVerified(
                new ByteArrayInputStream(BYTES), stage, 6158339904L, HASH, () -> true, n -> {}));
        assertThrows(CancellationException.class, () -> ModelVerifier.copyVerified(
                new ByteArrayInputStream(BYTES), stage, 8L * 1024 * 1024 * 1024, HASH, () -> true, n -> {}));
        ByteArrayInputStream untouched = new ByteArrayInputStream(BYTES);
        assertThrows(IOException.class, () -> ModelVerifier.copyVerified(
                untouched, stage, 8L * 1024 * 1024 * 1024 + 1, HASH, () -> false, n -> {}));
        assertEquals(3, untouched.available());
    }
    private static String hash(byte[] bytes) throws Exception {
        StringBuilder result = new StringBuilder();
        for (byte b : MessageDigest.getInstance("SHA-256").digest(bytes)) result.append(String.format("%02x", b & 255));
        return result.toString();
    }
    private static List<ModelVerifier.Part> parts(byte[] whole, int partBytes) throws Exception {
        List<ModelVerifier.Part> result = new ArrayList<>();
        int count = (whole.length + partBytes - 1) / partBytes;
        for (int i = 0; i < count; i++) {
            int end = Math.min(whole.length, (i + 1) * partBytes);
            result.add(new ModelVerifier.Part(i + 1, String.format("synthetic.gguf.part-%05d-of-%05d", i + 1, count),
                i * partBytes, end - i * partBytes, hash(Arrays.copyOfRange(whole, i * partBytes, end))));
        }
        return result;
    }
    @Test public void multipartSelectionMatchesCanonicalNamesInAnyOrder() throws Exception {
        List<ModelVerifier.Part> parts = parts("abcdefg".getBytes("UTF-8"), 2);
        boolean[] seen = new boolean[4];
        for (int index : new int[]{3, 1, 0, 2})
            assertEquals(index, ModelVerifier.matchSelectedPart(parts.get(index).name, parts.get(index).bytes, parts, seen));
        for (boolean value : seen) assertTrue(value);
        assertThrows(IOException.class, () -> ModelVerifier.matchSelectedPart(parts.get(0).name, 2, parts, seen));
        assertThrows(IOException.class, () -> ModelVerifier.matchSelectedPart("renamed-part", 2, parts, new boolean[4]));
        assertThrows(IOException.class, () -> ModelVerifier.matchSelectedPart(parts.get(0).name, 1, parts, new boolean[4]));
    }
    @Test public void multipartBytesAndBothHashLevelsVerifiedInOrder() throws Exception {
        byte[] whole = "abcdefg".getBytes("UTF-8");
        List<ModelVerifier.Part> parts = parts(whole, 2);
        File stage = folder.newFile("multipart.partial");
        List<Integer> opened = new ArrayList<>();
        List<Long> progress = new ArrayList<>();
        ModelVerifier.copyVerifiedParts(part -> {
            opened.add(part.index);
            return new ByteArrayInputStream(Arrays.copyOfRange(whole, (int) part.offset, (int) (part.offset + part.bytes)));
        }, stage, whole.length, hash(whole), "synthetic.gguf", 2, parts, () -> false, progress::add);
        assertEquals(Arrays.asList(1, 2, 3, 4), opened);
        assertEquals(Arrays.asList(2L, 4L, 6L, 7L), progress);
        assertArrayEquals(whole, Files.readAllBytes(stage.toPath()));
    }
    @Test public void multipartCorruptTruncatedAndOversizedPartAllFail() throws Exception {
        byte[] whole = "abcdefg".getBytes("UTF-8");
        List<ModelVerifier.Part> parts = parts(whole, 2);
        File stage = folder.newFile("multipart.partial");
        for (byte[] invalid : new byte[][]{new byte[]{'z','z'}, new byte[]{'a'}, new byte[]{'a','b','c'}})
            assertThrows(IOException.class, () -> ModelVerifier.copyVerifiedParts(part -> new ByteArrayInputStream(invalid),
                stage, whole.length, hash(whole), "synthetic.gguf", 2, parts, () -> false, n -> {}));
    }
    @Test public void validPartHashesDoNotBypassWrongWholeHash() throws Exception {
        byte[] whole = "abcdefg".getBytes("UTF-8");
        List<ModelVerifier.Part> parts = parts(whole, 2);
        File stage = folder.newFile("multipart.partial");
        assertThrows(IOException.class, () -> ModelVerifier.copyVerifiedParts(part ->
            new ByteArrayInputStream(Arrays.copyOfRange(whole, (int) part.offset, (int) (part.offset + part.bytes))),
            stage, whole.length, HASH, "synthetic.gguf", 2, parts, () -> false, n -> {}));
    }
    @Test public void missingOrWrongOrderPartsRejectedBeforeSourceOpen() throws Exception {
        byte[] whole = "abcdefg".getBytes("UTF-8");
        List<ModelVerifier.Part> parts = parts(whole, 2);
        File stage = folder.newFile("multipart.partial");
        AtomicInteger opened = new AtomicInteger();
        ModelVerifier.PartSource source = part -> { opened.incrementAndGet(); return new ByteArrayInputStream(whole); };
        List<ModelVerifier.Part> missing = new ArrayList<>(parts); missing.remove(0);
        assertThrows(IOException.class, () -> ModelVerifier.copyVerifiedParts(source, stage, whole.length, hash(whole),
            "synthetic.gguf", 2, missing, () -> false, n -> {}));
        List<ModelVerifier.Part> wrong = new ArrayList<>(parts); java.util.Collections.reverse(wrong);
        assertThrows(IOException.class, () -> ModelVerifier.copyVerifiedParts(source, stage, whole.length, hash(whole),
            "synthetic.gguf", 2, wrong, () -> false, n -> {}));
        assertEquals(0, opened.get());
    }
    @Test public void multipartCancellationClosesCurrentInputAndPreservesPreviousModel() throws Exception {
        byte[] whole = "abcdefg".getBytes("UTF-8");
        List<ModelVerifier.Part> parts = parts(whole, 2);
        File previous = folder.newFile("approved-model.gguf"); Files.write(previous.toPath(), BYTES);
        File stage = folder.newFile("multipart.partial");
        AtomicBoolean cancelled = new AtomicBoolean(), closed = new AtomicBoolean();
        AtomicInteger opened = new AtomicInteger();
        assertThrows(CancellationException.class, () -> ModelVerifier.copyVerifiedParts(part -> {
            opened.incrementAndGet();
            return new ByteArrayInputStream(new byte[]{'a','b'}) {
                @Override public void close() throws IOException { closed.set(true); super.close(); }
            };
        }, stage, whole.length, hash(whole), "synthetic.gguf", 2, parts, cancelled::get, n -> cancelled.set(true)));
        assertTrue(closed.get()); assertEquals(1, opened.get());
        assertArrayEquals(BYTES, Files.readAllBytes(previous.toPath()));
    }
    @Test public void multipartLargeOffsetsAreValidatedWithoutAllocatingModel() throws Exception {
        long whole = 6158339904L, partBytes = 1610612736L;
        List<ModelVerifier.Part> parts = new ArrayList<>();
        for (int i = 0; i < 4; i++) parts.add(new ModelVerifier.Part(i + 1,
            String.format("SmolLM3-3B-BF16.gguf.part-%05d-of-00004", i + 1), i * partBytes,
            Math.min(partBytes, whole - i * partBytes), HASH));
        ModelVerifier.validateParts("SmolLM3-3B-BF16.gguf", whole, partBytes, parts);
        assertEquals(4831838208L, parts.get(3).offset);
        assertEquals(1326501696L, parts.get(3).bytes);
        assertThrows(IOException.class, () -> ModelVerifier.validateParts("../bad.gguf", whole, partBytes, parts));
        assertThrows(IOException.class, () -> ModelVerifier.validateParts("SmolLM3-3B-BF16.gguf", whole, 2L * 1024 * 1024 * 1024, parts));
    }
}
