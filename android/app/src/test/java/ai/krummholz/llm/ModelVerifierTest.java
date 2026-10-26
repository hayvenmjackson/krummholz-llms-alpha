package ai.krummholz.llm;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.concurrent.CancellationException;
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
}
