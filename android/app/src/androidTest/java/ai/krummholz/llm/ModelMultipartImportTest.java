package ai.krummholz.llm;

import android.content.Context;
import android.content.ContextWrapper;
import android.net.Uri;
import android.os.Bundle;
import android.os.CancellationSignal;
import android.os.OperationCanceledException;
import android.os.SystemClock;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import static org.junit.Assert.*;

/** Real Binder/regular-file import using seven synthetic bytes; no model inference or production pin changes. */
@RunWith(AndroidJUnit4.class)
public final class ModelMultipartImportTest {
    private static final Uri ROOT = Uri.parse("content://ai.krummholz.llm.test.import-fixtures");
    private static final byte[] WHOLE = {'a','b','c','d','e','f','g'};
    private Context realContext, isolatedContext;
    private File ownedDirectory;
    private ModelStore store;

    @Before public void prepare() throws Exception {
        realContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        ownedDirectory = new File(realContext.getCacheDir(), "multipart-test-" + UUID.randomUUID());
        assertTrue(ownedDirectory.mkdir());
        isolatedContext = new ContextWrapper(realContext) {
            @Override public Context getApplicationContext() { return this; }
            @Override public File getNoBackupFilesDir() { return ownedDirectory; }
        };
        realContext.getContentResolver().call(ROOT, "reset", null, null);
        store = new ModelStore(isolatedContext, new ModelStore.Spec(syntheticSpec()));
    }

    @After public void cleanup() throws Exception {
        if (realContext != null) {
            realContext.getContentResolver().call(ROOT, "reset", null, null);
            realContext.getContentResolver().call(ROOT, "clear-model", null, null);
        }
        if (ownedDirectory != null) {
            File models = new File(ownedDirectory, "models");
            File[] files = models.listFiles();
            if (files != null) for (File file : files) {
                assertEquals(models.getCanonicalPath(), file.getCanonicalFile().getParent());
                assertTrue(file.isFile()); assertTrue(file.delete());
            }
            if (models.exists()) assertTrue(models.delete());
            assertTrue(ownedDirectory.delete());
        }
    }

    private static JSONObject syntheticSpec() throws Exception {
        JSONObject json = new JSONObject().put("approved", true).put("name", "Synthetic seven-byte fixture")
            .put("protocol", "smollm3-chat-v1").put("filename", "synthetic.gguf")
            .put("bytes", WHOLE.length).put("sha256", hash(WHOLE)).put("download_url", "")
            .put("license", "synthetic-test").put("upstream", "synthetic-test");
        JSONArray parts = new JSONArray();
        for (int i = 0; i < 4; i++) {
            int end = Math.min(WHOLE.length, i * 2 + 2);
            parts.put(new JSONObject().put("index", i + 1)
                .put("name", String.format(java.util.Locale.ROOT, "synthetic.gguf.part-%05d-of-00004", i + 1))
                .put("offset", i * 2).put("bytes", end - i * 2).put("sha256", hash(Arrays.copyOfRange(WHOLE, i * 2, end))));
        }
        return json.put("multipart", new JSONObject().put("schema_version", 1).put("format", "raw-concatenation-v1")
            .put("manifest_sha256", hash("synthetic manifest fixture".getBytes("UTF-8"))).put("part_bytes", 2).put("parts", parts));
    }
    private static String hash(byte[] bytes) throws Exception {
        StringBuilder value = new StringBuilder();
        for (byte b : MessageDigest.getInstance("SHA-256").digest(bytes)) value.append(String.format("%02x", b & 255));
        return value.toString();
    }
    private static Uri part(int number) { return Uri.parse(ROOT + "/part/" + number); }
    private static List<Uri> selection() { return Arrays.asList(part(4), part(2), part(1), part(3)); }
    private void mode(String mode) { realContext.getContentResolver().call(ROOT, "mode", mode, null); }
    private File importGood() throws Exception {
        File model = store.importModels(selection(), () -> false, n -> {}, new CancellationSignal());
        assertArrayEquals(WHOLE, Files.readAllBytes(model.toPath()));
        return model;
    }
    private void intact(File previous) throws Exception {
        assertEquals(previous, store.verifiedModel(() -> false));
        assertArrayEquals(WHOLE, Files.readAllBytes(previous.toPath()));
        File[] partials = previous.getParentFile().listFiles((directory, name) -> name.startsWith("import-") && name.endsWith(".partial"));
        assertNotNull(partials); assertEquals(0, partials.length);
    }

    @Test(timeout = 15_000) public void arbitraryPickerOrderImportsVerifiedBytes() throws Exception {
        assertTrue(store.spec.hasMultipart()); assertEquals(4, store.spec.parts.size());
        intact(importGood());
    }

    @Test(timeout = 20_000) public void badPartsAndMetadataNeverReplacePreviousModel() throws Exception {
        File previous = importGood();
        for (String mode : new String[]{"corrupt", "truncated", "wrong-metadata", "pipe-part"}) {
            mode(mode);
            assertThrows(IOException.class, () -> store.importModels(selection(), () -> false, n -> {}, new CancellationSignal()));
            intact(previous);
        }
        mode("normal"); intact(importGood());
    }

    @Test(timeout = 15_000) public void missingDuplicateAndUnknownSelectionsAreRejected() throws Exception {
        File previous = importGood();
        assertThrows(IOException.class, () -> store.importModels(selection().subList(0, 3), () -> false, n -> {}, new CancellationSignal()));
        assertThrows(IOException.class, () -> store.importModels(Arrays.asList(part(1), part(1), part(3), part(4)), () -> false, n -> {}, new CancellationSignal()));
        List<Uri> alias = new ArrayList<>(Arrays.asList(part(1), Uri.parse(ROOT + "/alias/1"), part(3), part(4)));
        assertThrows(IOException.class, () -> store.importModels(alias, () -> false, n -> {}, new CancellationSignal()));
        assertThrows(IOException.class, () -> store.importModels(Arrays.asList(part(1), Uri.parse(ROOT + "/unknown"), part(3), part(4)), () -> false, n -> {}, new CancellationSignal()));
        intact(previous);
    }

    @Test(timeout = 20_000) public void cancellingRealProviderQueryPreservesPreviousAndAllowsRetry() throws Exception {
        File previous = importGood(); mode("blocked-query");
        CancellationSignal signal = new CancellationSignal();
        AtomicBoolean cancelled = new AtomicBoolean();
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<File> pending = executor.submit(() -> store.importModels(selection(), cancelled::get, n -> {}, signal));
        try {
            long deadline = SystemClock.elapsedRealtime() + 5_000;
            boolean entered = false;
            while (SystemClock.elapsedRealtime() < deadline) {
                Bundle status = realContext.getContentResolver().call(ROOT, "status", null, null);
                if (status != null && status.getInt("blocked_queries") > 0) { entered = true; break; }
                if (pending.isDone()) break;
                SystemClock.sleep(25);
            }
            assertTrue("Real provider query did not start.", entered);
            cancelled.set(true); signal.cancel();
            try { pending.get(5, TimeUnit.SECONDS); fail("Cancelled metadata query activated a model."); }
            catch (ExecutionException error) {
                assertTrue(error.getCause() instanceof CancellationException || error.getCause() instanceof OperationCanceledException);
            }
        } finally {
            cancelled.set(true); signal.cancel(); pending.cancel(true); executor.shutdownNow(); mode("normal");
        }
        intact(previous); intact(importGood());
    }

    @Test(timeout = 15_000) public void cancellingDuringPartCopyNeverActivatesPartialBytes() throws Exception {
        File previous = importGood();
        AtomicBoolean cancelled = new AtomicBoolean();
        CancellationSignal signal = new CancellationSignal();
        assertThrows(CancellationException.class, () -> store.importModels(selection(), cancelled::get,
            n -> { cancelled.set(true); signal.cancel(); }, signal));
        intact(previous);
    }

    @Test(timeout = 15_000) public void exactWholeFileStillUsesContentProviderAndApprovalGateRemains() throws Exception {
        File source = new File(ownedDirectory, "fixture-whole.gguf");
        Files.write(source.toPath(), WHOLE);
        try {
            Uri whole = ImportFixtureProvider.registerModel(realContext, source);
            File result = store.importModel(whole, () -> false, n -> {}, new CancellationSignal());
            intact(result);
            ModelStore held = new ModelStore(isolatedContext, new ModelStore.Spec(syntheticSpec().put("approved", false)));
            assertThrows(IOException.class, () -> held.importModels(selection(), () -> false, n -> {}, new CancellationSignal()));
            intact(result);
        } finally { assertTrue(source.delete()); }
    }

    @Test public void stringAndFractionalByteCountsDoNotCoerceIntoPins() throws Exception {
        assertThrows(IOException.class, () -> new ModelStore.Spec(syntheticSpec().put("bytes", "7")));
        assertThrows(IOException.class, () -> new ModelStore.Spec(syntheticSpec().put("bytes", 7.0)));
        JSONObject invalid = syntheticSpec(); invalid.getJSONObject("multipart").getJSONArray("parts").getJSONObject(0).put("bytes", "2");
        assertThrows(IOException.class, () -> new ModelStore.Spec(invalid));
    }
}
