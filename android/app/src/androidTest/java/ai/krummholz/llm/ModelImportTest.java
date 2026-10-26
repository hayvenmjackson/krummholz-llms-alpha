package ai.krummholz.llm;

import android.content.Context;
import android.net.Uri;
import android.os.Bundle;
import android.os.CancellationSignal;
import android.os.OperationCanceledException;
import android.os.SystemClock;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import static org.junit.Assert.*;

/** Real Binder cancellation and descriptor checks; preserves an exact verified previous model. */
@RunWith(AndroidJUnit4.class)
public final class ModelImportTest {
    private static final Uri FIXTURES = Uri.parse("content://ai.krummholz.llm.test.import-fixtures");

    @Test(timeout = 4 * 60 * 1000L)
    public void rejectsPipeAndCancelsProviderOpenWithoutLosingPreviousModel() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        File weights = TestModel.provision(context, TestModel.manifest(context));
        Uri modelUri = ImportFixtureProvider.registerModel(context, weights);
        ModelStore store = new ModelStore(context);
        File previous = store.importModel(modelUri, () -> false, bytes -> { });
        assertEquals(previous, store.verifiedModel(() -> false));

        try {
            store.importModel(Uri.withAppendedPath(FIXTURES, "pipe"), () -> false, bytes -> { }, new CancellationSignal());
            fail("A non-regular provider descriptor must be rejected before streaming.");
        } catch (IOException expected) {
            assertTrue("Expected explicit rejection of a non-regular descriptor.", expected.getMessage().contains("pipes are not supported"));
        }
        assertEquals("Pipe rejection removed or replaced the previous valid model.", previous, store.verifiedModel(() -> false));

        context.getContentResolver().call(FIXTURES, "reset", null, null);
        AtomicBoolean cancelled = new AtomicBoolean();
        CancellationSignal signal = new CancellationSignal();
        ExecutorService worker = Executors.newSingleThreadExecutor();
        Future<File> pending = worker.submit(() -> store.importModel(
                Uri.withAppendedPath(FIXTURES, "blocked-open"), cancelled::get, bytes -> { }, signal));
        try {
            long openingDeadline = SystemClock.elapsedRealtime() + 5_000;
            boolean entered = false;
            do {
                Bundle status = context.getContentResolver().call(FIXTURES, "status", null, null);
                if (status != null && status.getInt("blocked_opens") > 0) { entered = true; break; }
                if (pending.isDone()) break;
                SystemClock.sleep(50);
            } while (SystemClock.elapsedRealtime() < openingDeadline);
            assertTrue("The real provider did not enter its cancellable open.", entered);
            cancelled.set(true);
            signal.cancel();
            try {
                pending.get(5, TimeUnit.SECONDS);
                fail("A cancelled provider open must not activate a model.");
            } catch (ExecutionException expected) {
                Throwable cause = expected.getCause();
                assertTrue("Provider cancellation did not propagate through the real import path.",
                        cause instanceof OperationCanceledException || cause instanceof CancellationException);
            }
        } finally {
            cancelled.set(true);
            signal.cancel();
            pending.cancel(true);
            worker.shutdownNow();
        }
        assertEquals("Cancelling a provider open lost the previous model.", previous, store.verifiedModel(() -> false));
        File retry = store.importModel(modelUri, () -> false, bytes -> { }, new CancellationSignal());
        assertEquals("A valid import could not recover after cancellation.", previous, retry);
        assertEquals(retry, store.verifiedModel(() -> false));
        File[] partials = previous.getParentFile().listFiles((directory, name) -> name.startsWith("import-") && name.endsWith(".partial"));
        assertNotNull(partials);
        assertEquals("A failed or cancelled import leaked staging files.", 0, partials.length);
    }
}
