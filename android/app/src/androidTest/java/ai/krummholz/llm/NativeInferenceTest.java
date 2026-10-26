package ai.krummholz.llm;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.util.Collections;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/** Real GGUF/JNI execution. Missing provisioning is a failure, never a skipped test. */
@RunWith(AndroidJUnit4.class)
public final class NativeInferenceTest {
    private static final int CONTEXT = 4096;
    private static final int OUTPUT = 1024;
    private static final int THREADS = 2;
    private static final int CASE_TIMEOUT_SECONDS = 180;
    private static final String LOG_TAG = "KrummholzInferenceTest";

    @Test(timeout = 20 * 60 * 1000L)
    public void approvedModelCompletesProfilesUnicodeAndRecoversAfterCancellation() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        JSONObject manifest = TestModel.manifest(context);
        long expectedBytes = manifest.getLong("bytes");
        String expectedHash = manifest.getString("sha256");
        File model = TestModel.provision(context, manifest);

        Protocol.Request acadia = request("acadia", "How long is the return window?",
                "Fictional demonstration policy: unused items may be returned within 30 days. "
                + "Refunds above 100 demo credits require a human reviewer.");
        Protocol.Request atlantic = request("atlantic", "What is the project codename?",
                "Fictional local note: the demonstration project's codename is Paper Lantern. "
                + "No messages, purchases or file changes have been authorized.");
        Protocol.Request katahdin = request("katahdin",
                "What survival percentage did the validated calculator report?",
                "Fictional training plot: a validated calculator reports 90 surviving seedlings "
                + "out of 100 initially planted, equal to 90% survival. "
                + "This is invented demonstration data, not a field observation.");
        Protocol.Request unicode = request("atlantic", "What exact marker is named in the note?",
                "Fictional local note: the exact marker is Café 🌲.");

        try (NativeEngine engine = new NativeEngine()) {
            long loadStarted = SystemClock.elapsedRealtime();
            engine.loadModel(model.getAbsolutePath(), CONTEXT, THREADS);
            logEvent(new JSONObject().put("event", "verified_model_loaded")
                    .put("sha256", expectedHash).put("bytes", expectedBytes)
                    .put("context", CONTEXT).put("threads", THREADS)
                    .put("elapsed_ms", SystemClock.elapsedRealtime() - loadStarted));

            Protocol.Answer first = complete(engine, acadia, "acadia-first", "30 days");
            Protocol.Answer second = complete(engine, atlantic, "atlantic", "Paper Lantern");
            assertFalse("A later profile retained a previous fixture's fact.",
                    evidence(second).contains("30 days"));
            Protocol.Answer third = complete(engine, katahdin, "katahdin", "90%");
            assertFalse("A later profile retained a previous fixture's fact.",
                    evidence(third).contains("paper lantern"));
            Protocol.Answer repeated = complete(engine, acadia, "acadia-repeated", "30 days");
            assertEquals("Greedy generation changed after unrelated profiles; inspect request/KV isolation.",
                    snapshot(first), snapshot(repeated));

            // Java-only verification can be cancelled while JNI is idle. Such a cancellation
            // belongs to that earlier job and must not poison this new, unrelated request.
            engine.cancel();
            Protocol.Answer unicodeAnswer = complete(engine, unicode, "unicode-after-idle-cancel", "Café 🌲");
            assertFalse("Native output contained replacement characters.",
                    evidence(unicodeAnswer).contains("\uFFFD"));

            AtomicBoolean cancellationRequested = new AtomicBoolean();
            boolean cancelled = false;
            try {
                generate(engine, Protocol.prompt(acadia, CONTEXT, OUTPUT), "cancelled", count -> {
                    if (count > 0 && cancellationRequested.compareAndSet(false, true)) engine.cancel();
                });
            } catch (CancellationException expected) {
                cancelled = true;
            }
            assertTrue("Cancellation must be requested during real token generation.",
                    cancellationRequested.get());
            assertTrue("A cancelled request must not return a displayable partial completion.", cancelled);
            Protocol.Answer afterCancel = complete(engine, acadia, "acadia-after-cancel", "30 days");
            assertEquals("Cancellation left state in the next request.", snapshot(first), snapshot(afterCancel));
        }
    }

    private static Protocol.Request request(String profile, String question, String source) {
        return Protocol.request(profile, question, Collections.singletonList(
                new Protocol.Source(profile + "/instrumentation-fixture", source)));
    }

    private static Protocol.Answer complete(NativeEngine engine, Protocol.Request request,
                                           String fixture, String requiredFact) throws Exception {
        String raw = generate(engine, Protocol.prompt(request, CONTEXT, OUTPUT), fixture, count -> { });
        Protocol.Answer answer;
        try {
            answer = Protocol.parseAnswer(raw, request);
        } catch (Protocol.ValidationException failure) {
            throw new AssertionError("No complete, valid answer for actual-model fixture " + fixture
                    + ". Raw generation is deliberately omitted from logs.", failure);
        }
        assertEquals("The parser must preserve the human-review status.", "needs_human_review", answer.status);
        assertFalse("This supported fixture requires at least one literal source citation.", answer.citations.isEmpty());
        for (Protocol.Citation citation : answer.citations) {
            assertEquals("A citation crossed profile boundaries.", request.sources.get(0).id, citation.sourceId);
            assertTrue("A citation did not match the supplied fixture.",
                    Protocol.quoteMatches(citation.quote, request.sources.get(0).text));
        }
        assertTrue("Actual inference did not recover the supplied fact for " + fixture + ".",
                evidence(answer).contains(requiredFact.toLowerCase(Locale.ROOT)));
        assertFalse("Internal protocol tokens reached the displayed answer.", answer.text.contains("<|"));
        logEvent(new JSONObject().put("event", "validated_answer").put("fixture", fixture)
                .put("citations", answer.citations.size()).put("status", answer.status));
        return answer;
    }

    private static String generate(NativeEngine engine, String prompt, String fixture,
                                   NativeEngine.ProgressListener progress) throws Exception {
        AtomicBoolean timedOut = new AtomicBoolean();
        AtomicInteger generated = new AtomicInteger();
        ScheduledExecutorService watchdog = Executors.newSingleThreadScheduledExecutor();
        long started = SystemClock.elapsedRealtime();
        ScheduledFuture<?> deadline = watchdog.schedule(() -> {
            timedOut.set(true);
            engine.cancel();
        }, CASE_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        try {
            String result = engine.generate(prompt, OUTPUT, count -> {
                generated.set(count);
                progress.onProgress(count);
            });
            assertFalse("Actual inference exceeded the bounded per-case deadline.", timedOut.get());
            assertTrue("The native engine did not report actual token generation.", generated.get() > 0);
            return result;
        } catch (CancellationException cancelled) {
            if (timedOut.get()) throw new AssertionError("Actual inference timed out for " + fixture + ".");
            throw cancelled;
        } finally {
            deadline.cancel(false);
            watchdog.shutdownNow();
            logEvent(new JSONObject().put("event", "generation_finished").put("fixture", fixture)
                    .put("generated_tokens", generated.get())
                    .put("elapsed_ms", SystemClock.elapsedRealtime() - started)
                    .put("timed_out", timedOut.get()));
        }
    }

    private static String evidence(Protocol.Answer answer) {
        StringBuilder result = new StringBuilder(answer.text);
        for (Protocol.Citation citation : answer.citations) result.append('\n').append(citation.quote);
        return result.toString().toLowerCase(Locale.ROOT);
    }

    private static String snapshot(Protocol.Answer answer) {
        StringBuilder result = new StringBuilder(answer.text);
        for (Protocol.Citation citation : answer.citations)
            result.append('\n').append(citation.sourceId).append(':').append(citation.quote);
        return result.toString();
    }

    private static void logEvent(JSONObject event) {
        // Fixtures, counts and public model identity only; no prompt, raw output or reasoning.
        Log.i(LOG_TAG, event.toString());
    }
}
