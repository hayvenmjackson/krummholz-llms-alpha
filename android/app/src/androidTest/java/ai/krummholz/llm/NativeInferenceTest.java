package ai.krummholz.llm;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.json.JSONObject;
import org.json.JSONArray;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
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
    private static final LocalDate REFERENCE_DATE = LocalDate.of(2026, 10, 26);
    private String protocol;

    @Test(timeout = 20 * 60 * 1000L)
    public void approvedModelCompletesProfilesUnicodeAndRecoversAfterCancellation() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        JSONObject manifest = TestModel.manifest(context);
        protocol = manifest.getString("protocol");
        assertTrue("Instrumentation requires an explicit supported protocol.",
                "pleias-rag-v1".equals(protocol) || SmolLm3Protocol.ID.equals(protocol));
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

            verifyPublisherTokenParity(engine, acadia, katahdin, atlantic);

            CheckedAnswer first = complete(engine, acadia, "acadia-first", "30 days");
            CheckedAnswer second = complete(engine, atlantic, "atlantic", "Paper Lantern");
            assertFalse("A later profile retained a previous fixture's fact.",
                    evidence(second).contains("30 days"));
            CheckedAnswer third = complete(engine, katahdin, "katahdin", "90%");
            assertFalse("A later profile retained a previous fixture's fact.",
                    evidence(third).contains("paper lantern"));
            CheckedAnswer repeated = complete(engine, acadia, "acadia-repeated", "30 days");
            assertEquals("Greedy generation changed after unrelated profiles; inspect request/KV isolation.",
                    snapshot(first), snapshot(repeated));

            // Java-only verification can be cancelled while JNI is idle. Such a cancellation
            // belongs to that earlier job and must not poison this new, unrelated request.
            engine.cancel();
            CheckedAnswer unicodeAnswer = complete(engine, unicode, "unicode-after-idle-cancel", "Café 🌲");
            assertFalse("Native output contained replacement characters.",
                    evidence(unicodeAnswer).contains("\uFFFD"));

            AtomicBoolean cancellationRequested = new AtomicBoolean();
            boolean cancelled = false;
            try {
                generate(engine, prompt(acadia), "cancelled", count -> {
                    if (count > 0 && cancellationRequested.compareAndSet(false, true)) engine.cancel();
                });
            } catch (CancellationException expected) {
                cancelled = true;
            }
            assertTrue("Cancellation must be requested during real token generation.",
                    cancellationRequested.get());
            assertTrue("A cancelled request must not return a displayable partial completion.", cancelled);
            CheckedAnswer afterCancel = complete(engine, acadia, "acadia-after-cancel", "30 days");
            assertEquals("Cancellation left state in the next request.", snapshot(first), snapshot(afterCancel));
            if (SmolLm3Protocol.ID.equals(protocol)) {
                // A syntactically plausible prefix is not a complete model answer: JNI
                // must observe this model's real EOS, never accept budget truncation.
                assertThrows(IllegalStateException.class, () -> engine.generate(prompt(acadia), 1, null));
            }
        }
    }

    private void verifyPublisherTokenParity(NativeEngine engine, Protocol.Request acadia,
                                                   Protocol.Request katahdin, Protocol.Request atlantic) throws Exception {
        if (SmolLm3Protocol.ID.equals(protocol)) {
            verifySmolPublisherTokenParity(engine, acadia, katahdin, atlantic);
            return;
        }
        // Expected IDs are independently produced by the pinned HF tokenizers0.22.2
        // tokenizer.json (SHA256 dd7acf61196c61741cf95cc0f0a2b920443e44094438d325ab6e65a048fe8be4).
        // Recorded in Actions40315924744 framing-diagnostic.json; they are identical
        // for the publisher's350M/1B tokenizer and our corrected GGUF vocabulary.
        assertArrayEquals("Acadia's real JNI tokens differ from the publisher.", new int[] {
            2, 65517, 6601, 1465, 351, 265, 1222, 6893, 34, 65518, 189, 65519, 65520, 20,
            39898, 268, 37075, 4294, 29, 53013, 6820, 712, 359, 8099, 1706, 1374, 3411, 17,
            9565, 691, 86, 2384, 1800, 59577, 35335, 2921, 249, 2664, 3863, 251, 17,
            65521, 189, 65522, 189
        }, engine.tokenizeForTest(Protocol.prompt(acadia, CONTEXT, OUTPUT)));
        assertArrayEquals("Katahdin's real JNI tokens differ from the publisher.", new int[] {
            2, 65517, 6869, 12889, 11056, 2035, 265, 33226, 3168, 1281, 1952, 34, 65518,
            189, 65519, 65520, 20, 39898, 268, 5717, 13715, 29, 249, 33226, 3168, 1281,
            8491, 3781, 49466, 11277, 18469, 799, 284, 1800, 14765, 36978, 15, 5067, 296,
            3781, 8, 12889, 17, 1159, 351, 45924, 37075, 1080, 15, 529, 249, 2556, 12398,
            17, 65521, 189, 65522, 189
        }, engine.tokenizeForTest(Protocol.prompt(katahdin, CONTEXT, OUTPUT)));
        assertArrayEquals("Atlantic's real JNI tokens differ from the publisher.", new int[] {
            2, 65517, 6869, 351, 265, 3175, 11608, 11076, 34, 65518, 189, 65519, 65520, 20,
            39898, 268, 2647, 4893, 29, 265, 37075, 3175, 590, 11608, 11076, 351, 22686,
            22015, 949, 17, 1522, 12028, 15, 29273, 406, 3208, 3292, 659, 916, 16140, 17,
            65521, 189, 65522, 189
        }, engine.tokenizeForTest(Protocol.prompt(atlantic, CONTEXT, OUTPUT)));
        Protocol.Request unicodeDigits = Protocol.request("atlantic",
                "What's 123456789? café, naïve, 東京 🌲", Collections.emptyList());
        assertArrayEquals("Unicode/digit JNI tokens differ from the publisher.", new int[] {
            2, 65517, 6869, 590, 47480, 3344, 23309, 34, 37378, 15, 63585, 345, 15, 211,
            59941, 52242, 225, 114, 65518, 189, 65522, 189
        }, engine.tokenizeForTest(Protocol.prompt(unicodeDigits, CONTEXT, OUTPUT)));

        String prompt = Protocol.prompt(acadia, CONTEXT, OUTPUT);
        assertMalformedNativePromptRejected(engine, prompt.substring("<|end_of_text|>".length()));
        assertMalformedNativePromptRejected(engine, "<|end_of_text|>" + prompt);
        assertMalformedNativePromptRejected(engine, prompt.substring(0, prompt.length() - 1));
    }

    private void verifySmolPublisherTokenParity(NativeEngine engine, Protocol.Request acadia,
                                                Protocol.Request katahdin, Protocol.Request atlantic) throws Exception {
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        JSONObject evidence;
        try (InputStream input = testContext.getAssets().open("smollm3-publisher-parity.json");
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096]; int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
                assertTrue("Tokenizer fixture unexpectedly large.", output.size() <= 128 * 1024);
            }
            evidence = new JSONObject(output.toString(StandardCharsets.UTF_8.name()));
        }
        assertEquals("a07cc9a04f16550a088caea529712d1d335b0ac1", evidence.getString("publisher_revision"));
        assertEquals("7b6a500b662a34eb3f0374db856ba4ad7de4c81040571d78dc0d357238930005",
                evidence.getString("tokenizer_sha256"));
        Protocol.Request unicodeDigits = Protocol.request("atlantic",
                "What's 123456789? café, naïve, 東京 🌲", Collections.emptyList());
        Protocol.Request[] requests = {acadia, katahdin, atlantic, unicodeDigits};
        String[] names = {"acadia", "katahdin", "atlantic", "unicode-digits"};
        JSONObject fixtures = evidence.getJSONObject("fixtures");
        for (int i = 0; i < requests.length; i++) {
            JSONObject fixture = fixtures.getJSONObject(names[i]);
            String rawPrompt = prompt(requests[i]);
            assertEquals("Java must preserve the independent publisher prompt bytes.",
                    fixture.getString("prompt"), rawPrompt);
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(rawPrompt.getBytes(StandardCharsets.UTF_8));
            StringBuilder hash = new StringBuilder();
            for (byte value : digest) hash.append(String.format(Locale.ROOT, "%02x", value & 255));
            assertEquals(fixture.getString("prompt_sha256"), hash.toString());
            JSONArray ids = fixture.getJSONArray("input_token_ids");
            int[] expected = new int[ids.length()];
            for (int j = 0; j < ids.length(); j++) expected[j] = ids.getInt(j);
            assertArrayEquals("SmolLM3 JNI tokens differ from the independent publisher tokenizer: " + names[i],
                    expected, engine.tokenizeForTest(rawPrompt));
        }
        String rawPrompt = prompt(acadia);
        assertMalformedNativePromptRejected(engine, "<|begin_of_text|>" + rawPrompt);
        assertMalformedNativePromptRejected(engine, rawPrompt.substring(0, rawPrompt.length() - 1));
        assertMalformedNativePromptRejected(engine, rawPrompt.replace("<|im_start|>user\n", "<|im_start|>assistant\n"));
        assertMalformedNativePromptRejected(engine, rawPrompt.replace("How long", "<|end_of_text|>How long"));
        for (String marker : new String[]{"<think>", "</think>", "<tool_response>", "</tool_response>",
                "<tool_call>", "</tool_call>", "<code>", "</code>"}) {
            assertMalformedNativePromptRejected(engine, rawPrompt.replace("How long", marker + "How long"));
        }
        assertMalformedNativePromptRejected(engine, Protocol.prompt(acadia, CONTEXT, OUTPUT));
    }

    private static void assertMalformedNativePromptRejected(NativeEngine engine, String prompt) {
        try {
            engine.tokenizeForTest(prompt);
            fail("Native tokenization accepted malformed publisher framing.");
        } catch (IllegalArgumentException expected) {
            // A rejected framing check must not consume or corrupt the next valid operation.
        }
    }

    private static Protocol.Request request(String profile, String question, String source) {
        String sourceId = "acadia".equals(profile) ? "acadia/demo-policy"
                : "katahdin".equals(profile) ? "katahdin/demo-result" : "atlantic/demo-note";
        return Protocol.request(profile, question, Collections.singletonList(
                new Protocol.Source(sourceId, source)));
    }

    private int outputTokens() { return SmolLm3Protocol.ID.equals(protocol) ? 384 : OUTPUT; }

    private String prompt(Protocol.Request request) {
        return SmolLm3Protocol.ID.equals(protocol)
                ? SmolLm3Protocol.qaPrompt(request, CONTEXT, outputTokens(), REFERENCE_DATE)
                : Protocol.prompt(request, CONTEXT, outputTokens());
    }

    private static final class CheckedAnswer {
        final String text, status;
        final List<String> ids = new ArrayList<>(), quotes = new ArrayList<>();
        CheckedAnswer(Protocol.Answer answer) {
            text = answer.text; status = answer.status;
            for (Protocol.Citation citation : answer.citations) { ids.add(citation.sourceId); quotes.add(citation.quote); }
        }
        CheckedAnswer(SmolLm3Protocol.Answer answer) {
            text = answer.text; status = answer.status;
            assertFalse("This supported fixture must not claim missing evidence.", answer.insufficientEvidence);
            for (SmolLm3Protocol.Citation citation : answer.citations) { ids.add(citation.sourceId); quotes.add(citation.quote); }
        }
    }

    private CheckedAnswer complete(NativeEngine engine, Protocol.Request request,
                                           String fixture, String requiredFact) throws Exception {
        String raw = generate(engine, prompt(request), fixture, count -> { });
        CheckedAnswer answer;
        try {
            answer = SmolLm3Protocol.ID.equals(protocol)
                    ? new CheckedAnswer(SmolLm3Protocol.parseQaAnswer(raw, request))
                    : new CheckedAnswer(Protocol.parseAnswer(raw, request));
        } catch (Protocol.ValidationException failure) {
            throw new AssertionError("No complete, valid answer for actual-model fixture " + fixture
                    + ". Raw generation is deliberately omitted from logs.", failure);
        }
        assertEquals("The parser must preserve its semantic verification limitation.",
                SmolLm3Protocol.ID.equals(protocol) ? "semantic_review_not_performed" : "needs_human_review", answer.status);
        assertFalse("This supported fixture requires at least one literal source citation.", answer.quotes.isEmpty());
        for (int i = 0; i < answer.quotes.size(); i++) {
            assertEquals("A citation crossed profile boundaries.", request.sources.get(0).id, answer.ids.get(i));
            assertTrue("A citation did not match the supplied fixture.",
                    Protocol.quoteMatches(answer.quotes.get(i), request.sources.get(0).text));
        }
        assertTrue("Actual inference did not recover the supplied fact for " + fixture + ".",
                answer.text.toLowerCase(Locale.ROOT).contains(requiredFact.toLowerCase(Locale.ROOT)));
        assertFalse("Internal protocol tokens reached the displayed answer.", answer.text.contains("<|"));
        logEvent(new JSONObject().put("event", "validated_answer").put("fixture", fixture)
                .put("citations", answer.quotes.size()).put("status", answer.status));
        return answer;
    }

    private String generate(NativeEngine engine, String prompt, String fixture,
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
            String result = engine.generate(prompt, outputTokens(), count -> {
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

    private static String evidence(CheckedAnswer answer) {
        StringBuilder result = new StringBuilder(answer.text);
        for (String quote : answer.quotes) result.append('\n').append(quote);
        return result.toString().toLowerCase(Locale.ROOT);
    }

    private static String snapshot(CheckedAnswer answer) {
        StringBuilder result = new StringBuilder(answer.text);
        for (int i = 0; i < answer.quotes.size(); i++)
            result.append('\n').append(answer.ids.get(i)).append(':').append(answer.quotes.get(i));
        return result.toString();
    }

    private static void logEvent(JSONObject event) {
        // Fixtures, counts and public model identity only; no prompt, raw output or reasoning.
        Log.i(LOG_TAG, event.toString());
    }
}
