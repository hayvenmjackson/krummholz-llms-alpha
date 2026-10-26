package ai.krummholz.llm;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import static org.junit.Assert.*;

public class SmolLm3ProtocolTest {
    private static final LocalDate FROZEN = LocalDate.of(2026, 10, 26);
    private Protocol.Request request() {
        return Protocol.request("acadia", "What is the return window?", Arrays.asList(
            new Protocol.Source("acadia/policy", "Unused coats may be returned within 21 calendar days after delivery."),
            new Protocol.Source("acadia/shipping", "The delivery estimate is seven working days.")));
    }
    private String valid() {
        return "{\"answer\":\"Unused coats: 21 calendar days after delivery.\",\"citations\":["
            + "{\"source_id\":\"acadia/policy\",\"quote\":\"21 calendar days after delivery\"}],\"insufficient_evidence\":false}";
    }
    private void reject(String raw) {
        assertThrows(Protocol.ValidationException.class, () -> SmolLm3Protocol.parseQaAnswer(raw, request()));
    }
    private static String sha(String text) throws Exception {
        byte[] bytes = MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8));
        StringBuilder hash = new StringBuilder();
        for (byte value : bytes) hash.append(String.format(java.util.Locale.ROOT, "%02x", value & 255));
        return hash.toString();
    }
    @Test public void sevenPromptsExactlyMatchRetainedPublisherEvidence() throws Exception {
        Map<?, ?> fixtures;
        try (InputStream source = getClass().getResourceAsStream("/smollm3-parity.json")) {
            assertNotNull(source);
            fixtures = (Map<?, ?>) StrictJson.parse(new String(source.readAllBytes(), StandardCharsets.UTF_8));
        }
        assertEquals(SmolLm3Protocol.PUBLISHER_REVISION, fixtures.get("publisher_revision"));
        assertEquals(SmolLm3Protocol.TEMPLATE_SHA256, fixtures.get("template_sha256"));
        List<?> rows = (List<?>) fixtures.get("cases");
        assertEquals(7, rows.size());
        for (Object raw : rows) {
            Map<?, ?> row = (Map<?, ?>) raw;
            Map<?, ?> data = (Map<?, ?>) row.get("request");
            List<Protocol.Source> sources = new ArrayList<>();
            for (Object rawSource : (List<?>) data.get("sources")) {
                Map<?, ?> source = (Map<?, ?>) rawSource;
                sources.add(new Protocol.Source((String) source.get("id"), (String) source.get("text")));
            }
            Protocol.Request request = Protocol.request((String) row.get("profile"), (String) data.get("query"), sources);
            String prompt = "qa".equals(row.get("kind"))
                ? SmolLm3Protocol.qaPrompt(request, 4096, 384, FROZEN)
                : SmolLm3Protocol.toolPrompt(request, 4096, 384, FROZEN);
            assertEquals(row.get("id").toString(), row.get("prompt"), prompt);
            assertEquals(row.get("prompt_sha256"), sha(prompt));
        }
    }
    @Test public void dateIsExplicitAndNoThinkingFramingIsFixed() {
        String prompt = SmolLm3Protocol.qaPrompt(request(), 4096, 384, LocalDate.of(2027, 1, 3));
        assertTrue(prompt.contains("Today Date: 03 January 2027\n"));
        assertTrue(prompt.contains("Reasoning Mode: /no_think\n"));
        assertTrue(prompt.endsWith("<|im_start|>assistant\n<think>\n\n</think>\n"));
        assertEquals(1, prompt.split("<\\|im_end\\|>", -1).length - 1);
        assertFalse(prompt.contains("<|end_of_text|>"));
        assertFalse(prompt.contains("### Tools"));
    }
    @Test public void promptJsonEscapesRequestTextWithoutChangingUnicode() {
        Protocol.Request request = Protocol.request("acadia", "Café 🌲? \"quoted\"\nnext", Collections.singletonList(
            new Protocol.Source("acadia/x", "Literal slash \\ and tab\t.")));
        String prompt = SmolLm3Protocol.qaPrompt(request, 4096, 384, FROZEN);
        assertTrue(prompt.contains("\"query\": \"Café 🌲? \\\"quoted\\\"\\nnext\""));
        assertTrue(prompt.contains("Literal slash \\\\ and tab\\t."));
    }
    @Test public void contextAndOutputBoundsFailBeforeNativeGeneration() {
        assertThrows(Protocol.ValidationException.class, () -> SmolLm3Protocol.qaPrompt(request(), 4096, 385, FROZEN));
        assertThrows(Protocol.ValidationException.class, () -> SmolLm3Protocol.qaPrompt(request(), 512, 384, FROZEN));
        Protocol.Request longRequest = Protocol.request("acadia", "Question", Collections.singletonList(new Protocol.Source("acadia/x", "树".repeat(900))));
        assertThrows(Protocol.ValidationException.class, () -> SmolLm3Protocol.qaPrompt(longRequest, 4096, 384, FROZEN));
    }
    @Test public void everyPinnedAddedTokenIsRejectedInUserPayloads() throws Exception {
        Map<?, ?> fixtures;
        try (InputStream source = getClass().getResourceAsStream("/smollm3-parity.json")) {
            fixtures = (Map<?, ?>) StrictJson.parse(new String(source.readAllBytes(), StandardCharsets.UTF_8));
        }
        assertEquals(SmolLm3Protocol.TOKENIZER_CONFIG_SHA256, fixtures.get("tokenizer_config_sha256"));
        List<?> tokens = (List<?>) fixtures.get("added_token_literals");
        assertEquals(256, tokens.size());
        for (Object rawToken : tokens) {
            String token = (String) rawToken;
            assertThrows(token, Protocol.ValidationException.class, () -> {
                Protocol.Request input = Protocol.request("acadia", "Question " + token, request().sources);
                SmolLm3Protocol.qaPrompt(input, 4096, 384, FROZEN);
            });
            assertThrows(token, Protocol.ValidationException.class, () -> {
                Protocol.Request input = Protocol.request("acadia", "Question", Collections.singletonList(new Protocol.Source("acadia/x", "Text " + token)));
                SmolLm3Protocol.toolPrompt(input, 4096, 384, FROZEN);
            });
        }
    }
    @Test public void ordinaryLessThanAndNonreservedMarkupRemainPlainText() {
        Protocol.Request input = Protocol.request("acadia", "Is 2 < 3?", Collections.singletonList(
            new Protocol.Source("acadia/x", "The note says 2 < 3; <section> is ordinary text.")));
        assertTrue(SmolLm3Protocol.qaPrompt(input, 4096, 384, FROZEN).contains("2 < 3"));
    }
    @Test public void separateCitationsDoNotInventInlineAttachments() {
        SmolLm3Protocol.Answer answer = SmolLm3Protocol.parseQaAnswer(valid(), request());
        assertEquals("Unused coats: 21 calendar days after delivery.", answer.text);
        assertFalse(answer.text.contains("[1]"));
        assertEquals("separate_list", answer.citationPresentation);
        assertEquals(1, answer.citations.get(0).number);
        assertEquals(1, answer.citations.get(0).sourceNumber);
        assertEquals("semantic_review_not_performed", answer.status);
        assertTrue(answer.warning.contains("does not verify"));
        assertThrows(UnsupportedOperationException.class, () -> answer.citations.clear());
    }
    @Test public void wrongPassageQuoteAndUnknownSourceAreRejected() {
        reject(valid().replace("acadia/policy", "acadia/shipping"));
        reject(valid().replace("acadia/policy", "acadia/unknown"));
        reject(valid().replace("21 calendar days after delivery", "365 days after delivery"));
    }
    @Test public void exactQuotesCannotNormalizeWhitespaceOrUseEllipsis() {
        reject(valid().replace("21 calendar days after delivery", "21  calendar days after delivery"));
        reject(valid().replace("21 calendar days after delivery", "21 calendar (…) delivery"));
        reject(valid().replace("21 calendar days after delivery", "21 calendar days after Delivery"));
    }
    @Test public void completeStrictSchemaRejectsCoercionExtraFieldsAndDuplicateKeys() {
        reject(valid().replace("false", "\"false\""));
        reject(valid().replace("false", "null"));
        reject(valid().replace("\"insufficient_evidence\":false", "\"unexpected\":false"));
        reject(valid().replace("\"citations\":", "\"tool_calls\":[],\"citations\":"));
        reject(valid().replace("\"answer\":", "\"answer\":\"different\",\"answer\":"));
        reject(valid().replace("\"source_id\":", "\"source_id\":\"acadia/shipping\",\"source_id\":"));
        reject(valid().replace("\"source_id\":\"acadia/policy\"", "\"source_id\":1"));
    }
    @Test public void rawReasoningEosTruncationMarkdownAndMultipleObjectsAreWithheld() {
        for (String raw : new String[]{"<think>private</think>" + valid(), valid() + "<|im_end|>",
                valid() + " [end of text]", valid().substring(0, valid().length() - 1),
                "```json\n" + valid() + "\n```", valid() + valid(), "<tool_call>" + valid() + "</tool_call>"})
            reject(raw);
    }
    @Test public void numberedReferencesReservedMarkersAndUnicodeBlankAnswersAreRejected() {
        reject(valid().replace("Unused coats: 21 calendar days after delivery.", "Answer [1]"));
        reject(valid().replace("Unused coats: 21 calendar days after delivery.", "<|im_start|>"));
        for (String token : new String[]{"<think>", "</think>", "<tool_call>", "</tool_call>",
                "<tool_response>", "</tool_response>", "<code>", "</code>"})
            reject(valid().replace("Unused coats: 21 calendar days after delivery.", token));
        reject(valid().replace("Unused coats: 21 calendar days after delivery.", "\\ud800"));
        reject(valid().replace("Unused coats: 21 calendar days after delivery.", "\u00a0\u0085"));
    }
    @Test public void missingEvidenceCanAbstainButAnUncitedAssertionCannotPassParsing() {
        String answer = "{\"answer\":\"The date is absent from these sources.\",\"citations\":[],\"insufficient_evidence\":true}";
        assertTrue(SmolLm3Protocol.parseQaAnswer(answer, request()).insufficientEvidence);
        reject(answer.replace("true", "false"));
    }
    @Test public void correctQuotedNumberDoesNotProveMeaningOrExecuteAnAction() {
        String wrong = valid().replace("Unused coats: 21 calendar days after delivery.", "Every item:21business days after dispatch. I emailed approval.");
        SmolLm3Protocol.Answer answer = SmolLm3Protocol.parseQaAnswer(wrong, request());
        assertEquals("semantic_review_not_performed", answer.status);
        assertTrue(answer.warning.contains("does not verify"));
        // This fabricated unsupported claim remains a semantic failure, not a parser-proven fact.
    }
    @Test public void searchProposalIsInertAndContainsNoActionAuthority() {
        String raw = "<tool_call>{\"name\":\"search_public_web\",\"arguments\":{\"query\":\"Belfast Maine current weather\"}}</tool_call>";
        SmolLm3Protocol.InertToolProposal proposal = SmolLm3Protocol.parseToolProposal(raw, request());
        assertEquals("search_public_web", proposal.name);
        assertEquals("Belfast Maine current weather", proposal.query);
        assertNull(proposal.answer);
        assertFalse(proposal.executed);
        assertTrue(proposal.authorizedActions.isEmpty());
        assertThrows(UnsupportedOperationException.class, () -> proposal.authorizedActions.add("search"));
    }
    @Test public void toolAnswerUsesTheSameExactQuoteChecks() {
        String answer = valid().replace(",\"insufficient_evidence\":false", "");
        String raw = "<tool_call>{\"name\":\"answer\",\"arguments\":" + answer + "}</tool_call>";
        SmolLm3Protocol.InertToolProposal proposal = SmolLm3Protocol.parseToolProposal(raw, request());
        assertFalse(proposal.executed);
        assertNotNull(proposal.answer);
        assertThrows(Protocol.ValidationException.class, () -> SmolLm3Protocol.parseToolProposal(raw.replace("21 calendar", "90 calendar"), request()));
    }
    @Test public void toolAllowlistAndSingleProposalSchemaAreStrict() {
        String raw = "<tool_call>{\"name\":\"search_public_web\",\"arguments\":{\"query\":\"weather\"}}</tool_call>";
        for (String bad : new String[]{raw + raw, raw.replace("search_public_web", "send_email"),
                raw.replace("\"query\":\"weather\"", "\"query\":\"weather\",\"url\":\"https://example.com\""),
                raw.replace("\"arguments\":", "\"authorized\":true,\"arguments\":"),
                raw.replace("\"query\":\"weather\"", "\"query\":42"), "<think>x</think>" + raw})
            assertThrows(Protocol.ValidationException.class, () -> SmolLm3Protocol.parseToolProposal(bad, request()));
    }
}
