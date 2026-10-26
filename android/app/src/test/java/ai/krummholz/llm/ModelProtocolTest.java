package ai.krummholz.llm;

import java.time.LocalDate;
import java.util.Collections;
import org.junit.Test;
import static org.junit.Assert.*;

public class ModelProtocolTest {
    private Protocol.Request request() {
        return Protocol.request("acadia", "Which label?", Collections.singletonList(new Protocol.Source("acadia/x", "The label is Cedar.")));
    }
    @Test public void manifestProtocolExplicitlySelectsFramingAndBudget() {
        assertEquals(384, ModelProtocol.outputTokens("smollm3-chat-v1"));
        assertEquals(1024, ModelProtocol.outputTokens("pleias-rag-v1"));
        assertTrue(ModelProtocol.prompt("smollm3-chat-v1", request(), 4096, LocalDate.of(2026, 10, 26)).startsWith("<|im_start|>system\n"));
        assertTrue(ModelProtocol.prompt("pleias-rag-v1", request(), 4096, LocalDate.of(2026, 10, 26)).startsWith("<|end_of_text|><|query_start|>"));
        for (String unknown : new String[]{"", "SmolLM3", "automatic", null}) {
            assertThrows(Protocol.ValidationException.class, () -> ModelProtocol.outputTokens(unknown));
            assertThrows(Protocol.ValidationException.class, () -> ModelProtocol.parse(unknown, "{}", request()));
        }
    }
    @Test public void providersCannotSilentlyParseTheOtherProtocol() {
        String smol = "{\"answer\":\"Cedar\",\"citations\":[{\"source_id\":\"acadia/x\",\"quote\":\"Cedar\"}],\"insufficient_evidence\":false}";
        String pleias = "<|answer_start|>Cedar <ref name=\"1\">Cedar</ref><|answer_end|>";
        assertThrows(Protocol.ValidationException.class, () -> ModelProtocol.parse(ModelProtocol.PLEIAS, smol, request()));
        assertThrows(Protocol.ValidationException.class, () -> ModelProtocol.parse(SmolLm3Protocol.ID, pleias, request()));
        ModelProtocol.Answer answer = ModelProtocol.parse(SmolLm3Protocol.ID, smol, request());
        assertTrue(answer.separateCitations);
        assertEquals("Cedar", answer.text);
        assertEquals("acadia/x", answer.citations.get(0).sourceId);
        assertThrows(UnsupportedOperationException.class, () -> answer.citations.clear());
        assertFalse(ModelProtocol.parse(ModelProtocol.PLEIAS, pleias, request()).separateCitations);
    }
    @Test public void sourceQaCannotAcceptAnInertToolProposalAsItsAnswer() {
        assertThrows(Protocol.ValidationException.class, () -> ModelProtocol.parse(SmolLm3Protocol.ID,
            "<tool_call>{\"name\":\"search_public_web\",\"arguments\":{\"query\":\"weather\"}}</tool_call>", request()));
    }
}
