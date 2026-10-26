package ai.krummholz.llm;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Explicit manifest-to-protocol dispatch. No guessing from model output. */
public final class ModelProtocol {
    public static final String PLEIAS = "pleias-rag-v1";
    private ModelProtocol() {}

    public static final class Citation {
        public final int number;
        public final String sourceId, quote;
        private Citation(int number, String sourceId, String quote) {
            this.number = number; this.sourceId = sourceId; this.quote = quote;
        }
    }
    public static final class Answer {
        public final String text, warning;
        public final boolean separateCitations;
        public final List<Citation> citations;
        private Answer(String text, String warning, boolean separateCitations, List<Citation> citations) {
            this.text = text; this.warning = warning; this.separateCitations = separateCitations;
            this.citations = Collections.unmodifiableList(new ArrayList<>(citations));
        }
    }
    public static int outputTokens(String protocol) {
        if (SmolLm3Protocol.ID.equals(protocol)) return SmolLm3Protocol.MAX_OUTPUT_TOKENS;
        if (PLEIAS.equals(protocol)) return 1024;
        throw unknown();
    }
    public static String prompt(String protocol, Protocol.Request request, int contextTokens, LocalDate date) {
        int outputTokens = outputTokens(protocol);
        if (SmolLm3Protocol.ID.equals(protocol)) return SmolLm3Protocol.qaPrompt(request, contextTokens, outputTokens, date);
        return Protocol.prompt(request, contextTokens, outputTokens);
    }
    public static Answer parse(String protocol, String raw, Protocol.Request request) {
        outputTokens(protocol); // Fail closed before parsing any unknown provider.
        List<Citation> citations = new ArrayList<>();
        if (SmolLm3Protocol.ID.equals(protocol)) {
            SmolLm3Protocol.Answer answer = SmolLm3Protocol.parseQaAnswer(raw, request);
            for (SmolLm3Protocol.Citation citation : answer.citations)
                citations.add(new Citation(citation.number, citation.sourceId, citation.quote));
            return new Answer(answer.text, answer.warning, true, citations);
        }
        Protocol.Answer answer = Protocol.parseAnswer(raw, request);
        for (Protocol.Citation citation : answer.citations)
            citations.add(new Citation(citation.number, citation.sourceId, citation.quote));
        return new Answer(answer.text, answer.warning, false, citations);
    }
    private static Protocol.ValidationException unknown() {
        return new Protocol.ValidationException("This model has no supported protocol. Import the model approved for this app version.");
    }
}
