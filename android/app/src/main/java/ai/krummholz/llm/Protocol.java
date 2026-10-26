package ai.krummholz.llm;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Literal protocol validation, kept in sync with krummholz_llm/protocol.py. */
public final class Protocol {
    private Protocol() {}
    private static final Pattern RESERVED = Pattern.compile("<\\|[^<>]*\\|>|\\[PAD\\]|\\[UNK\\]");
    private static final Pattern CONTROL = Pattern.compile("[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]");
    private static final Pattern CITATION = Pattern.compile("<ref name=\"(?:<\\|source_id\\|>)?([1-9][0-9]*)\">(.*?)</ref>", Pattern.DOTALL);
    private static final Pattern MALFORMED_REF = Pattern.compile("<\\s*/?\\s*ref\\b", Pattern.CASE_INSENSITIVE);
    private static final Pattern ELLIPSIS = Pattern.compile("\\(\\s*(?:…|\\.{3})\\s*\\)|…");
    private static final String START = "<|answer_start|>", END = "<|answer_end|>";

    public static final class ValidationException extends IllegalArgumentException {
        public ValidationException(String message) { super(message); }
    }
    public static final class Source {
        public final String id, text;
        public Source(String id, String text) { this.id = id; this.text = text; }
    }
    public static final class Request {
        public final String profile, query;
        public final List<Source> sources;
        private Request(String profile, String query, List<Source> sources) {
            this.profile = profile; this.query = query;
            this.sources = Collections.unmodifiableList(sources);
        }
    }
    public static final class Citation {
        public final int number, sourceNumber;
        public final String sourceId, quote;
        private Citation(int number, int sourceNumber, String sourceId, String quote) {
            this.number = number; this.sourceNumber = sourceNumber;
            this.sourceId = sourceId; this.quote = quote;
        }
    }
    public static final class Answer {
        public final String text;
        public final List<Citation> citations;
        public final String status = "needs_human_review";
        public final String warning;
        private Answer(String text, List<Citation> citations) {
            this.text = text; this.citations = Collections.unmodifiableList(citations);
            warning = citations.isEmpty()
                ? "No verified citations. Check whether the answer should abstain."
                : "Quotes match the source text. This does not verify the answer’s meaning or completeness.";
        }
    }

    public static Request request(String profile, String query, List<Source> sourceData) {
        if (!"acadia".equals(profile) && !"katahdin".equals(profile) && !"atlantic".equals(profile))
            throw new ValidationException("Choose a known profile.");
        query = safeText(query, "Question", 2048);
        if (sourceData == null || sourceData.size() > 10)
            throw new ValidationException("Use at most 10 sources.");
        List<Source> sources = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (Source source : sourceData) {
            if (source == null) throw new ValidationException("A source is missing.");
            String id = safeText(source.id, "Source ID", 120);
            if (!id.matches(Pattern.quote(profile) + "/[A-Za-z0-9_.-]+"))
                throw new ValidationException("The source belongs to a different profile.");
            if (!seen.add(id)) throw new ValidationException("Source IDs must be unique.");
            sources.add(new Source(id, safeText(source.text, "Source", 12000)));
        }
        return new Request(profile, query, sources);
    }

    public static String prompt(Request request, int contextTokens, int outputTokens) {
        if (contextTokens < 512 || contextTokens > 4096 || outputTokens < 32 || outputTokens > 1024)
            throw new ValidationException("Unsupported context or answer budget.");
        // Match the publisher tokenizer postprocessor: one initial EOS, no automatic BOS.
        StringBuilder text = new StringBuilder("<|end_of_text|><|query_start|>").append(request.query).append("<|query_end|>\n");
        for (int i = 0; i < request.sources.size(); i++)
            text.append("<|source_start|><|source_id|>").append(i + 1).append(' ')
                .append(request.sources.get(i).text).append("<|source_end|>\n");
        text.append("<|language_start|>\n");
        int bytes = text.toString().getBytes(StandardCharsets.UTF_8).length;
        if (bytes + outputTokens + 64 > contextTokens)
            throw new ValidationException("This source and question are too long for the alpha. Shorten them and try again.");
        return text.toString();
    }

    public static Answer parseAnswer(String raw, Request request) {
        if (raw == null || count(raw, START) != 1 || count(raw, END) != 1)
            throw new ValidationException("The model did not produce one complete answer. Try a shorter question.");
        int start = raw.indexOf(START) + START.length(), end = raw.indexOf(END);
        if (end <= start) throw new ValidationException("The model’s answer was incomplete.");
        String answer = strip(raw.substring(start, end));
        if (answer.isEmpty() || CONTROL.matcher(answer).find())
            throw new ValidationException("The model’s answer was empty or malformed.");
        List<Citation> citations = new ArrayList<>();
        Matcher matcher = CITATION.matcher(answer);
        StringBuffer clean = new StringBuffer();
        while (matcher.find()) {
            String digits = matcher.group(1);
            if (digits.length() > Integer.toString(request.sources.size()).length())
                throw new ValidationException("A citation referred to a source that was not supplied.");
            int sourceNumber = Integer.parseInt(digits);
            if (sourceNumber > request.sources.size())
                throw new ValidationException("A citation referred to a source that was not supplied.");
            Source source = request.sources.get(sourceNumber - 1);
            String quote = strip(matcher.group(2));
            if (RESERVED.matcher(quote).find() || !quoteMatches(quote, source.text))
                throw new ValidationException("A citation did not match its source. The answer has been withheld.");
            citations.add(new Citation(citations.size() + 1, sourceNumber, source.id, quote));
            matcher.appendReplacement(clean, Matcher.quoteReplacement("[" + citations.size() + "]"));
        }
        matcher.appendTail(clean);
        if (MALFORMED_REF.matcher(clean).find() || RESERVED.matcher(clean).find())
            throw new ValidationException("The model returned malformed answer markup. The answer has been withheld.");
        return new Answer(clean.toString(), citations);
    }

    public static boolean quoteMatches(String quote, String source) {
        String haystack = normalized(source);
        int position = 0, pieces = 0;
        for (String part : ELLIPSIS.split(normalized(quote), -1)) {
            String piece = strip(part);
            if (piece.isEmpty()) continue;
            int found = haystack.indexOf(piece, position);
            if (found < 0) return false;
            position = found + piece.length(); pieces++;
        }
        return pieces > 0;
    }
    private static String safeText(String value, String name, int limit) {
        if (value == null) throw new ValidationException(name + " is missing.");
        value = strip(value);
        if (value.isEmpty()) throw new ValidationException(name + " must not be empty.");
        if (value.codePointCount(0, value.length()) > limit)
            throw new ValidationException(name + " is too long.");
        if (CONTROL.matcher(value).find() || RESERVED.matcher(value).find())
            throw new ValidationException(name + " contains a reserved token or unsupported control character.");
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (Character.isHighSurrogate(c)) {
                if (++i >= value.length() || !Character.isLowSurrogate(value.charAt(i)))
                    throw new ValidationException(name + " contains invalid Unicode.");
            } else if (Character.isLowSurrogate(c)) throw new ValidationException(name + " contains invalid Unicode.");
        }
        return value;
    }
    private static int count(String text, String needle) {
        int count = 0, at = 0;
        while ((at = text.indexOf(needle, at)) >= 0) { count++; at += needle.length(); }
        return count;
    }
    private static boolean whitespace(int cp) {
        return Character.isWhitespace(cp) || Character.isSpaceChar(cp) || cp == 0x85;
    }
    private static String strip(String text) {
        int start = 0, end = text.length();
        while (start < end && whitespace(text.codePointAt(start))) start += Character.charCount(text.codePointAt(start));
        while (end > start && whitespace(text.codePointBefore(end))) end -= Character.charCount(text.codePointBefore(end));
        return text.substring(start, end);
    }
    private static String normalized(String text) {
        StringBuilder out = new StringBuilder(); boolean space = false;
        for (int i = 0; i < text.length();) {
            int cp = text.codePointAt(i); i += Character.charCount(cp);
            if (whitespace(cp)) { space = out.length() > 0; continue; }
            if (space) out.append(' ');
            space = false; out.appendCodePoint(cp);
        }
        return out.toString();
    }
}
