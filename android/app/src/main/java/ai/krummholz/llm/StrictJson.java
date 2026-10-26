package ai.krummholz.llm;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Small strict JSON reader for model output; no Android lenient JSON coercion. */
final class StrictJson {
    private static final int MAX_LENGTH = 65536, MAX_DEPTH = 16, MAX_ITEMS = 256;
    private final String input;
    private int position;
    private StrictJson(String input) { this.input = input; }

    static Object parse(String input) {
        if (input == null || input.length() > MAX_LENGTH) throw invalid();
        StrictJson reader = new StrictJson(input);
        Object value = reader.value(0);
        reader.space();
        if (reader.position != input.length()) throw invalid();
        return value;
    }

    private Object value(int depth) {
        if (depth > MAX_DEPTH) throw invalid();
        space();
        if (position >= input.length()) throw invalid();
        char next = input.charAt(position);
        if (next == '"') return string();
        if (next == '{') {
            position++;
            Map<String, Object> map = new LinkedHashMap<>();
            space();
            if (take('}')) return map;
            do {
                space();
                if (position >= input.length() || input.charAt(position) != '"') throw invalid();
                String key = string();
                if (map.containsKey(key) || map.size() >= MAX_ITEMS) throw invalid();
                space();
                if (!take(':')) throw invalid();
                map.put(key, value(depth + 1));
                space();
                if (take('}')) return map;
            } while (take(','));
            throw invalid();
        }
        if (next == '[') {
            position++;
            List<Object> list = new ArrayList<>();
            space();
            if (take(']')) return list;
            do {
                if (list.size() >= MAX_ITEMS) throw invalid();
                list.add(value(depth + 1));
                space();
                if (take(']')) return list;
            } while (take(','));
            throw invalid();
        }
        if (literal("true")) return Boolean.TRUE;
        if (literal("false")) return Boolean.FALSE;
        if (literal("null")) return null;
        int start = position;
        take('-');
        if (!take('0')) digits();
        else if (position < input.length() && digit(input.charAt(position))) throw invalid();
        if (take('.')) digits();
        if (take('e') || take('E')) {
            if (!take('+')) take('-');
            digits();
        }
        if (position == start || position - start > 256) throw invalid();
        try { return new BigDecimal(input.substring(start, position)); }
        catch (NumberFormatException error) { throw invalid(); }
    }

    private String string() {
        if (!take('"')) throw invalid();
        StringBuilder text = new StringBuilder();
        while (position < input.length()) {
            char next = input.charAt(position++);
            if (next == '"') {
                unicode(text.toString());
                return text.toString();
            }
            if (next < 0x20) throw invalid();
            if (next == '\\') {
                if (position >= input.length()) throw invalid();
                char escaped = input.charAt(position++);
                switch (escaped) {
                    case '"': case '\\': case '/': text.append(escaped); break;
                    case 'b': text.append('\b'); break;
                    case 'f': text.append('\f'); break;
                    case 'n': text.append('\n'); break;
                    case 'r': text.append('\r'); break;
                    case 't': text.append('\t'); break;
                    case 'u':
                        if (position + 4 > input.length()) throw invalid();
                        int code = 0;
                        for (int i = 0; i < 4; i++) {
                            char hex = input.charAt(position++);
                            int digit = hex >= '0' && hex <= '9' ? hex - '0'
                                : hex >= 'a' && hex <= 'f' ? hex - 'a' + 10
                                : hex >= 'A' && hex <= 'F' ? hex - 'A' + 10 : -1;
                            if (digit < 0) throw invalid();
                            code = code * 16 + digit;
                        }
                        text.append((char) code);
                        break;
                    default: throw invalid();
                }
            } else text.append(next);
        }
        throw invalid();
    }

    static String quote(String value) {
        if (value == null) throw invalid();
        unicode(value);
        StringBuilder text = new StringBuilder("\"");
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"': text.append("\\\""); break;
                case '\\': text.append("\\\\"); break;
                case '\b': text.append("\\b"); break;
                case '\f': text.append("\\f"); break;
                case '\n': text.append("\\n"); break;
                case '\r': text.append("\\r"); break;
                case '\t': text.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        String digits = Integer.toHexString(c);
                        text.append("\\u").append("0000", 0, 4 - digits.length()).append(digits);
                    } else text.append(c);
            }
        }
        return text.append('"').toString();
    }

    static void unicode(String value) {
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (Character.isHighSurrogate(c)) {
                if (++i >= value.length() || !Character.isLowSurrogate(value.charAt(i))) throw invalid();
            } else if (Character.isLowSurrogate(c)) throw invalid();
        }
    }
    private void digits() {
        int start = position;
        while (position < input.length() && digit(input.charAt(position))) position++;
        if (position == start) throw invalid();
    }
    private static boolean digit(char c) { return c >= '0' && c <= '9'; }
    private void space() {
        while (position < input.length() && " \t\r\n".indexOf(input.charAt(position)) >= 0) position++;
    }
    private boolean take(char c) {
        if (position < input.length() && input.charAt(position) == c) { position++; return true; }
        return false;
    }
    private boolean literal(String value) {
        if (input.startsWith(value, position)) { position += value.length(); return true; }
        return false;
    }
    private static Protocol.ValidationException invalid() {
        return new Protocol.ValidationException("The model returned invalid JSON. The answer has been withheld.");
    }
}
