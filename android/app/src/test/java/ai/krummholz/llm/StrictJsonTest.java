package ai.krummholz.llm;

import java.math.BigDecimal;
import java.util.Map;
import org.junit.Test;
import static org.junit.Assert.*;

public class StrictJsonTest {
    private void reject(String raw) { assertThrows(Protocol.ValidationException.class, () -> StrictJson.parse(raw)); }

    @Test public void validJsonAndUnicodeRoundTripWithoutCoercion() {
        Map<?, ?> object = (Map<?, ?>) StrictJson.parse("{\"text\":\"Café 🌲\\n\\t\\\"\\\\\",\"flag\":true,\"n\":1.25e2,\"none\":null}");
        assertEquals("Café 🌲\n\t\"\\", object.get("text"));
        assertEquals(Boolean.TRUE, object.get("flag"));
        assertEquals(new BigDecimal("1.25e2"), object.get("n"));
        assertTrue(object.containsKey("none"));
        String text = "Café 🌲\n\t\"\\\u2028";
        assertEquals(text, StrictJson.parse(StrictJson.quote(text)));
        assertEquals("🌲", StrictJson.parse("\"\\ud83c\\udf32\""));
    }
    @Test public void duplicateKeysAreRejectedAtEveryDepthIncludingNull() {
        reject("{\"x\":null,\"x\":1}");
        reject("{\"arguments\":{\"name\":\"answer\",\"name\":\"send_email\"}}");
        reject("{\"a\":1,\"\\u0061\":2}");
    }
    @Test public void malformedAndNonJsonNumbersAreRejected() {
        for (String raw : new String[]{"NaN", "Infinity", "-Infinity", "+1", "01", "-", "1.", "1e", "1e+", "1e999999999999"})
            reject(raw);
    }
    @Test public void malformedDelimitersCommentsAndTrailingContentAreRejected() {
        for (String raw : new String[]{"", "{", "[", "{\"a\":1,}", "[1,]", "{a:1}", "{'a':1}", "{}{}", "{} trailing", "//x\n{}", "\ufeff{}"})
            reject(raw);
    }
    @Test public void invalidEscapesRawControlsAndLoneSurrogatesAreRejected() {
        for (String raw : new String[]{"\"\\x20\"", "\"\\u12xz\"", "\"\\ud800\"", "\"\\udc00\"", "\"\ud800\"", "\"line\nbreak\"", "\"trailing\\"})
            reject(raw);
        assertThrows(Protocol.ValidationException.class, () -> StrictJson.quote("\ud800"));
    }
    @Test public void depthLengthAndCollectionBoundsPreventUnboundedParsing() {
        reject("[".repeat(18) + "0" + "]".repeat(18));
        reject("[" + "0,".repeat(256) + "0]");
        reject("\"" + "a".repeat(65536) + "\"");
    }
    @Test public void errorMessagesDoNotEchoRawPrivateOrReasoningText() {
        Protocol.ValidationException error = assertThrows(Protocol.ValidationException.class,
            () -> StrictJson.parse("PRIVATE REASONING"));
        assertFalse(error.getMessage().contains("PRIVATE"));
    }
}
