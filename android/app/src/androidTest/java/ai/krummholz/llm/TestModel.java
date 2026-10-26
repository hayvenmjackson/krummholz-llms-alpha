package ai.krummholz.llm;

import android.content.Context;
import org.json.JSONObject;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import static org.junit.Assert.*;

/** CI stages and verifies weights separately; neither APK embeds a multi-GB model. */
final class TestModel {
    private TestModel() { }

    static JSONObject manifest(Context context) throws Exception {
        try (InputStream input = context.getAssets().open("model-manifest.json");
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
                assertTrue("The embedded manifest is unexpectedly large.", output.size() <= 128 * 1024);
            }
            JSONObject manifest = new JSONObject(new String(output.toByteArray(), StandardCharsets.UTF_8));
            assertTrue("CI must embed its verified model manifest before building the APK.", manifest.getBoolean("approved"));
            assertTrue("The approved model must have a positive byte count.", manifest.getLong("bytes") > 0);
            assertTrue("The approved model must have an exact SHA-256.", manifest.getString("sha256").matches("[0-9a-f]{64}"));
            return manifest;
        }
    }

    static File provision(Context context, JSONObject manifest) throws Exception {
        File directory = context.getExternalFilesDir(null);
        assertNotNull("The emulator's app-specific staging directory is unavailable.", directory);
        File model = new File(directory, "test-model.gguf");
        assertFalse("The test model must be a staged file, not a symlink.", Files.isSymbolicLink(model.toPath()));
        assertEquals("Test weights must stay in the app-specific staging directory.",
                directory.getCanonicalFile(), model.getCanonicalFile().getParentFile());
        assertTrue("CI must push the separately verified test model before instrumentation.", model.isFile());
        // Only the instrumentation APK reads this staging path. Production code continues
        // to accept a user-selected local document through its verified import path.
        ModelVerifier.verify(model, manifest.getLong("bytes"), manifest.getString("sha256"), () -> false);
        return model;
    }
}
