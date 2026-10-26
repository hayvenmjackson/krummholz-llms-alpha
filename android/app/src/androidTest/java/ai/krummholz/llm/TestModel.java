package ai.krummholz.llm;

import android.content.Context;
import android.os.Bundle;
import androidx.test.platform.app.InstrumentationRegistry;
import org.json.JSONObject;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import static org.junit.Assert.*;

/** CI weights live in the separate test APK, never in a production import backdoor. */
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
        Bundle arguments = InstrumentationRegistry.getArguments();
        String suppliedPath = arguments.getString("modelPath");
        File model = suppliedPath == null ? new File(context.getFilesDir(), "test-model.gguf") : new File(suppliedPath);
        String privateRoot = context.getFilesDir().getCanonicalPath() + File.separator;
        assertTrue("Test weights must stay in this app's private files directory.", model.getCanonicalPath().startsWith(privateRoot));
        if (!model.exists()) {
            File stage = File.createTempFile("test-model-import-", ".partial", model.getParentFile());
            Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
            try (InputStream input = testContext.getAssets().open("test-model.gguf")) {
                ModelVerifier.copyVerified(input, stage, manifest.getLong("bytes"), manifest.getString("sha256"),
                        () -> false, bytes -> { });
                Files.move(stage.toPath(), model.toPath(), StandardCopyOption.ATOMIC_MOVE);
            } finally {
                Files.deleteIfExists(stage.toPath());
            }
        }
        assertTrue("CI did not provision the actual test model.", model.isFile());
        ModelVerifier.verify(model, manifest.getLong("bytes"), manifest.getString("sha256"), () -> false);
        return model;
    }
}
