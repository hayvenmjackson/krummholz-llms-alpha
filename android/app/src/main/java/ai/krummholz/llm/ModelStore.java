package ai.krummholz.llm;

import android.content.Context;
import android.net.Uri;
import org.json.JSONObject;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;

/** Only the model is retained, in app-private storage excluded from backups. */
public final class ModelStore {
    public static final class Spec {
        public final String name, filename, sha256, downloadUrl, license, upstream;
        public final long bytes;
        public final boolean approved;
        private Spec(JSONObject json) throws Exception {
            name = json.getString("name"); filename = json.getString("filename");
            sha256 = json.getString("sha256"); bytes = json.getLong("bytes");
            downloadUrl = json.getString("download_url"); license = json.getString("license");
            upstream = json.getString("upstream"); approved = json.optBoolean("approved", false);
            if (approved && (bytes <= 0 || bytes > 4L * 1024 * 1024 * 1024
                    || !sha256.matches("[0-9a-f]{64}") || !filename.matches("[A-Za-z0-9_.-]+\\.gguf")))
                throw new IOException("The approved model manifest is invalid.");
            if (!downloadUrl.isEmpty()) {
                Uri uri = Uri.parse(downloadUrl);
                if (!"https".equals(uri.getScheme()) || !"github.com".equals(uri.getHost())
                        || uri.getUserInfo() != null || !uri.getPath().contains("/releases/download/"))
                    throw new IOException("The model download must be an HTTPS GitHub release asset.");
            }
        }
    }
    public final Spec spec;
    private final Context context;
    private final File directory, model;
    public ModelStore(Context context) throws Exception {
        this.context = context.getApplicationContext();
        try (InputStream input = context.getAssets().open("model-manifest.json")) {
            ByteArrayOutputStream data = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024]; int read;
            while ((read = input.read(buffer)) >= 0) {
                if (data.size() + read > 32768) throw new IOException("The model manifest is too large.");
                data.write(buffer, 0, read);
            }
            spec = new Spec(new JSONObject(data.toString(StandardCharsets.UTF_8.name())));
        }
        directory = new File(context.getNoBackupFilesDir(), "models");
        if (!directory.isDirectory() && !directory.mkdirs()) throw new IOException("Private model storage is unavailable.");
        model = new File(directory, "approved-model.gguf");
    }
    public boolean hasModel() { return model.isFile(); }
    public File verifiedModel(BooleanSupplier cancelled) throws IOException {
        requireApproved();
        ModelVerifier.verify(model, spec.bytes, spec.sha256, cancelled);
        return model;
    }
    public File importModel(Uri uri, BooleanSupplier cancelled, ModelVerifier.Progress progress) throws IOException {
        requireApproved();
        if (directory.getUsableSpace() < spec.bytes + 32L * 1024 * 1024)
            throw new IOException("Free some storage before importing the model. The previous model is unchanged.");
        File stage = File.createTempFile("import-", ".partial", directory);
        try {
            try (InputStream input = context.getContentResolver().openInputStream(uri)) {
                if (input == null) throw new IOException("That file could not be opened. Try a locally downloaded copy.");
                ModelVerifier.copyVerified(input, stage, spec.bytes, spec.sha256, cancelled, progress);
            }
            if (cancelled.getAsBoolean()) throw new CancellationException();
            // Same-directory atomic replacement: a failed import never removes the valid model.
            Files.move(stage.toPath(), model.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            return model;
        } finally {
            // This is the uniquely named staging file created by this operation, never a user file.
            Files.deleteIfExists(stage.toPath());
        }
    }
    private void requireApproved() throws IOException {
        if (!spec.approved) throw new IOException("Model provisioning is not complete in this development build.");
    }
}
