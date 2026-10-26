package ai.krummholz.llm;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.CancellationSignal;
import android.os.ParcelFileDescriptor;
import android.os.SystemClock;
import android.provider.OpenableColumns;
import android.system.ErrnoException;
import android.system.Os;
import android.system.OsConstants;
import android.system.StructStat;
import org.json.JSONObject;
import org.json.JSONArray;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

/** Only the model is retained, in app-private storage excluded from backups. */
public final class ModelStore {
    private static final long PROVIDER_CALL_MILLIS = 30_000;
    private static final ScheduledExecutorService PROVIDER_TIMER = Executors.newSingleThreadScheduledExecutor(task -> {
        Thread thread = new Thread(task, "model-provider-deadline"); thread.setDaemon(true); return thread;
    });
    public static final class Spec {
        public final String name, filename, sha256, downloadUrl, license, upstream, protocol;
        public final long bytes, partBytes;
        public final String multipartManifestSha256;
        public final List<ModelVerifier.Part> parts;
        public final boolean approved;
        Spec(JSONObject json) throws Exception {
            name = json.getString("name"); filename = json.getString("filename");
            sha256 = json.getString("sha256"); bytes = exactLong(json, "bytes");
            downloadUrl = json.getString("download_url"); license = json.getString("license");
            upstream = json.getString("upstream"); approved = json.optBoolean("approved", false);
            protocol = json.optString("protocol", "");
            JSONObject multipart = json.optJSONObject("multipart");
            List<ModelVerifier.Part> parsed = new ArrayList<>();
            if (multipart == null) {
                if (json.has("multipart")) throw new IOException("The multipart manifest must be an object.");
                partBytes = 0; multipartManifestSha256 = "";
            } else {
                if (exactLong(multipart, "schema_version") != 1
                        || !"raw-concatenation-v1".equals(multipart.getString("format")))
                    throw new IOException("Unknown multipart model format.");
                partBytes = exactLong(multipart, "part_bytes");
                multipartManifestSha256 = multipart.getString("manifest_sha256");
                if (!multipartManifestSha256.matches("[0-9a-f]{64}"))
                    throw new IOException("The multipart manifest pin is invalid.");
                JSONArray list = multipart.getJSONArray("parts");
                if (list.length() == 0 || list.length() > 64) throw new IOException("Invalid multipart model count.");
                for (int i = 0; i < list.length(); i++) {
                    JSONObject part = list.getJSONObject(i);
                    long index = exactLong(part, "index");
                    if (index < 1 || index > 64) throw new IOException("Invalid model part index.");
                    parsed.add(new ModelVerifier.Part((int) index, part.getString("name"),
                        exactLong(part, "offset"), exactLong(part, "bytes"), part.getString("sha256")));
                }
                ModelVerifier.validateParts(filename, bytes, partBytes, parsed);
            }
            parts = Collections.unmodifiableList(parsed);
            if (approved && (bytes <= 0 || bytes > 8L * 1024 * 1024 * 1024
                    || !("pleias-rag-v1".equals(protocol) || "smollm3-chat-v1".equals(protocol))
                    || !sha256.matches("[0-9a-f]{64}") || !filename.matches("[A-Za-z0-9_.-]+\\.gguf")))
                throw new IOException("The approved model manifest is invalid.");
            if (!downloadUrl.isEmpty()) {
                Uri uri = Uri.parse(downloadUrl);
                if (!"https".equals(uri.getScheme()) || !"github.com".equals(uri.getHost())
                        || uri.getUserInfo() != null || uri.getPort() != -1 || uri.getQuery() != null || uri.getFragment() != null
                        || uri.getPath() == null || !uri.getPath().matches(
                            "/hayvenmjackson/krummholz-llms-alpha/releases/(?:tag/[A-Za-z0-9][A-Za-z0-9._-]{0,99}|download/[A-Za-z0-9][A-Za-z0-9._-]{0,99}/[A-Za-z0-9][A-Za-z0-9._-]{0,199})"))
                    throw new IOException("The model download must be the project's HTTPS GitHub release or release asset.");
            }
        }
        public boolean hasMultipart() { return !parts.isEmpty(); }
        private static long exactLong(JSONObject object, String key) throws Exception {
            Object value = object.get(key);
            if (!(value instanceof Long) && !(value instanceof Integer))
                throw new IOException("Manifest integer was represented ambiguously.");
            return ((Number) value).longValue();
        }
    }
    public final Spec spec;
    private final Context context;
    private final File directory, model;
    public ModelStore(Context context) throws Exception {
        this(context, readSpec(context));
    }
    // Package-private dependency injection for synthetic instrumentation. The production
    // constructor above obtains its immutable spec only from the signed APK asset.
    ModelStore(Context context, Spec spec) throws IOException {
        this.context = context.getApplicationContext();
        this.spec = spec;
        directory = new File(context.getNoBackupFilesDir(), "models");
        if (!directory.isDirectory() && !directory.mkdirs()) throw new IOException("Private model storage is unavailable.");
        model = new File(directory, "approved-model.gguf");
    }
    private static Spec readSpec(Context context) throws Exception {
        try (InputStream input = context.getAssets().open("model-manifest.json")) {
            ByteArrayOutputStream data = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024]; int read;
            while ((read = input.read(buffer)) >= 0) {
                if (data.size() + read > 32768) throw new IOException("The model manifest is too large.");
                data.write(buffer, 0, read);
            }
            return new Spec(new JSONObject(data.toString(StandardCharsets.UTF_8.name())));
        }
    }
    public boolean hasModel() { return model.isFile(); }
    public File verifiedModel(BooleanSupplier cancelled) throws IOException {
        requireApproved();
        ModelVerifier.verify(model, spec.bytes, spec.sha256, cancelled);
        return model;
    }
    public File importModel(Uri uri, BooleanSupplier cancelled, ModelVerifier.Progress progress) throws IOException {
        return importModel(uri, cancelled, progress, new CancellationSignal());
    }
    public File importModel(Uri uri, BooleanSupplier cancelled, ModelVerifier.Progress progress,
                            CancellationSignal signal) throws IOException {
        return importModels(Collections.singletonList(uri), cancelled, progress, signal);
    }
    public File importModels(List<Uri> uris, BooleanSupplier cancelled, ModelVerifier.Progress progress,
                             CancellationSignal signal) throws IOException {
        requireApproved();
        checkCancelled(cancelled, signal);
        if (uris == null || uris.isEmpty() || (uris.size() != 1 && uris.size() != spec.parts.size()) || uris.size() > 64)
            throw new IOException("Choose the complete model file, or all of its original downloaded parts together.");
        List<Uri> selected = new ArrayList<>(uris);
        Set<Uri> unique = new HashSet<>();
        for (Uri uri : selected) {
            if (uri == null || !"content".equals(uri.getScheme()) || !unique.add(uri))
                throw new IOException("Choose distinct local files using the document picker.");
        }
        if (directory.getUsableSpace() < spec.bytes + 32L * 1024 * 1024)
            throw new IOException("Free some storage before importing the model. The previous model is unchanged.");
        AtomicReference<ParcelFileDescriptor> active = new AtomicReference<>();
        signal.setOnCancelListener(() -> closeDescriptor(active.getAndSet(null)));
        File stage = null;
        try {
            final Uri[] ordered = selected.size() == 1 ? null : orderParts(selected, cancelled, signal);
            checkCancelled(cancelled, signal);
            stage = File.createTempFile("import-", ".partial", directory);
            BooleanSupplier stopped = () -> cancelled.getAsBoolean() || signal.isCanceled();
            if (ordered == null) {
                try (InputStream input = openLocal(selected.get(0), spec.bytes, stopped, signal, active)) {
                    ModelVerifier.copyVerified(input, stage, spec.bytes, spec.sha256, stopped, progress);
                }
            } else {
                ModelVerifier.copyVerifiedParts(part -> openLocal(ordered[part.index - 1], part.bytes, stopped, signal, active),
                    stage, spec.bytes, spec.sha256, spec.filename, spec.partBytes, spec.parts, stopped, progress);
            }
            checkCancelled(cancelled, signal);
            // Same-directory atomic replacement: a failed import never removes the valid model.
            Files.move(stage.toPath(), model.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            return model;
        } finally {
            signal.setOnCancelListener(null);
            closeDescriptor(active.getAndSet(null));
            // This is the uniquely named staging file created by this operation, never a user file.
            if (stage != null) Files.deleteIfExists(stage.toPath());
        }
    }
    private Uri[] orderParts(List<Uri> selected, BooleanSupplier cancelled, CancellationSignal signal) throws IOException {
        if (!spec.hasMultipart() || selected.size() != spec.parts.size())
            throw new IOException("This build does not have verified pins for that set of model parts.");
        Uri[] ordered = new Uri[spec.parts.size()];
        boolean[] seen = new boolean[ordered.length];
        for (Uri uri : selected) {
            checkCancelled(cancelled, signal);
            try (ProviderDeadline deadline = new ProviderDeadline(signal, cancelled);
                 Cursor cursor = context.getContentResolver().query(uri,
                     new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE}, null, null, null, signal)) {
                deadline.check();
                if (cursor == null || !cursor.moveToFirst()) throw new IOException("A selected model part has no readable metadata.");
                int nameColumn = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME), sizeColumn = cursor.getColumnIndex(OpenableColumns.SIZE);
                if (nameColumn < 0 || sizeColumn < 0 || cursor.isNull(nameColumn) || cursor.isNull(sizeColumn))
                    throw new IOException("Download each model part completely to local storage before importing.");
                String name = cursor.getString(nameColumn);
                long bytes = cursor.getLong(sizeColumn);
                if (cursor.moveToNext()) throw new IOException("The document provider returned ambiguous part metadata.");
                deadline.check();
                int index = ModelVerifier.matchSelectedPart(name, bytes, spec.parts, seen);
                ordered[index] = uri;
            }
        }
        for (Uri uri : ordered) if (uri == null) throw new IOException("One or more model parts are missing.");
        return ordered;
    }
    private InputStream openLocal(Uri uri, long expectedBytes, BooleanSupplier cancelled, CancellationSignal signal,
                                  AtomicReference<ParcelFileDescriptor> active) throws IOException {
        checkCancelled(cancelled, signal);
        ParcelFileDescriptor descriptor = null;
        try (ProviderDeadline deadline = new ProviderDeadline(signal, cancelled)) {
            descriptor = context.getContentResolver().openFileDescriptor(uri, "r", signal);
            if (descriptor == null) throw new IOException("That file could not be opened. Try a locally downloaded copy.");
            active.set(descriptor);
            deadline.check();
            StructStat stat = Os.fstat(descriptor.getFileDescriptor());
            if (!OsConstants.S_ISREG(stat.st_mode))
                throw new IOException("Import fully downloaded local files. Cloud streams and pipes are not supported.");
            if (stat.st_size != expectedBytes)
                throw new IOException("A downloaded model file is incomplete or has the wrong size.");
            deadline.check();
            return new ParcelFileDescriptor.AutoCloseInputStream(descriptor);
        } catch (ErrnoException error) {
            closeDescriptor(descriptor); active.compareAndSet(descriptor, null);
            throw new IOException("The downloaded file could not be read. Choose a local copy from Downloads.", error);
        } catch (IOException | RuntimeException error) {
            closeDescriptor(descriptor); active.compareAndSet(descriptor, null);
            throw error;
        }
    }
    private static void checkCancelled(BooleanSupplier cancelled, CancellationSignal signal) {
        if (cancelled.getAsBoolean() || signal.isCanceled() || Thread.currentThread().isInterrupted())
            throw new CancellationException();
    }
    private static final class ProviderDeadline implements AutoCloseable {
        private final CancellationSignal signal;
        private final BooleanSupplier cancelled;
        private final long expires = SystemClock.elapsedRealtime() + PROVIDER_CALL_MILLIS;
        private final ScheduledFuture<?> timeout;
        ProviderDeadline(CancellationSignal signal, BooleanSupplier cancelled) {
            this.signal = signal; this.cancelled = cancelled;
            timeout = PROVIDER_TIMER.schedule(signal::cancel, PROVIDER_CALL_MILLIS, TimeUnit.MILLISECONDS);
        }
        void check() throws IOException {
            if (SystemClock.elapsedRealtime() >= expires) throw new IOException("The local file provider took too long. Try files saved in Downloads.");
            checkCancelled(cancelled, signal);
        }
        @Override public void close() { timeout.cancel(false); }
    }
    private static void closeDescriptor(ParcelFileDescriptor descriptor) {
        if (descriptor == null) return;
        try { descriptor.close(); } catch (IOException ignored) { /* Already closed or cancelled. */ }
    }
    private void requireApproved() throws IOException {
        if (!spec.approved) throw new IOException("Model provisioning is not complete in this development build.");
    }
}
