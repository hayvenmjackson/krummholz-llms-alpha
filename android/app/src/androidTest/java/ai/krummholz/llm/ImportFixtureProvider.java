package ai.krummholz.llm;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.Binder;
import android.os.Bundle;
import android.os.CancellationSignal;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import android.system.Os;
import android.system.OsConstants;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Test APK only. Fixed synthetic bytes, or an explicitly registered read-only model descriptor. */
public final class ImportFixtureProvider extends ContentProvider {
    private final AtomicInteger blockedOpens = new AtomicInteger();
    private final AtomicInteger blockedQueries = new AtomicInteger();
    private volatile String mode = "normal";
    private ParcelFileDescriptor model;
    private static final byte[][] PARTS = {{'a','b'}, {'c','d'}, {'e','f'}, {'g'}};

    static Uri registerModel(Context context, File verifiedTestModel) throws IOException {
        Uri root = Uri.parse("content://ai.krummholz.llm.test.import-fixtures");
        try (ParcelFileDescriptor input = ParcelFileDescriptor.open(verifiedTestModel, ParcelFileDescriptor.MODE_READ_ONLY)) {
            Bundle extras = new Bundle(); extras.putParcelable("descriptor", input);
            context.getContentResolver().call(root, "register-model", null, extras);
        }
        return Uri.withAppendedPath(root, "whole");
    }

    @Override public boolean onCreate() { return true; }

    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        return openFile(uri, mode, null);
    }

    @Override public ParcelFileDescriptor openFile(Uri uri, String mode, CancellationSignal signal)
            throws FileNotFoundException {
        requireCaller();
        if (!"r".equals(mode)) throw new FileNotFoundException("Fixture is read-only.");
        if ("/whole".equals(uri.getPath())) {
            synchronized (this) {
                if (model == null) throw new FileNotFoundException("No verified model descriptor was registered.");
                try {
                    // Actual-model tests import serially. Reset the shared read-only description before each duplicate.
                    Os.lseek(model.getFileDescriptor(), 0, OsConstants.SEEK_SET);
                    return ParcelFileDescriptor.dup(model.getFileDescriptor());
                } catch (IOException | android.system.ErrnoException error) {
                    throw new FileNotFoundException("Registered model descriptor is unavailable.");
                }
            }
        }
        int part = partIndex(uri);
        if ("/pipe".equals(uri.getPath()) || (part == 1 && "pipe-part".equals(this.mode))) {
            try {
                ParcelFileDescriptor[] pipe = ParcelFileDescriptor.createPipe();
                pipe[1].close();
                return pipe[0];
            } catch (IOException error) {
                throw new FileNotFoundException("The synthetic pipe could not be created.");
            }
        }
        if (part >= 0) {
            byte[] bytes = PARTS[part];
            if (part == 1 && "corrupt".equals(this.mode)) bytes = new byte[]{'x','x'};
            if (part == 1 && "truncated".equals(this.mode)) bytes = new byte[]{'c'};
            File file = new File(getContext().getCacheDir(), "synthetic-part-" + part);
            try (FileOutputStream output = new FileOutputStream(file)) { output.write(bytes); }
            catch (IOException error) { throw new FileNotFoundException("Synthetic part could not be written."); }
            return ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY);
        }
        if ("/blocked-open".equals(uri.getPath())) {
            if (signal == null) throw new FileNotFoundException("Cancellation transport was not supplied.");
            CountDownLatch cancellation = new CountDownLatch(1);
            signal.setOnCancelListener(cancellation::countDown);
            blockedOpens.incrementAndGet();
            try {
                cancellation.await(20, TimeUnit.SECONDS);
                signal.throwIfCanceled();
                throw new FileNotFoundException("The test open reached its bounded deadline.");
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new FileNotFoundException("The test open was interrupted.");
            } finally {
                signal.setOnCancelListener(null);
            }
        }
        throw new FileNotFoundException("Unknown synthetic fixture.");
    }

    @Override public Bundle call(String method, String arg, Bundle extras) {
        requireCaller();
        if ("reset".equals(method)) { blockedOpens.set(0); blockedQueries.set(0); mode = "normal"; }
        else if ("mode".equals(method)) {
            if (!java.util.Arrays.asList("normal", "corrupt", "truncated", "wrong-metadata", "pipe-part", "blocked-query").contains(arg))
                throw new IllegalArgumentException("Unknown synthetic mode.");
            mode = arg;
        } else if ("register-model".equals(method)) {
            if (extras == null) throw new IllegalArgumentException("Descriptor required.");
            ParcelFileDescriptor supplied = extras.getParcelable("descriptor");
            if (supplied == null) throw new IllegalArgumentException("Descriptor required.");
            synchronized (this) {
                try {
                    if (model != null) model.close();
                    model = ParcelFileDescriptor.dup(supplied.getFileDescriptor());
                    supplied.close();
                } catch (IOException error) { throw new IllegalStateException(error); }
            }
        } else if ("clear-model".equals(method)) {
            synchronized (this) {
                try { if (model != null) model.close(); }
                catch (IOException error) { throw new IllegalStateException(error); }
                finally { model = null; }
            }
        }
        else if (!"status".equals(method)) throw new IllegalArgumentException("Unknown fixture method.");
        Bundle status = new Bundle();
        status.putInt("blocked_opens", blockedOpens.get());
        status.putInt("blocked_queries", blockedQueries.get());
        return status;
    }

    @Override public String getType(Uri uri) { return "application/octet-stream"; }
    @Override public Cursor query(Uri uri, String[] projection, String selection, String[] args, String order) {
        return query(uri, projection, selection, args, order, null);
    }
    @Override public Cursor query(Uri uri, String[] projection, String selection, String[] args, String order, CancellationSignal signal) {
        requireCaller();
        if ("blocked-query".equals(mode)) {
            if (signal == null) throw new IllegalStateException("Query cancellation transport missing.");
            CountDownLatch cancellation = new CountDownLatch(1);
            signal.setOnCancelListener(cancellation::countDown); blockedQueries.incrementAndGet();
            try { cancellation.await(20, TimeUnit.SECONDS); signal.throwIfCanceled(); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IllegalStateException(error); }
            finally { signal.setOnCancelListener(null); }
        }
        int part = partIndex(uri);
        String name = part < 0 ? "unrelated.bin" : String.format(java.util.Locale.ROOT, "synthetic.gguf.part-%05d-of-00004", part + 1);
        long size = part < 0 ? 2 : PARTS[part].length;
        if ("wrong-metadata".equals(mode) && part == 1) size++;
        MatrixCursor cursor = new MatrixCursor(new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE});
        cursor.addRow(new Object[]{name, size});
        return cursor;
    }
    private static int partIndex(Uri uri) {
        String path = uri.getPath();
        if (path != null && path.matches("/(?:part|alias)/[1-4]")) return path.charAt(path.length() - 1) - '1';
        return -1;
    }
    private void requireCaller() {
        try {
            if (Binder.getCallingUid() != getContext().getPackageManager().getPackageUid("ai.krummholz.llm", 0))
                throw new SecurityException("Only the instrumented target app may access test fixtures.");
        } catch (android.content.pm.PackageManager.NameNotFoundException error) { throw new SecurityException(error); }
    }
    @Override public Uri insert(Uri uri, ContentValues values) { throw new UnsupportedOperationException(); }
    @Override public int update(Uri uri, ContentValues values, String selection, String[] args) { throw new UnsupportedOperationException(); }
    @Override public int delete(Uri uri, String selection, String[] args) { throw new UnsupportedOperationException(); }
}
