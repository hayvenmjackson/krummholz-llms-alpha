package ai.krummholz.llm;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.CancellationSignal;
import android.os.ParcelFileDescriptor;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Separate test-APK provider: no access to target-app files and no arbitrary paths. */
public final class ImportFixtureProvider extends ContentProvider {
    private final AtomicInteger blockedOpens = new AtomicInteger();

    @Override public boolean onCreate() { return true; }

    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        return openFile(uri, mode, null);
    }

    @Override public ParcelFileDescriptor openFile(Uri uri, String mode, CancellationSignal signal)
            throws FileNotFoundException {
        if (!"r".equals(mode)) throw new FileNotFoundException("Fixture is read-only.");
        if ("/pipe".equals(uri.getPath())) {
            try {
                ParcelFileDescriptor[] pipe = ParcelFileDescriptor.createPipe();
                pipe[1].close();
                return pipe[0];
            } catch (IOException error) {
                throw new FileNotFoundException("The synthetic pipe could not be created.");
            }
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
        if ("reset".equals(method)) blockedOpens.set(0);
        else if (!"status".equals(method)) throw new IllegalArgumentException("Unknown fixture method.");
        Bundle status = new Bundle();
        status.putInt("blocked_opens", blockedOpens.get());
        return status;
    }

    @Override public String getType(Uri uri) { return "application/octet-stream"; }
    @Override public Cursor query(Uri uri, String[] projection, String selection, String[] args, String order) {
        throw new UnsupportedOperationException("No queryable data.");
    }
    @Override public Uri insert(Uri uri, ContentValues values) { throw new UnsupportedOperationException(); }
    @Override public int update(Uri uri, ContentValues values, String selection, String[] args) { throw new UnsupportedOperationException(); }
    @Override public int delete(Uri uri, String selection, String[] args) { throw new UnsupportedOperationException(); }
}
