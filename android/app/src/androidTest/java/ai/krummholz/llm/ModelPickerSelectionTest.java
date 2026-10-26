package ai.krummholz.llm;

import android.content.ClipData;
import android.content.Intent;
import android.net.Uri;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

/** Real Android picker-data shapes only; no files, models, network or native inference. */
@RunWith(AndroidJUnit4.class)
public class ModelPickerSelectionTest {
    private static Uri uri(int number) { return Uri.parse("content://fictional.picker/model-" + number); }
    private static Intent intent(ClipData files) {
        Intent intent = new Intent();
        intent.setClipData(files);
        return intent;
    }

    @Test public void singleFileSelectionRemainsSupportedAndImmutable() {
        List<Uri> selected = MainActivity.modelUris(new Intent().setData(uri(1)));
        assertEquals(Collections.singletonList(uri(1)), selected);
        assertThrows(UnsupportedOperationException.class, () -> selected.add(uri(2)));
    }
    @Test public void multipleFilesAreCopiedInPickerOrderWithoutTrustingNames() {
        ClipData files = new ClipData("fictional", new String[]{"application/octet-stream"}, new ClipData.Item(uri(3)));
        files.addItem(new ClipData.Item(uri(1)));
        List<Uri> selected = MainActivity.modelUris(intent(files));
        files.addItem(new ClipData.Item(uri(2)));
        assertEquals(Arrays.asList(uri(3), uri(1)), selected);
        assertThrows(UnsupportedOperationException.class, selected::clear);
    }
    @Test public void emptyAndTextOnlySelectionsAreRejected() {
        assertThrows(Protocol.ValidationException.class, () -> MainActivity.modelUris(null));
        assertThrows(Protocol.ValidationException.class, () -> MainActivity.modelUris(new Intent()));
        ClipData text = ClipData.newPlainText("not a file", "fictional text");
        assertThrows(Protocol.ValidationException.class, () -> MainActivity.modelUris(intent(text)));
    }
    @Test public void oversizedPickerListIsRejectedBeforeAnyImport() {
        ClipData files = new ClipData("fictional", new String[]{"application/octet-stream"}, new ClipData.Item(uri(0)));
        for (int index = 1; index <= 32; index++) files.addItem(new ClipData.Item(uri(index)));
        assertThrows(Protocol.ValidationException.class, () -> MainActivity.modelUris(intent(files)));
    }
}
