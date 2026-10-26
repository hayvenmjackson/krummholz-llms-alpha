package ai.krummholz.llm;

import android.Manifest;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.*;

/** Inspect the installed target APK rather than trusting the source manifest alone. */
@RunWith(AndroidJUnit4.class)
public final class ManifestPrivacyTest {
    @Test
    public void installedAlphaHasNoNetworkOrPersonalDataPermissionsAndNoBackup() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        PackageInfo info = context.getPackageManager().getPackageInfo(context.getPackageName(),
                PackageManager.GET_PERMISSIONS | PackageManager.GET_ACTIVITIES
                        | PackageManager.GET_SERVICES | PackageManager.GET_RECEIVERS
                        | PackageManager.GET_PROVIDERS);
        List<String> requested = info.requestedPermissions == null ? Collections.emptyList()
                : Arrays.asList(info.requestedPermissions);
        for (String permission : Arrays.asList(Manifest.permission.INTERNET,
                Manifest.permission.ACCESS_NETWORK_STATE, Manifest.permission.RECORD_AUDIO,
                Manifest.permission.CAMERA, Manifest.permission.READ_CONTACTS,
                Manifest.permission.READ_CALENDAR, Manifest.permission.READ_SMS,
                Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION,
                Manifest.permission.MANAGE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
            assertFalse("Unexpected permission in the installed local-only text alpha: " + permission,
                    requested.contains(permission));
        }
        assertNotNull(info.applicationInfo);
        assertEquals("Instrumentation must exercise the non-debuggable release target.", 0,
                info.applicationInfo.flags & ApplicationInfo.FLAG_DEBUGGABLE);
        assertEquals("Automatic backup must remain disabled.", 0,
                info.applicationInfo.flags & ApplicationInfo.FLAG_ALLOW_BACKUP);
        assertEquals("Cleartext traffic must remain disabled.", 0,
                info.applicationInfo.flags & ApplicationInfo.FLAG_USES_CLEARTEXT_TRAFFIC);
        if (info.activities != null) for (ActivityInfo activity : info.activities) {
            if (activity.exported) assertEquals("Only the launcher activity may be exported.",
                    "ai.krummholz.llm.MainActivity", activity.name);
        }
        if (info.services != null) for (android.content.pm.ServiceInfo service : info.services)
            assertFalse("Unexpected exported service.", service.exported);
        if (info.receivers != null) for (ActivityInfo receiver : info.receivers)
            assertFalse("Unexpected exported receiver.", receiver.exported);
        if (info.providers != null) for (android.content.pm.ProviderInfo provider : info.providers)
            assertFalse("Unexpected exported provider.", provider.exported);
    }
}
