package ai.krummholz.llm;

import android.app.Instrumentation;
import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;
import java.io.File;
import java.io.FileOutputStream;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import static org.junit.Assert.*;

/** Real release Activity, real model import core and actual JNI answer; all input is fictional. */
@RunWith(AndroidJUnit4.class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public final class WorkspaceUiTest {
    private static final String EMPTY_ANSWER = "Your checked answer will appear here.";

    @Test(timeout = 4 * 60 * 1000L)
    public void a_workspaceDoesNotRestorePrivateTextAndProfilesResetInputs() throws Exception {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            await(scenario, activity -> activity.findViewById(R.id.cancel_button).getVisibility() != View.VISIBLE,
                    "Initial setup did not settle.");
            scenario.onActivity(activity -> {
                EditText source = activity.findViewById(R.id.source_input);
                EditText question = activity.findViewById(R.id.question_input);
                for (EditText input : new EditText[]{source, question}) {
                    assertFalse("User text must not enter saved view state.", input.isSaveEnabled());
                    assertFalse("Parent views must not save user text.", input.isSaveFromParentEnabled());
                    assertEquals(View.IMPORTANT_FOR_AUTOFILL_NO, input.getImportantForAutofill());
                    assertNotEquals("The app must request no personalized keyboard learning.", 0,
                            input.getImeOptions() & EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING);
                }
                source.setText("Fictional recreation probe: cedar-4821.");
                question.setText("Which fictional probe was typed?");
            });
            scenario.recreate();
            await(scenario, activity -> activity.findViewById(R.id.cancel_button).getVisibility() != View.VISIBLE,
                    "Recreated setup did not settle.");
            scenario.onActivity(activity -> {
                assertFalse(text(activity, R.id.source_input).contains("cedar-4821"));
                assertFalse(text(activity, R.id.question_input).contains("probe"));
                assertTrue(text(activity, R.id.source_input).contains("Fictional demonstration policy"));
                assertEquals(EMPTY_ANSWER, text(activity, R.id.answer_text));
                ((Spinner) activity.findViewById(R.id.profile_picker)).setSelection(1);
            });
            await(scenario, activity -> text(activity, R.id.source_input).contains("90 surviving seedlings"),
                    "Katahdin did not replace the previous profile's source.");
            scenario.onActivity(activity -> {
                assertTrue(text(activity, R.id.question_input).contains("survival percentage"));
                ((Button) activity.findViewById(R.id.clear_button)).performClick();
                assertEquals("", text(activity, R.id.source_input));
                assertEquals("", text(activity, R.id.question_input));
                assertEquals(EMPTY_ANSWER, text(activity, R.id.answer_text));
            });
        }
    }

    @Test(timeout = 6 * 60 * 1000L)
    public void b_approvedImportLoadsAndActualAcadiaAnswerIsDisplayed() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        File weights = TestModel.provision(context, TestModel.manifest(context));
        ModelStore store = new ModelStore(context);
        // The test-APK provider receives only this explicitly registered read-only model descriptor.
        // Production pin checks and the real content-provider import path stay in force.
        store.importModel(ImportFixtureProvider.registerModel(context, weights), () -> false, bytes -> { });
        store.verifiedModel(() -> false);
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            await(scenario, activity -> activity.findViewById(R.id.ask_button).isEnabled(),
                    "The real imported model did not become ready.");
            scenario.onActivity(activity -> {
                assertTrue(text(activity, R.id.model_status).contains("Verified and ready"));
                assertEquals(0, ((Spinner) activity.findViewById(R.id.profile_picker)).getSelectedItemPosition());
            });
            screenshot("01-local-model-ready.png");
            scenario.onActivity(activity -> {
                assertTrue(((Button) activity.findViewById(R.id.ask_button)).performClick());
                assertFalse(activity.findViewById(R.id.profile_picker).isEnabled());
                assertEquals(View.VISIBLE, activity.findViewById(R.id.cancel_button).getVisibility());
            });
            await(scenario, activity -> activity.findViewById(R.id.cancel_button).getVisibility() != View.VISIBLE,
                    "The UI request did not finish or cancel at its deadline.");
            scenario.onActivity(activity -> {
                assertEquals("Answer ready for your review.", text(activity, R.id.operation_status));
                String answer = text(activity, R.id.answer_text);
                assertNotEquals("The UI returned no actual answer.", EMPTY_ANSWER, answer);
                assertFalse("Raw protocol must not be shown.", answer.contains("<|"));
                String displayed = allText(activity.findViewById(android.R.id.content));
                assertTrue("The actual response must display its workspace citation.", displayed.contains("acadia/workspace"));
                ViewGroup root = activity.findViewById(android.R.id.content);
                ScrollView scroll = (ScrollView) root.getChildAt(0);
                scroll.fullScroll(View.FOCUS_DOWN);
            });
            screenshot("02-actual-acadia-answer.png");
            scenario.onActivity(activity -> ((Spinner) activity.findViewById(R.id.profile_picker)).setSelection(2));
            await(scenario, activity -> text(activity, R.id.source_input).contains("Paper Lantern"),
                    "Atlantic did not replace the previous profile's source.");
            scenario.onActivity(activity -> {
                assertEquals("Profile changes must clear the previous real answer.", EMPTY_ANSWER, text(activity, R.id.answer_text));
                assertFalse(allText(activity.findViewById(android.R.id.content)).contains("acadia/workspace"));
                ((Button) activity.findViewById(R.id.clear_button)).performClick();
                assertEquals("", text(activity, R.id.source_input));
                assertEquals("", text(activity, R.id.question_input));
            });
        }
    }

    private static String text(MainActivity activity, int id) {
        return ((TextView) activity.findViewById(id)).getText().toString();
    }

    private static String allText(View view) {
        StringBuilder text = new StringBuilder();
        if (view instanceof TextView) text.append(((TextView) view).getText()).append('\n');
        if (view instanceof ViewGroup) {
            ViewGroup group = (ViewGroup) view;
            for (int index = 0; index < group.getChildCount(); index++) text.append(allText(group.getChildAt(index)));
        }
        return text.toString();
    }

    private static void await(ActivityScenario<MainActivity> scenario, Predicate<MainActivity> predicate,
                              String failure) {
        long until = SystemClock.elapsedRealtime() + 185_000;
        AtomicBoolean ready = new AtomicBoolean();
        do {
            scenario.onActivity(activity -> ready.set(predicate.test(activity)));
            if (ready.get()) return;
            SystemClock.sleep(100);
        } while (SystemClock.elapsedRealtime() < until);
        fail(failure);
    }

    private static void screenshot(String name) throws Exception {
        Instrumentation instrumentation = InstrumentationRegistry.getInstrumentation();
        instrumentation.waitForIdleSync();
        Bitmap image = instrumentation.getUiAutomation().takeScreenshot();
        assertNotNull("The emulator did not provide a real screen image.", image);
        File directory = instrumentation.getTargetContext().getExternalFilesDir("test-evidence");
        assertNotNull("Screenshot evidence directory is unavailable.", directory);
        File destination = new File(directory, name);
        try (FileOutputStream output = new FileOutputStream(destination)) {
            assertTrue(image.compress(Bitmap.CompressFormat.PNG, 100, output));
        } finally {
            image.recycle();
        }
        Bundle status = new Bundle();
        status.putString("stream", "SCREENSHOT_PATH=" + destination.getAbsolutePath() + "\n");
        instrumentation.sendStatus(0, status);
    }
}
