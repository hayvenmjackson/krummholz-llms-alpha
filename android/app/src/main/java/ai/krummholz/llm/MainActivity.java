package ai.krummholz.llm;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.CancellationSignal;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/** Native Android screen. Prompt and response state live only in this Activity. */
public final class MainActivity extends Activity {
    private static final int FOREST = 0xff10251f, CARD = 0xff1b352b, CREAM = 0xfff3eedd;
    private static final int MUTED = 0xffb4c5ba, LIME = 0xffc6d89b, LINE = 0xff3b5849;
    private static final int IMPORT_MODEL = 71, CONTEXT = 4096, OUTPUT = 1024;
    private static final String[] KEYS = {"acadia", "katahdin", "atlantic"};
    private static final String[] NAMES = {"Acadia", "Katahdin", "Atlantic"};
    private static final String[] PURPOSES = {
        "Customer-service policies and evidence for a human handoff.",
        "Forestry and fisheries source questions. Numerical facts come from the source.",
        "Questions about local documents. No autonomous actions."
    };
    private static final String[] SOURCES = {
        "Fictional demonstration policy: unused items may be returned within 30 days. Refunds above 100 demo credits require a human reviewer.",
        "Fictional training plot: a validated calculator reports 90 surviving seedlings out of 100 initially planted, equal to 90% survival. This is invented demonstration data, not a field observation.",
        "Fictional local note: the demonstration project's codename is Paper Lantern. No messages, purchases or file changes have been authorized."
    };
    private static final String[] QUESTIONS = {
        "How long is the return window?", "What survival percentage did the validated calculator report?", "What is the project codename?"
    };
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final ScheduledExecutorService timer = Executors.newSingleThreadScheduledExecutor();
    private ModelStore store;
    private NativeEngine engine;
    private volatile boolean destroyed;
    private boolean loaded;
    private int selectedProfile;
    private Job current;
    private Spinner profiles;
    private EditText sourceInput, questionInput;
    private TextView modelStatus, operationStatus, purpose, answerText, answerNote;
    private LinearLayout citations;
    private Button downloadButton, importButton, reloadButton, askButton, cancelButton, clearButton, exampleButton;
    private ProgressBar progress;
    private AlertDialog aboutDialog;

    private static final class Job {
        final AtomicBoolean cancelled = new AtomicBoolean();
        final CancellationSignal importCancellation = new CancellationSignal();
        volatile boolean timedOut;
        ScheduledFuture<?> deadline;
    }

    @Override public void onCreate(Bundle state) {
        super.onCreate(null);
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);
        buildScreen();
        try {
            store = new ModelStore(this);
            engine = new NativeEngine();
            modelStatus.setText(store.spec.name + "\n" + (store.spec.approved
                ? String.format(Locale.ROOT, "%.0f MB download · %s\nOne shared baseline for all three profiles.", store.spec.bytes / 1_000_000.0, store.spec.license)
                : "Model provisioning is not complete in this development build."));
            controls();
            if (store.spec.approved && store.hasModel()) loadExisting();
        } catch (Exception | LinkageError error) {
            modelStatus.setText("The local engine could not start. Reinstall the official alpha for this device.");
            operationStatus.setText("Setup unavailable");
            controls();
        }
    }

    private void buildScreen() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true); scroll.setBackgroundColor(FOREST);
        scroll.setSaveEnabled(false);
        scroll.setImportantForAutofill(View.IMPORTANT_FOR_AUTOFILL_NO_EXCLUDE_DESCENDANTS);
        scroll.setOnApplyWindowInsetsListener((view, insets) -> {
            view.setPadding(insets.getSystemWindowInsetLeft(), insets.getSystemWindowInsetTop(),
                insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom());
            return insets;
        });
        LinearLayout body = column(); body.setPadding(dp(22), dp(24), dp(22), dp(30));
        scroll.addView(body); setContentView(scroll);
        body.addView(text("KRUMMHOLZ  /  RESEARCH", 12, LIME, true));
        TextView title = text("Krummholz LLMs", 30, CREAM, true); margin(body, title, 8, 4);
        margin(body, text("ALPHA  ·  0.1.0-alpha.1", 11, LIME, true), 0, 16);
        margin(body, text("Ask from a source.\nKeep the work on your phone.", 20, CREAM, false), 0, 12);
        margin(body, text("An early local Q&A workspace. Start with the fictional examples below. Answers need your review.", 14, MUTED, false), 0, 24);

        LinearLayout modelCard = card(body);
        modelCard.addView(text("01  /  YOUR LOCAL MODEL", 11, LIME, true));
        modelStatus = text("Preparing local setup…", 15, CREAM, false); modelStatus.setId(R.id.model_status);
        margin(modelCard, modelStatus, 12, 14);
        downloadButton = button("Get model from GitHub", false); downloadButton.setId(R.id.download_button);
        downloadButton.setOnClickListener(v -> openModelDownload()); modelCard.addView(downloadButton);
        importButton = button("Import downloaded model", true); importButton.setId(R.id.import_button);
        importButton.setOnClickListener(v -> chooseModel()); margin(modelCard, importButton, 8, 10);
        reloadButton = button("Load saved model", false);
        reloadButton.setOnClickListener(v -> loadExisting()); modelCard.addView(reloadButton);
        modelCard.addView(text("Download opens your browser. Import checks the exact model before use. After setup, inference works offline.", 12, MUTED, false));

        LinearLayout profileCard = card(body);
        profileCard.addView(text("02  /  CHOOSE A PROFILE", 11, LIME, true));
        profiles = new Spinner(this); profiles.setId(R.id.profile_picker); profiles.setSaveEnabled(false);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, NAMES);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        profiles.setAdapter(adapter); profiles.setBackgroundTintList(ColorStateList.valueOf(LIME));
        margin(profileCard, profiles, 10, 6);
        purpose = text(PURPOSES[0], 14, MUTED, false); profileCard.addView(purpose);
        margin(profileCard, text("Three workflows, one upstream model. These are not separately trained Krummholz models.", 12, MUTED, false), 12, 0);

        LinearLayout promptCard = card(body);
        promptCard.addView(text("03  /  ASK WITH EVIDENCE", 11, LIME, true));
        margin(promptCard, text("Source text", 15, CREAM, true), 18, 8);
        sourceInput = input("Paste a short source", 5); sourceInput.setId(R.id.source_input); promptCard.addView(sourceInput);
        margin(promptCard, text("Your question", 15, CREAM, true), 18, 8);
        questionInput = input("Ask about the source", 2); questionInput.setId(R.id.question_input); promptCard.addView(questionInput);
        exampleButton = button("Reset to fictional example", false);
        exampleButton.setOnClickListener(v -> resetExample()); margin(promptCard, exampleButton, 10, 4);
        clearButton = button("Clear this workspace", false); clearButton.setId(R.id.clear_button);
        clearButton.setOnClickListener(v -> { sourceInput.setText(""); questionInput.setText(""); clearAnswer(); });
        promptCard.addView(clearButton);
        margin(promptCard, text("Text is kept only on this screen and is not saved to a history or backup. Your keyboard and Android remain separate from this app.", 12, MUTED, false), 12, 0);
        askButton = button("Ask on this phone", true); askButton.setId(R.id.ask_button);
        askButton.setOnClickListener(v -> ask()); margin(body, askButton, 0, 10);
        operationStatus = text("Import the model to begin.", 14, MUTED, false); operationStatus.setId(R.id.operation_status);
        operationStatus.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE); body.addView(operationStatus);
        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setProgressTintList(ColorStateList.valueOf(LIME)); progress.setVisibility(View.GONE);
        margin(body, progress, 4, 4);
        cancelButton = button("Cancel", false); cancelButton.setId(R.id.cancel_button);
        cancelButton.setOnClickListener(v -> cancelCurrent(false)); cancelButton.setVisibility(View.GONE); body.addView(cancelButton);

        LinearLayout answerCard = card(body);
        answerCard.addView(text("FOR YOUR REVIEW", 11, LIME, true));
        answerText = text("Your checked answer will appear here.", 17, CREAM, false);
        answerText.setId(R.id.answer_text); answerText.setTextIsSelectable(true);
        margin(answerCard, answerText, 12, 12);
        citations = column(); answerCard.addView(citations);
        answerNote = text("Citation checks match quoted words. They do not establish that the answer is correct.", 12, MUTED, false);
        answerCard.addView(answerNote);
        margin(body, text("No action tools. No telemetry. No cloud fallback.\nPleias upstream baseline · local CPU inference", 12, MUTED, false), 6, 0);
        Button aboutButton = button("About & licenses", false);
        aboutButton.setOnClickListener(v -> showAbout()); margin(body, aboutButton, 12, 0);

        profiles.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                purpose.setText(PURPOSES[position]);
                if (position != selectedProfile) {
                    selectedProfile = position; cancelCurrent(false); resetExample();
                }
            }
            @Override public void onNothingSelected(AdapterView<?> parent) {}
        });
        resetExample(); controls();
    }

    private void showAbout() {
        if (aboutDialog != null && aboutDialog.isShowing()) return;
        StringBuilder content = new StringBuilder("Krummholz LLMs [ALPHA] · 0.1.0-alpha.1\n\n")
            .append("Acadia, Katahdin and Atlantic are three workflows using one upstream Pleias RAG baseline. ")
            .append("This alpha does not include Krummholz-trained model weights.\n\n")
            .append("Inference runs on your phone. Source text, questions and answers are not saved to a history or backup. ")
            .append("The app has no network permission, telemetry or action tools.\n\n");
        try {
            appendLicense(content, "NOTICE.txt");
            String[] files = getAssets().list("licenses");
            if (files != null) {
                Arrays.sort(files);
                for (String file : files) if (!"NOTICE.txt".equals(file)) appendLicense(content, file);
            }
        } catch (IOException error) {
            content.append("Some bundled notices could not be read in this build. See the official release for the complete license files.\n");
        }
        ScrollView scroll = new ScrollView(this); scroll.setFillViewport(true);
        scroll.setBackgroundColor(FOREST); scroll.setSaveEnabled(false);
        TextView notices = text(content.toString(), 14, CREAM, false);
        notices.setTextIsSelectable(true); notices.setPadding(dp(22), dp(18), dp(22), dp(22));
        scroll.addView(notices);
        aboutDialog = new AlertDialog.Builder(this).setTitle("About & licenses").setView(scroll)
            .setPositiveButton("Done", (dialog, which) -> dialog.dismiss()).create();
        aboutDialog.setOnDismissListener(dialog -> aboutDialog = null);
        aboutDialog.show();
    }

    private void appendLicense(StringBuilder content, String file) throws IOException {
        // Bundled plain text only. Never render notices as HTML or fetch remote content.
        try (InputStream input = getAssets().open("licenses/" + file);
             ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096]; int count;
            while ((count = input.read(buffer)) != -1) {
                if (bytes.size() + count > 262_144 || content.length() + bytes.size() + count > 1_048_576)
                    throw new IOException("Bundled notice exceeds the display limit.");
                bytes.write(buffer, 0, count);
            }
            content.append("────────\n").append(file).append("\n\n")
                .append(new String(bytes.toByteArray(), StandardCharsets.UTF_8)).append("\n\n");
        }
    }

    private void openModelDownload() {
        if (store == null || !store.spec.approved || store.spec.downloadUrl.isEmpty()) return;
        try { startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(store.spec.downloadUrl))); }
        catch (ActivityNotFoundException error) { operationStatus.setText("No browser is available. Download the model on another device and import the file here."); }
    }
    private void chooseModel() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE)
            .setType("*/*").putExtra(Intent.EXTRA_LOCAL_ONLY, true)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        try { startActivityForResult(intent, IMPORT_MODEL); }
        catch (ActivityNotFoundException error) { operationStatus.setText("This device has no file picker available."); }
    }
    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == IMPORT_MODEL && resultCode == RESULT_OK && data != null && data.getData() != null && current == null)
            importModel(data.getData());
    }

    private void importModel(Uri uri) {
        Job job = begin("Importing and verifying the model…");
        worker.execute(() -> {
            try {
                final int[] lastPercent = {-1};
                File file = store.importModel(uri, job.cancelled::get, bytes -> {
                    int percent = (int) (bytes * 100 / store.spec.bytes);
                    if (percent != lastPercent[0]) {
                        lastPercent[0] = percent;
                        ui(job, () -> { progress.setIndeterminate(false); progress.setProgress(percent); operationStatus.setText("Checking model bytes · " + percent + "%"); });
                    }
                }, job.importCancellation);
                load(job, file);
            } catch (Exception error) { failed(job, error, "The model could not be imported. Try a local copy of the approved file."); }
            finally { finish(job); }
        });
    }
    private void loadExisting() {
        Job job = begin("Verifying your saved model…");
        worker.execute(() -> {
            try { load(job, store.verifiedModel(job.cancelled::get)); }
            catch (Exception error) { failed(job, error, "The model could not be loaded. Import the approved file again."); }
            finally { finish(job); }
        });
    }
    private void load(Job job, File file) {
        if (job.cancelled.get()) throw new CancellationException();
        ui(job, () -> { loaded = false; progress.setIndeterminate(true); operationStatus.setText("Loading the local model…"); });
        engine.loadModel(file.getAbsolutePath(), CONTEXT, 2);
        if (job.cancelled.get()) throw new CancellationException();
        ui(job, () -> {
            loaded = true;
            modelStatus.setText(store.spec.name + "\nVerified and ready · runs on this phone\nShared by Acadia, Katahdin and Atlantic.");
            operationStatus.setText("Ready. Ask about your source.");
        });
    }
    private void ask() {
        if (!loaded || current != null || engine == null) return;
        final Protocol.Request request;
        final String prompt;
        try {
            String profile = KEYS[profiles.getSelectedItemPosition()];
            request = Protocol.request(profile, questionInput.getText().toString(), Collections.singletonList(
                new Protocol.Source(profile + "/workspace", sourceInput.getText().toString())));
            prompt = Protocol.prompt(request, CONTEXT, OUTPUT);
        } catch (Protocol.ValidationException error) { operationStatus.setText(error.getMessage()); return; }
        clearAnswer();
        Job job = begin("Reading the source on this phone…");
        worker.execute(() -> {
            try {
                if (job.cancelled.get()) throw new CancellationException();
                String raw = engine.generate(prompt, OUTPUT, count -> {
                    if (job.cancelled.get()) { engine.cancel(); return; }
                    ui(job, () -> {
                        if (!job.cancelled.get()) operationStatus.setText("Working locally · " + count + " tokens generated");
                    });
                });
                if (job.cancelled.get()) throw new CancellationException();
                Protocol.Answer answer = Protocol.parseAnswer(raw, request);
                ui(job, () -> { if (!job.cancelled.get()) { showAnswer(answer); operationStatus.setText("Answer ready for your review."); } });
            } catch (Exception error) { failed(job, error, "The local model could not complete this request. Try a shorter source."); }
            finally { finish(job); }
        });
    }

    private Job begin(String status) {
        Job job = new Job(); current = job;
        operationStatus.setText(status); progress.setIndeterminate(true); progress.setVisibility(View.VISIBLE);
        job.deadline = timer.schedule(() -> ui(job, () -> {
            job.timedOut = true; job.cancelled.set(true);
            job.importCancellation.cancel();
            if (engine != null) engine.cancel();
        }), 180, TimeUnit.SECONDS);
        controls(); return job;
    }
    private void failed(Job job, Exception error, String fallback) {
        final String message = job.timedOut ? "This operation reached the alpha’s time limit. Try a shorter source."
            : job.cancelled.get() || error instanceof CancellationException ? "Cancelled. Nothing was added to a history."
            : error instanceof Protocol.ValidationException || error instanceof IOException ? error.getMessage() : fallback;
        ui(job, () -> operationStatus.setText(message));
    }
    private void finish(Job job) {
        if (job.deadline != null) job.deadline.cancel(false);
        ui(job, () -> { current = null; progress.setVisibility(View.GONE); controls(); });
    }
    private void ui(Job job, Runnable action) {
        runOnUiThread(() -> { if (!destroyed && current == job) action.run(); });
    }
    private void cancelCurrent(boolean quiet) {
        Job job = current;
        if (job == null) return;
        job.cancelled.set(true);
        job.importCancellation.cancel();
        if (engine != null) engine.cancel();
        if (!quiet) operationStatus.setText("Cancelling…");
    }
    private void controls() {
        if (askButton == null) return;
        boolean busy = current != null;
        boolean provisioned = store != null && store.spec.approved && engine != null;
        askButton.setEnabled(!busy && loaded && engine != null);
        importButton.setEnabled(!busy && provisioned);
        reloadButton.setVisibility(provisioned && store.hasModel() && !loaded ? View.VISIBLE : View.GONE);
        reloadButton.setEnabled(!busy && provisioned);
        downloadButton.setEnabled(!busy && provisioned && !store.spec.downloadUrl.isEmpty());
        profiles.setEnabled(!busy); sourceInput.setEnabled(!busy); questionInput.setEnabled(!busy);
        clearButton.setEnabled(!busy); exampleButton.setEnabled(!busy);
        cancelButton.setVisibility(busy ? View.VISIBLE : View.GONE);
    }
    private void resetExample() {
        int index = Math.max(0, profiles.getSelectedItemPosition());
        sourceInput.setText(SOURCES[index]); questionInput.setText(QUESTIONS[index]); clearAnswer();
    }
    private void clearAnswer() {
        answerText.setText("Your checked answer will appear here."); citations.removeAllViews();
        answerNote.setText("Citation checks match quoted words. They do not establish that the answer is correct.");
    }
    private void showAnswer(Protocol.Answer answer) {
        // TextView renders plain text only: no HTML, links, Markdown, or raw reasoning.
        answerText.setText(answer.text); citations.removeAllViews();
        for (Protocol.Citation citation : answer.citations) {
            TextView label = text("[" + citation.number + "]  " + citation.sourceId, 12, LIME, true);
            margin(citations, label, 8, 6);
            TextView quote = text("“" + citation.quote + "”", 14, CREAM, false);
            quote.setTextIsSelectable(true); margin(citations, quote, 0, 14);
        }
        answerNote.setText(answer.warning);
    }
    @Override protected void onStop() { cancelCurrent(true); super.onStop(); }
    @Override protected void onSaveInstanceState(Bundle state) {
        // Do not persist view hierarchy, source text, question, answer, or citations.
        super.onSaveInstanceState(new Bundle());
    }
    @Override protected void onDestroy() {
        destroyed = true; cancelCurrent(true);
        if (aboutDialog != null) aboutDialog.dismiss();
        if (engine != null) engine.close();
        worker.shutdownNow(); timer.shutdownNow(); super.onDestroy();
    }

    private LinearLayout column() {
        LinearLayout view = new LinearLayout(this); view.setOrientation(LinearLayout.VERTICAL);
        view.setSaveEnabled(false); view.setSaveFromParentEnabled(false); return view;
    }
    private LinearLayout card(LinearLayout parent) {
        LinearLayout view = column(); view.setPadding(dp(18), dp(18), dp(18), dp(18));
        view.setBackground(background(CARD, LINE, 20)); margin(parent, view, 0, 18); return view;
    }
    private TextView text(String value, int size, int color, boolean bold) {
        TextView view = new TextView(this); view.setText(value); view.setTextSize(size); view.setTextColor(color);
        view.setLineSpacing(dp(3), 1); view.setSaveEnabled(false); view.setSaveFromParentEnabled(false);
        if (bold) view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        return view;
    }
    private EditText input(String hint, int lines) {
        EditText view = new EditText(this); view.setTextSize(15); view.setTextColor(CREAM); view.setHintTextColor(MUTED);
        view.setHint(hint); view.setMinLines(lines); view.setGravity(Gravity.TOP | Gravity.START);
        view.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        view.setImeOptions(EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING | EditorInfo.IME_FLAG_NO_EXTRACT_UI);
        view.setImportantForAutofill(View.IMPORTANT_FOR_AUTOFILL_NO);
        view.setSaveEnabled(false); view.setSaveFromParentEnabled(false);
        view.setPadding(dp(12), dp(12), dp(12), dp(12)); view.setBackground(background(FOREST, LINE, 12));
        return view;
    }
    private Button button(String label, boolean filled) {
        Button button = new Button(this); button.setText(label); button.setAllCaps(false);
        button.setTextColor(filled ? FOREST : CREAM); button.setTextSize(14); button.setMinHeight(dp(48));
        button.setBackgroundTintList(ColorStateList.valueOf(filled ? LIME : CARD));
        button.setSaveEnabled(false); return button;
    }
    private GradientDrawable background(int color, int stroke, int radius) {
        GradientDrawable background = new GradientDrawable(); background.setColor(color);
        background.setCornerRadius(dp(radius)); background.setStroke(dp(1), stroke); return background;
    }
    private void margin(LinearLayout parent, View child, int top, int bottom) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.setMargins(0, dp(top), 0, dp(bottom)); parent.addView(child, params);
    }
    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
}
