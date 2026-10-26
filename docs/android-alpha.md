# Krummholz LLMs Android alpha

**Draft guide for 0.1.0-alpha.1 (build 1). Source compilation passes; model, device and release validation are pending.** The checked-in development model manifest is unprovisioned, so its setup buttons remain disabled until the release build contains the approved model details. The install steps below apply once that verified release is available.

The app is a small workspace for asking questions about a passage you supply. Acadia, Katahdin and Atlantic are three workflows. The initial upstream Pleias candidates failed quality review, and a replacement is being evaluated. This alpha does not contain separately trained Krummholz models.

## Install on your Samsung S26

Once the verified alpha is published:

1. Open the [official GitHub releases page](https://github.com/hayvenmjackson/krummholz-llms-alpha/releases) in your phone's browser and download the alpha APK. Open the completed download to install it. If Android asks for permission to install from that browser, follow its installation prompt.
2. Open **Krummholz LLMs [ALPHA]**. Tap **Get model from GitHub** to download the separate model file in your browser. Wait for the download to finish.
3. Return to the app, tap **Import downloaded model**, and choose the completed `.gguf` file from local Downloads. The app checks its exact size and SHA-256 hash before loading it.
4. Wait for **Verified and ready · runs on this phone**. You can then use the workspace offline.

There are two downloads: the APK and the approved model. No ADB, developer options, computer connection or Wi-Fi pairing is required. Leave room for both the downloaded model and the app's private imported copy. The app accepts the exact approved model for that build; another GGUF cannot be substituted.

## Try your first question

Start with the fictional example already on screen and tap **Ask on this phone**. The status shows progress; **Cancel** stops the current operation.

| Profile | Starting point |
| --- | --- |
| **Acadia** | Questions about customer-service policies and information for a human handoff. |
| **Katahdin** | Forestry and fisheries source questions. Numerical results must already be in the supplied source. |
| **Atlantic** | Questions about local notes or other text you paste into the workspace. |

Replace **Source text** with a short passage and put a direct question in **Your question**. This version takes pasted text, not document attachments. It answers from that input; it cannot browse websites, send messages, make purchases, edit files or take other external actions.

Completed answers appear under **FOR YOUR REVIEW**, with source quotations when present. Quotation checks verify that quoted words match the source; they do not prove that the answer's interpretation is correct. An answer without verified citations carries its own review note. Incomplete answers or invalid citations are withheld.

## Your workspace and model

**Clear this workspace** removes the current source, question, answer and citations. **Reset to fictional example** restores the selected profile's example. Switching profiles also replaces the current workspace with that profile's example.

Text is not saved to a conversation history or backup. Leaving the app cancels active work; if Android recreates the screen, such as after rotation or closing the app, it starts fresh with the default example. The verified model remains in app-private storage and is checked again when loaded. Clearing the workspace does not remove it.

Inference runs on the phone after setup. The app has no network permission, telemetry or cloud fallback; the download button opens your separate browser. Your keyboard and Android have their own settings. **About & licenses** contains the model and runtime notices.

## If something gets stuck

| What you see | What to try |
| --- | --- |
| **Model provisioning is not complete in this development build** | This build has no approved model manifest. Use the verified release once available. |
| A size or SHA-256 check fails | Download the model linked by this build again, let it finish, then import the local file. A failed import preserves any previously verified model. |
| The app asks for a fully downloaded local file | Save the file into local Downloads first. A cloud-only item or streamed file cannot be imported. |
| A free-storage message | Free space for the imported copy, then retry. |
| **Load saved model** appears | Tap it to verify and load the model already imported. |
| The source is too long, the answer is incomplete, or the time limit is reached | Try a shorter passage and one focused question. Each operation has a three-minute limit. Keep the app open while it works. |

S26 performance and installation testing are still pending.
