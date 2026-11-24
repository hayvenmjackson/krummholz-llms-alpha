---
license: apache-2.0
library_name: pytorch
tags:
- krummholz
- field-records
- on-device
- structured-data
- safetensors
- from-scratch
---

# Sebago 40K

Sebago reads a field record on the device you already have. Pick **Water temperature** in the lakeshore survey and it reads back the value. Hide that line, leave the other measurements visible, and ask again. Sebago can say the field is missing. We named it for Sebago Lake in Maine. Krummholz AI LLC, Belfast, Maine, released this checkpoint on November 24, 2026. [Try the app](https://krummholz.ai/apps/sebago/) · [Model page](https://krummholz.ai/llms/sebago/) · [Field notes](https://krummholz.ai/field-notes/sebago-field-reader/)

## Downloads and installation

| Device | Get the app |
| --- | --- |
| Windows | [Download the offline ZIP](https://huggingface.co/krummholz-ai/Sebago-40K/resolve/main/Sebago-offline.zip?download=true). Extract the ZIP and open Sebago.html in Edge, Chrome or Firefox. |
| macOS | [Download the offline ZIP](https://huggingface.co/krummholz-ai/Sebago-40K/resolve/main/Sebago-offline.zip?download=true). Extract it in Finder and open Sebago.html in Safari or another current browser. |
| Linux | [Download the offline ZIP](https://huggingface.co/krummholz-ai/Sebago-40K/resolve/main/Sebago-offline.zip?download=true). Extract it and open Sebago.html in Firefox or Chromium. |
| Android | [Install the web app](https://krummholz.ai/apps/sebago/#install). In Chrome choose **Menu → Add to Home screen → Install**. [Download the web-app bundle](https://huggingface.co/krummholz-ai/Sebago-40K/resolve/main/Sebago-web-app.zip?download=true). |
| iOS | [Install the web app](https://krummholz.ai/apps/sebago/#install). In Safari choose **Share → Add to Home Screen**. Enable **Open as Web App** if offered, then tap **Add**. [Download the web-app bundle](https://huggingface.co/krummholz-ai/Sebago-40K/resolve/main/Sebago-web-app.zip?download=true). |

The desktop ZIP contains a self-contained HTML app. The mobile installation downloads and caches the app in the browser. Open it once online on the phone and look for **Offline copy ready** before going into the field. The web-app ZIP is the same installable app for hosting on an HTTPS site. **App Store and Google Play listings are planned for the coming weeks.** [Single HTML download](https://huggingface.co/krummholz-ai/Sebago-40K/resolve/main/Sebago.html?download=true) · [Install guide](INSTALL.txt)

## Weights and architecture

| Item | Value |
| --- | --- |
| Origin | Independently initialized by Krummholz AI and trained from scratch |
| Task | Return the requested value in a structured record, or emit an abstention |
| Architecture | Causal Transformer, 2 layers, 4 attention heads, width 48, feed-forward width 96 |
| Trainable parameters | 40,530 |
| Vocabulary | 18 symbols with six field keys and eight value symbols |
| Record input | One requested key, three field/value pairs and an answer delimiter, 14 tokens total |
| Output | Three value symbols and an end token, or an abstention token and an end token |
| Checkpoint format | Float32 safetensors, 164,616 bytes including header |
| Runtime | PyTorch on CPU or the included browser decoder. The app interface is in English. |

[Download trained weights](https://huggingface.co/krummholz-ai/Sebago-40K/resolve/main/sebago-40k.safetensors?download=true) · [Download random initialization](https://huggingface.co/krummholz-ai/Sebago-40K/resolve/main/sebago-init.safetensors?download=true) · [Release checksums](SHA256SUMS)

## Training and evaluation

Training used 72,000 programmatically generated examples, half answerable and half missing the requested field. AdamW used a learning rate of 0.0015, batch size 48 and 1,500 updates. Training used seed 271124, two CPU threads, Python 3.13 and PyTorch 2.14.1. The record generator used 416 of 512 possible three-symbol values for training and reserved the other 96. There was no web text or customer field data in the training set. [Training program](train.py)

Testing used 512 pairs and 95 distinct requested values from the reserved set. Both halves of a pair contain the **same three values**. Replacing the requested field's key makes its value unsupported, even though the value is still there under a different heading.

| Held-out measure | Initial weights | Trained weights |
| --- | ---: | ---: |
| Copied the supplied value exactly | 0/512 | 511/512 |
| Abstained when the field was absent | 0/512 | 512/512 |
| Got both halves of the pair right | 0/512 | 511/512 |

The one error substituted a single symbol. The browser model matched PyTorch on all 1,024 held-out completions. On the three included fictional example records, the app returned the expected outcome on all 96 combinations of selected field and field visibility. Those two measurements answer different questions. The first tests the learned weights. The second also includes the app's deterministic verification. [Case-level results](evaluation.json) · [Browser inference](model.js)

## Intended use and limits

Sebago is useful for reading a known field from a short structured record when connectivity is unavailable. The example desks cover lake surveys, forestry inventory and fisheries landings. The app assigns unique symbolic codes to a record's values, then checks each generated answer against the visible requested field before displaying it. If verification fails, the app asks the reader to check the record. A returned value can still be wrong when the source measurement itself is wrong.

Sebago has no general-purpose language, image or speech interface. Its menu defines the questions it can answer. Training and testing used a single symbolic grammar and one initialization. Expected performance on a new field schema needs its own held-out test. We verified the app in Edge on Windows at desktop and phone viewport sizes, including a cached reload with the network disabled. Physical iOS, Android, macOS and Linux device performance and battery life have not been measured. The model does not send records to a server. The browser can retain local edits on the same device. [Field notes and a worked example](FIELD-NOTES.md)

## Retraining

You can retrain Sebago yourself. Download the source tree and work in a copy so that your experiment keeps its own checkpoints. In a CPU Python environment with PyTorch and NumPy installed, run these commands from that copy.

```sh
python train.py
python build.py
```

The trainer initializes all 40,530 parameters, generates the training and test records, and writes both checkpoints and evaluation.json. To adapt the task, change the `pair` record generator in train.py with examples from your schema. Preserve the three-pair input protocol and output vocabulary to keep the browser decoder compatible. Reserve new source records and values for evaluation. The builder checks the trained weight hash against evaluation.json before embedding that checkpoint into the app. Update the model card and the app's example/test labels for a changed protocol. Training requires PyTorch. Running the downloaded app only needs a browser. [Training program](train.py) · [App builder](build.py) · [App interface](app.js)

Krummholz AI can also retrain Sebago for a small fee. Send the fields you need, two or three representative examples and what should happen when a value is missing. [Contact Krummholz AI](https://krummholz.ai/contact/)

## License and credits

Krummholz AI's original weights, training code and app code are Apache-2.0. The corporate name and marks remain trademarks of Krummholz AI LLC. The lake photograph by NewtonCourt was resized, compressed and cropped under CC BY-SA 4.0. DM Sans and Libre Franklin ship under the SIL Open Font License. [Apache License](LICENSE) · [Attribution notices](NOTICE.txt) · [Original photograph](https://commons.wikimedia.org/wiki/File:Sebago_Lake_P1030495.jpg) · [Photo license](https://creativecommons.org/licenses/by-sa/4.0/) · [DM Sans license](assets/dmsans-OFL.txt) · [Libre Franklin license](assets/librefranklin-OFL.txt)
