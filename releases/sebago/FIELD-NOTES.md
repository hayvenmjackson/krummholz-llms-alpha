# Reading the right field

**November 24, 2026 · Krummholz AI, Belfast, Maine**

We built Sebago around a field problem. A number can be written on a form and still be the wrong answer to your question. A fisheries landing weight might be right next to a water temperature. Copying the first plausible number is an easy way to get the record wrong.

Open the fictional lakeshore survey in Sebago. The water temperature reads **14.6 °C** and visibility is **2.4 m**. Choose **Water temperature** and click **Read field**. Now hide the temperature line, leave visibility alone, and ask again. The source no longer supports a temperature reading. You can edit the value, restore the line and see how the answer changes. [Try the field reader](https://krummholz.ai/apps/sebago/)

The training test makes guessing a little harder than hiding the line. Each test record contains three values. In the second half of a pair, the expected value is still present, in the same position, beneath a different field key. A model that memorized values or copied the entry at that position would fail this test. We held out 96 possible values from training and tested 512 record pairs. The new weights copied the supplied field exactly 511 times and abstained in all 512 missing-field cases. The lone copy error changed one of the three symbols. [Inspect the predictions](evaluation.json)

The downloadable app has a second check. If the model proposes a value, the reader checks that it belongs to the visible field you asked about. The tests for the weights score their raw predictions before that check. You can reproduce them, retrain the model yourself and use a separate set of records to see whether a new schema works. Krummholz AI can also do this for a small fee. [Training source](train.py) · [Contact Krummholz AI](https://krummholz.ai/contact/)

The model weighs about 165 kB on disk. The larger app download includes the photo, typefaces and browser inference code, so it can run without a model server. We tested editing, abstention and offline reload in Edge on Windows, including phone-width layouts. A day in the field on a physical handset is still a different test. [Model card](README.md) · [Downloads](https://krummholz.ai/llms/sebago/)
