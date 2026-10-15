# SMS Spam Detector - Training Report

- Generated: 2026-10-15T14:03:31+00:00
- Dataset file: `/home/user/sms-spam-detector/data/raw/SMSSpamCollection.txt`
- Dataset SHA-256: `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`
- Python 3.13.16 | scikit-learn 1.6.1 | pandas 2.2.3 | numpy 2.3.5
- Training time: 4.33 s

## 1. Dataset

- Rows in file: **5574**
- Rows used after cleaning: **5160**
- Missing messages: 0 | missing labels: 0
- Duplicate rows found: 414 (removed: yes)
- Identical text with conflicting labels: 0
- Unrecognised labels dropped: 0

| Class | Rows | Share |
| --- | ---: | ---: |
| Ham | 4518 | 87.56% |
| Spam | 642 | 12.44% |

Message length (characters): min 2, median 61.0, mean 79.14, max 910.

## 2. Split

- Test size: 0.2 | random_state: 42 | stratified: True
- Train rows: 4128 | Test rows: 1032
- Positive class for Precision/Recall/F1: **spam**

## 3. Text cleaning and features

- Unicode NFKC normalisation, lowercasing, non-ASCII removal.
- URLs, e-mails, phone numbers, money amounts and numbers replaced by placeholder tokens (`__url__`, `__email__`, `__phone__`, `__money__`, `__num__`).
- TF-IDF: ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True.
- The vectorizer is fitted **inside** each pipeline on the training split only (no data leakage).

## 4. Results on the held-out test split

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | ROC-AUC | CV mean F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multinomial Naive Bayes | 0.9845 | 0.9667 | 0.9062 | 0.9355 | 0.9917 | 0.9444 |
| Logistic Regression | 0.9893 | 0.9606 | 0.9531 | 0.9569 | 0.9980 | 0.9552 |
| Linear Support Vector Machine | 0.9913 | 0.9760 | 0.9531 | 0.9644 | 0.9972 | 0.9578 |

Best model by F1 (spam): **Linear Support Vector Machine** (F1 = 0.9644).

## 5. Confusion matrices (test split)

**Multinomial Naive Bayes**

|  | predicted Ham | predicted Spam |
| --- | ---: | ---: |
| actual Ham | 900 | 4 |
| actual Spam | 12 | 116 |

**Logistic Regression**

|  | predicted Ham | predicted Spam |
| --- | ---: | ---: |
| actual Ham | 899 | 5 |
| actual Spam | 6 | 122 |

**Linear Support Vector Machine**

|  | predicted Ham | predicted Spam |
| --- | ---: | ---: |
| actual Ham | 901 | 3 |
| actual Spam | 6 | 122 |

## 6. Strongest learned indicators

**Multinomial Naive Bayes** - spam side:

`claim` (+5.855), `prize` (+5.709), `call __phone__` (+5.534), `nokia` (+5.350), `box __num__` (+5.313), `your mobile` (+5.285), `have won` (+5.271), `__phone__ now` (+5.242), `to claim` (+5.205), `ppm` (+5.196)

**Multinomial Naive Bayes** - ham side:

`gt` (-4.479), `lt` (-4.478), `lt gt` (-4.388), `lor` (-4.308), `da` (-4.263), `he` (-4.210), `later` (-3.869), `she` (-3.862), `lol` (-3.769), `anything` (-3.727)

**Logistic Regression** - spam side:

`__money__` (+12.273), `__phone__` (+11.463), `__num__` (+8.374), `__url__` (+7.201), `to __num__` (+7.100), `text` (+6.768), `txt` (+6.109), `call` (+5.752), `free` (+5.683), `your` (+5.233)

**Logistic Regression** - ham side:

`my` (-3.650), `me` (-3.218), `gt` (-2.517), `lt` (-2.488), `ok` (-2.394), `got` (-2.358), `lt gt` (-2.273), `that` (-2.228), `then` (-2.193), `your call` (-2.180)

**Linear Support Vector Machine** - spam side:

`__phone__` (+3.061), `__money__` (+3.004), `__url__` (+1.913), `__num__` (+1.823), `to __num__` (+1.792), `text` (+1.718), `txt` (+1.599), `ringtone` (+1.421), `free` (+1.302), `service` (+1.297)

**Linear Support Vector Machine** - ham side:

`my` (-0.848), `me` (-0.790), `your call` (-0.771), `gt` (-0.688), `lt` (-0.666), `lt gt` (-0.639), `got` (-0.605), `yup` (-0.579), `ok` (-0.538), `work` (-0.528)

## 7. Limitations

- The corpus is a 2011-vintage UK collection: vocabulary, currency and phone number formats differ from today's messages, so accuracy will drop on modern traffic.
- Roughly 5% of the corpus is spam: high accuracy alone is misleading, which is why precision, recall and F1 for the spam class are reported as well.
- Model scores are estimates/decision margins, not calibrated proof.
- The classifier only sees the message text; it has no metadata (sender reputation, sending frequency, link content).
