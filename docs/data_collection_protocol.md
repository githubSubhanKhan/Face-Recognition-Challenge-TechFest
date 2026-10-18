# Data Collection Protocol (own Pakistani dataset)

Your own consented collection is the most valuable data you have: it is the only data that (a) is truly
Pakistani, (b) has a real consent trail, which the presentation is judged on, and (c) is guaranteed NOT to be
inside the pretrained models' web-scraped training sets. Val/test splits are drawn from it by default.

## Target
- **50–80 people** minimum (more matters more than more photos per person). Aim for diversity across:
  - region / ethnicity: Sindhi, Punjabi, Pashtun, Baloch, Muhajir/Urdu-speaking, Kashmiri, Gilgit-Baltistan, Hazara …
    (Karachi + NED makes this realistic — ask friends, family, hostel residents, staff)
  - gender (target ≥ 40 % women), age groups (include parents / staff for 35+ and 50+), skin tones
- **15–25 photos per person** following the shot list below (≈ 10 minutes per person with two phones).

## Shot list per person (tick on the sheet)
| # | Condition | Notes |
|---|---|---|
| 1–3 | Frontal, good indoor light | neutral, smile, slight head turn |
| 4–5 | Outdoor daylight | one in direct sun (hard shadows) |
| 6–7 | Low light | corridor / evening, phone flash OFF |
| 8 | Backlit | person in front of a window |
| 9–10 | Head covering they normally wear | dupatta, hijab, niqab-off, turban, topi/cap — **only if they wear it** |
| 11 | Glasses / sunglasses | own glasses if they have them |
| 12 | Mask | surgical / cloth |
| 13–14 | Profile-ish (30–45°) | left and right |
| 15 | Far away (3–5 m) | gives genuinely low-resolution faces |
| 16 | Cheap / old phone or front camera | device diversity |
| 17 | CCTV-like | from above, 3 m away, through a doorway, or a screenshot of a video call |
| 18+ | Different day / outfit | if the person is available on a second day: very valuable (session variation) |

Two different phones per person if possible. Do not use beauty filters / portrait mode.

## Storage
```
data/raw/own_collection/P0001/IMG_xxx.jpg
data/raw/own_collection/P0002/...
```
- Folder name = participant code. Never names.
- Fill `data/identities.csv` (identity,gender,age_group,skin_tone,region) — copy from the form's optional fields.
- Optionally fill `data/images.csv` (filename,headwear,glasses,mask,lighting,device) — it powers the per-group
  breakdown that the judges explicitly asked for ("how the pipeline performs across different groups").
  Use values like headwear: none/hijab/dupatta/turban/cap; glasses: none/glasses/sunglasses; mask: none/mask;
  lighting: normal/low/backlit/outdoor; device: phone_a/phone_b/cctv_like.

## Ethics checklist for the slides
- Signed consent for every participant, adults only, code-based storage, deletion date, withdrawal route.
- Public datasets used only under their research licences; celebrity images used for training only,
  never for reported test numbers (they may be in the pretrained models' training data).
- Fairness: report per-group FAR/FRR, not just overall accuracy.
