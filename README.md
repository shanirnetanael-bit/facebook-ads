# מנוע עריכת וידאו (Python)

כל פריים מורכב בקוד (OpenCV + numpy), עברית מצוירת עם Pillow + RAQM, ותמונה וסאונד מחוברים עם ffmpeg.

## מבנה
| קובץ | תפקיד |
|---|---|
| `engine/transcribe.py` | תמלול → מילים עם חותמות זמן → `words.json` + טבלת `words.md`. מעדיף whisper.cpp + ivrit.ai; אם המודל לא זמין – `engine/asr.py` |
| `engine/asr.py` | Whisper large-v3-turbo מקומי (ONNX), זמן לכל מילה מ-cross-attention + DTW, פרומפט הקשר לשמות ומונחים |
| `engine/timemap.py` | מפת זמן מקור↔סופי: חיתוך שתיקות עם מעבר רך (דיזולב + crossfade), פריים קפוא, `src_to_out` |
| `engine/audio.py` | ניקוי קול (high-pass, RNNoise, EQ, דה-אסר, קומפרסור, גייט רך), חיתוך הסאונד לפי אותה מפה, מיקס אפקטים, נרמול ל-‎-14 LUFS |
| `engine/text.py` | טקסט עברי (RTL) עם RAQM, צללית, מלבן מעוגל, קומפוזיציה |
| `engine/captions.py` | כתוביות מונפשות מילה-אחר-מילה |
| `engine/music.py` | מוזיקת רקע מקורית שמסונתזת בקוד (ללא זכויות יוצרים): פאדים רכים או קלאסית-בארוק קצבית, פינוי תדרי הדיבור + דאקינג |
| `engine/camera.py` | מצלמה וירטואלית: זום, פאנץ׳, דחיפה איטית, רעידה – על התמונה ועל המסכה יחד |
| `engine/fx.py` | שכבות גרפיקה: כותרת מאחורי הדמות, כרטיס שם, חותמת, כרטיס זהב, רשימה, שעון, ציר זמן, היפוך מספר, חיצים למטה, כרטיס סיום |
| `engine/sfx.py` | אפקטים קוליים מסונתזים |
| `engine/mask.py` | מסכת דמות לכל פריים עם rembg / u2net_human_seg (מחושבת פעם אחת → `work/mask.mkv`) |
| `engine/render.py` | `frame_at(t)` → רינדור מקבילי על כל הליבות, קידוד H.264 איכותי + סאונד, גיליון קונטקט |
| `engine/project.py` | מחבר הכל: מקור, מילים, מפת זמן, שכבות |
| `projects/reel01/edit.py` | עריכת סרטון 1 (פוליסת סיעוד) |
| `projects/reel02/edit.py` | עריכת סרטון 2 ("הזמן שלכם שווה כסף") – טבלת המילים ב-`projects/reel02/words.md` |

## הרצה
```bash
scripts/setup.sh                                  # פעם אחת
cp <video>.mp4 input/reel02.mp4
python3 -m engine.transcribe input/reel02.mp4 work/reel02/ "שמות ומונחים להקשר"
python3 -m engine.mask input/reel02.mp4 work/reel02/mask.mkv
python3 -m projects.reel02.edit                   # → out/reel02_v1.mp4 + work/reel02/contact_v1_*.jpg
```

אפקט ממוקם לפי מילה: `p.w(<מספר מילה>)` מחזיר את הזמן שלה בסרטון הסופי.

פונט: Heebo (Google Fonts, רישיון OFL – `fonts/OFL.txt`).
