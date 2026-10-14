# מנוע עריכת וידאו (Python)

כל פריים מורכב בקוד (OpenCV + numpy), עברית מצוירת עם Pillow + RAQM, ותמונה וסאונד מחוברים עם ffmpeg.

## מבנה
| קובץ | תפקיד |
|---|---|
| `engine/transcribe.py` | תמלול whisper.cpp → מילים עם חותמות זמן, יישור לפי אנרגיית הסאונד → `words.json` + טבלת `words.md` |
| `engine/timemap.py` | מפת זמן מקור↔סופי: חיתוך שתיקות עם מעבר רך (דיזולב + crossfade), פריים קפוא, `src_to_out` |
| `engine/audio.py` | חיתוך הסאונד לפי אותה מפה, מיקס אפקטים קוליים, נרמול עוצמה ל-‎-14 LUFS |
| `engine/text.py` | טקסט עברי (RTL) עם RAQM, צללית, מלבן מעוגל, קומפוזיציה |
| `engine/captions.py` | כתוביות מונפשות מילה-אחר-מילה |
| `engine/music.py` | מוזיקת רקע מקורית שמסונתזת בקוד (ללא זכויות יוצרים) + דאקינג מתחת לדיבור |
| `engine/camera.py` | מצלמה וירטואלית: זום, פאנץ׳, דחיפה איטית, רעידה – על התמונה ועל המסכה יחד |
| `engine/fx.py` | שכבות גרפיקה: כותרת מאחורי הדמות, כרטיס שם, חותמת, כרטיס זהב, רשימה, כרטיס סיום |
| `engine/sfx.py` | אפקטים קוליים מסונתזים |
| `engine/mask.py` | מסכת דמות לכל פריים עם rembg / u2net_human_seg (מחושבת פעם אחת → `work/mask.mkv`) |
| `engine/render.py` | `frame_at(t)` → רינדור מקבילי על כל הליבות, קידוד H.264 איכותי + סאונד, גיליון קונטקט |
| `engine/project.py` | מחבר הכל: מקור, מילים, מפת זמן, שכבות |
| `projects/reel01/edit.py` | העריכה של הסרטון הנוכחי |

## הרצה
```bash
scripts/setup.sh                                  # פעם אחת
cp <video>.mp4 input/source.mp4
python3 -m engine.transcribe input/source.mp4 work/
python3 -m engine.mask input/source.mp4 work/mask.mkv
python3 -m projects.reel01.edit                   # → out/reel01_v1.mp4 + work/contact_v1_*.jpg
```

אפקט ממוקם לפי מילה: `p.w(<מספר מילה>)` מחזיר את הזמן שלה בסרטון הסופי.

פונט: Heebo (Google Fonts, רישיון OFL – `fonts/OFL.txt`).
