# دستیار دانش‌نامه — سیستم RAG محلی

سیستم پردازش زبان طبیعی با معماری **RAG** (تولید تقویت‌شده با بازیابی)، کاملاً محلی و
شی‌گرا، ساخته‌شده برای اجرای روی سرویس‌های llama.cpp شما:

| سرویس | مدل | پورت | نقش |
|---|---|---|---|
| مدل زبانی | Qwen3-4B (Q4_K_M) | `8080` | تولید پاسخ نهایی + فراخوانی ابزار |
| امبدینگ | bge-m3 (F16) | `8081` | بردارسازی متن‌ها و پرس‌وجوها |
| ذخیره‌گاه | FAISS (IndexFlatIP) | — | جست‌وجوی برداری معنایی |

---

## ۱) معماری

```
                      ┌──────────────────────────────────────────┐
                      │              docker-compose              │
                      │  ┌────────────┐      ┌────────────┐      │
                      │  │  Qwen3-4B  │      │   bge-m3   │      │
                      │  │  :8080     │      │   :8081    │      │
                      │  └─────┬──────┘      └──────┬─────┘      │
                      └────────┼────────────────────┼────────────┘
                               │ /v1/chat/          │ /v1/embeddings
                               │ completions        │
        ┌──────────────────────┼────────────────────┼─────────────────────┐
        │                      ▼                    ▼                     │
        │  پروژه پایتون (این ریپو)                                       │
        │                                                               │
        │  LLMClient ──────► RAGPipeline ◄────── EmbeddingService       │
        │  (api_client)         │    ▲            (EmbeddingClient)      │
        │                       │    │                                   │
        │   Retriever ◄─────────┘    │ خروجی مدل                         │
        │      │                     ▼                                   │
        │      │             ToolRegistry + tool_parsing                 │
        │      ▼                  (فراخوانی ابزار توسط خودِ مدل)          │
        │  VectorStore (FAISS + metadata.json)                           │
        │                                                               │
        │  ChatHistory (SQLite)          Streamlit UI (فارسی/RTL)        │
        └───────────────────────────────────────────────────────────────┘
```

جریان پاسخ‌دهی:

```
سوال کاربر ──► امبد شدن (bge-m3) ──► retrieve (FAISS) ──► نشستن در کانتکست
        ──► تولید پاسخ (Qwen3) ──► در صورت نیاز: فراخوانی ابزار search
        توسط خودِ مدل ──► نتیجهٔ ابزار به مدل ──► پاسخ نهایی + نمایش منابع
```

نکتهٔ مهم: کد امبدینگ (`EmbeddingClient` / `EmbeddingService`) کاملاً از کد مدل
زبانی (`LLMClient`) جدا است — دو کلاینت مستقل با مسئولیت‌های مشخص.

---

## ۲) فرمت دقیق `data.csv` (مهم)

دیتاست شما باید یک فایل CSV با **دو ستون** باشد:

| ستون | نام‌های قابل قبول | توضیح |
|---|---|---|
| نام فایل | `filename` (یا file_name / file / name / source / title) | نام فایل مبدأ؛ مثلاً `report.pdf` |
| محتوا | `content` (یا text / body / document / contents) | متن کامل استخراج‌شده از فایل |

قواعد:

1. **انکودینگ UTF-8** (BOM هم تحمل می‌شود؛ خروجی اکسل مشکلی ندارد)
2. **سطر اول هدر است** و نام ستون‌ها در آن می‌آید
3. **پاراگراف‌ها داخل محتوا با یک خط خالی** از هم جدا شوند (`\n\n`) — چانکر
   دقیقاً روی همین مرزها چانک می‌کند و چیزی از بین نمی‌رود
4. سلول‌های چندخطی طبق استاندارد CSV داخل `" "` هستند
5. هر ردیف = یک فایل PDF (یا هر سند دیگری)

نمونه:

```csv
filename,content
"report.pdf","فصل اول: مقدمه

این گزارش به بررسی وضعیت现状 می‌پردازد. متن پاراگراف دوم در همین بلاک قرار دارد.

فصل دوم: داده‌ها

توضیحات فصل دوم اینجا شروع می‌شود و ادامه می‌یابد."
"notes.pdf","یادداشت‌های جلسه …

متن یادداشت‌ها"
```

### ساخت دیتاست از روی فایل‌های PDF (اختیاری)

اسکریپت کوچک زیر با کتابخانهٔ `pypdf` (نصب جدا: `pip install pypdf`) از هر پوشهٔ
PDF دیتاست استاندارد می‌سازد:

```python
import csv
from pathlib import Path
from pypdf import PdfReader

with open("data/data.csv", "w", encoding="utf-8", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["filename", "content"])
    for pdf_path in sorted(Path("pdfs").glob("*.pdf")):
        reader = PdfReader(pdf_path)
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        content = "\n\n".join(page for page in pages if page)   # خط خالی بین صفحه‌ها
        writer.writerow([pdf_path.name, content])
print("دیتاست ساخته شد: data/data.csv")
```

نکته: اگر متن PDF شما خط‌به‌خط شکسته شده است (هر خط یک `\n`)، مشکلی نیست —
چانکر خطوط را مثل پاراگراف در نظر می‌گیرد و باز هم بدون افت متن چانک می‌شود.

---

## ۳) نصب

```bash
cd rag-system
python -m venv .venv
# ویندوز:  .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# فایل تنظیمات (اختیاری — پیش‌فرض‌ها با پورت‌های شما هماهنگ است)
cp .env.example .env
```

پیش‌نیاز: هر دو سرویس docker-compose شما بالا باشند (`qwen3` روی 8080 و
`bge-m3` روی 8081). برای اطمینان:

```bash
python scripts/check_services.py
```

---

## ۴) اجرا مرحله‌به‌مرحله

### مرحلهٔ ۱ — امبد دیتاست و ساخت ایندکس FAISS

```bash
# مسیر دیتاست را خودتان می‌دهید (فایل یا پوشه)
python scripts/embed_dataset.py --data /path/to/data.csv

# فقط دیدن گزارش چانکینگ بدون امبدینگ:
python scripts/embed_dataset.py --data /path/to/data.csv --preview

# تنظیم چانکینگ (پیش‌فرض‌ها: ۴ پاراگراف در هر چانک + ۱ پاراگراف هم‌پوشانی)
python scripts/embed_dataset.py --data data/data.csv --paragraphs 5 --overlap 2
```

خروجی در `storage/index` ذخیره می‌شود (`index.faiss` + `metadata.json`).
چانکر قبل از امبدینگ **پوشش کامل متن را راستی‌آزمایی می‌کند** — هیچ پاراگرافی
از چانک‌ها جا نمی‌ماند (گزارش را ببینید).

### مرحلهٔ ۲ — اجرای رابط کاربری

```bash
python run_app.py
# یا با پورت دلخواه:
python run_app.py --server.port 8502
```

سپس مرورگر: `http://localhost:8501`

---

## ۵) امکانات رابط کاربری

- **چت فارسی راست‌به‌چپ** با پاسخ جریانی (token به token)
- **مدیریت گفتگوها**: گفتگوی جدید، انتخاب، حذف — همه در SQLite ماندگارند
- **عمق حافظه**: اسلایدر تعداد پیام‌های گذشته که مدل می‌بیند (۰ تا ۲۰)
- **top-k و temperature** قابل تنظیم زنده
- **منابع آخرین پاسخ** در سایدبار: نام فایل + امتیاز شباهت + پیش‌نمایش متن
- **نوار وضعیت زنده**: بازیابی بافت، فراخوانی ابزار، دور تولید پاسخ

---

## ۶) سیستم ابزار (Tool Calling)

سیستم امبدینگ/جست‌وجو **به‌صورت ابزار در اختیار خودِ مدل** قرار گرفته:

- `search_knowledge_base(query, top_k)`: جست‌وجوی معنایی در FAISS — پرس‌وجو
  embed شده و مرتبط‌ترین چانک‌ها با نام فایل و امتیاز شباهت به مدل برمی‌گردند
- `get_knowledge_base_stats()`: آمار دانش‌نامه

جریان دو لایه: ابتدا همیشه بافت اولیه برای سوال کاربر بازیابی و در پرامپت
نشسته می‌شود (RAG خودکار)؛ اگر مدل به اطلاعات بیشتری نیاز داشته باشد، خودش
ابزار را با قالب hermes فراخوانی می‌کند:

```
<tool_call>
{"name": "search_knowledge_base", "arguments": {"query": "...", "top_k": 5}}
</tool_call>
```

نتیجهٔ ابزار در قالب بلوک `<tool_response> ... </tool_response>` به مدل برمی‌گردد و تولید پاسخ
ادامه می‌یابد (حداکثر ۳ بار فراخوانی — با `MAX_TOOL_CALLS` قابل تغییر).

**نکتهٔ فنی:** چون بلوک‌های فراخوانی ابزار مستقیماً از خروجی مدل parse
می‌شوند، **نیازی به اضافه‌کردن `--jinja` به docker-compose نیست** و روی هر
نسخهٔ llama.cpp کار می‌کند. بلوک‌های ابزار/استدلال هنگام نمایش به کاربر به‌صورت
زنده حذف می‌شوند (`StreamMaskFilter`) حتی اگر بین token ها شکسته شده باشند.

---

## ۷) ساختار پروژه

```
rag-system/
├── run_app.py                  # اجرای استریملیت
├── requirements.txt
├── .env.example                # نمونهٔ تنظیمات
├── data/                       # اینجا data.csv خود را بگذارید
├── storage/                    # خروجی: index.faiss + metadata.json + history.db
├── scripts/
│   ├── embed_dataset.py        # امبد دیتاست + ساخت ایندکس FAISS
│   ├── check_services.py       # بررسی سلامت سرویس‌ها
│   └── mock_server.py          # سرور آزمایشی برای تست بدون GPU
└── src/
    ├── config.py               # تنظیمات یکپارچه (‎.env‎ + پیش‌فرض‌ها)
    ├── core/
    │   ├── api_client.py       # LLMClient و EmbeddingClient (جدا از هم)
    │   ├── embedder.py         # EmbeddingService (batch + نرمال‌سازی)
    │   ├── chunker.py          # CSVLoader + ParagraphChunker (بدون افت متن)
    │   ├── vector_store.py     # VectorStore (FAISS + متادیتا)
    │   ├── retriever.py        # Retriever (امبد پرس‌وجو + جست‌وجو)
    │   ├── tools.py            # ToolRegistry (ابزارهای مدل)
    │   ├── tool_parsing.py     # parse فراخوانی ابزار + فیلتر جریانی
    │   ├── prompts.py          # پرامپت‌های فارسی + قالب بافت
    │   ├── history.py          # ChatHistory (SQLite)
    │   ├── rag_pipeline.py     # خط لولهٔ اصلی RAG
    │   ├── models.py           # ساختارهای داده (dataclass)
    │   └── exceptions.py       # خطاهای سفارشی
    └── ui/
        └── app.py              # رابط کاربری Streamlit
```

---

## ۸) تنظیمات (`.env`)

| کلید | پیش‌فرض | توضیح |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | آدرس سرویس مدل |
| `EMBEDDING_BASE_URL` | `http://localhost:8081/v1` | آدرس سرویس امبدینگ |
| `CHUNK_PARAGRAPHS` | `4` | تعداد پاراگراف هر چانک |
| `CHUNK_OVERLAP_PARAGRAPHS` | `1` | هم‌پوشانی پاراگرافی |
| `CHUNK_MAX_CHARS` | `3000` | سقف کاراکتر هر چانک |
| `PARAGRAPH_MAX_CHARS` | `1500` | سقف کاراکتر یک پاراگراف (قبل از ریزشگی) |
| `DEFAULT_TOP_K` | `5` | تعداد نتایج پیش‌فرض بازیابی |
| `MAX_CONTEXT_CHARS` | `6000` | سقف کاراکتر بافتِ ارسالی به مدل |
| `TEMPERATURE` / `MAX_TOKENS` | `0.3` / `1500` | پارامترهای تولید |
| `MAX_TOOL_CALLS` | `3` | حداکثر فراخوانی ابزار در یک پاسخ |
| `EMBED_BATCH_SIZE` | `16` | تعداد متن در هر درخواست امبدینگ |

مقادیر کامل در `.env.example`.

---

## ۹) رفع اشکال

| نشانه | راه‌حل |
|---|---|
| `اتصال برقرار نشد` | سرویس‌ها بالا نیستند → `docker compose up -d` و `python scripts/check_services.py` |
| `ایندکس FAISS ساخته نشده` | اول `embed_dataset.py` را اجرا کنید |
| خطای `context length` در پاسخ‌های بلند | در compose مقدار `-c` را بالاتر ببرید (مثلاً 16384) یا `MAX_CONTEXT_CHARS` را در ‎.env‎ کم کنید |
| جواب‌ها بی‌ربط‌اند | دیتاست را با `--preview` چک کنید؛ شاید چانک‌ها خیلی بزرگ/کوچک‌اند |
| تست بدون GPU | `python scripts/mock_server.py --port 8090` و تنظیم URL ها روی همان پورت |

---

## ۱۰) تست بدون سرویس واقعی

برای توسعهٔ آفلاین، سرور آزمایشی همهٔ endpoint ها را شبیه‌سازی می‌کند (همچنین
رفتار فراخوانی ابزار مدل را):

```bash
python scripts/mock_server.py --port 8090
# در فایل .env:
#   LLM_BASE_URL=http://localhost:8090/v1
#   EMBEDDING_BASE_URL=http://localhost:8090/v1
python scripts/embed_dataset.py --data data/data.csv
python run_app.py
```
