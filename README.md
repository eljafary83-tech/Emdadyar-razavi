# امدادیار رضوی

دستیار آموزشی فارسی جمعیت هلال احمر خراسان رضوی، با معماری یکپارچهٔ **RAG + LLM**. پیش از هر تولید، برنامه فقط در PDFهای `knowledge_base/` جست‌وجو می‌کند؛ در صورت عبور نتایج از آستانهٔ شباهت، Context همراه با نام فایل و صفحه به Provider مدل زبانی داده می‌شود.

## قابلیت‌ها

- پاسخ مستقیم فارسی به پرسش‌ها، همراه با citation منبع و صفحه.
- سناریوی آموزشی ساختاریافته برای موقعیت، شهرستان، فصل و حادثهٔ انتخابی.
- سؤال‌های ارزیابی چهارگزینه‌ای، پاسخ صحیح و توضیح مبتنی بر منبع.
- تشخیص خودکار تغییر PDFها با SHA-256، قفل فایل و بازسازی atomic index در اجرای طولانی Streamlit.
- fallback قطعی برای درخواست‌های بدون منبع مرتبط؛ API هرگز در این وضعیت فراخوانی نمی‌شود.

## معماری

```text
app.py / Streamlit UI
  -> services.AssistantService
     -> rag.Retriever (threshold + citations)
        -> rag.PDFLoader -> FAISS + JSON metadata
     -> prompts/*.txt
     -> llm.LLMProvider -> llm.ChatCompletionsClient
```

فقط یک مسیر index وجود دارد: `data/vector_store/`. metadata و manifest به JSON ذخیره می‌شوند، checksum دارند و پروژه هیچ‌گاه pickle یا `allow_dangerous_deserialization` استفاده نمی‌کند.

## نصب و اجرا

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

در اجرای نخست، مدل embedding چندزبانه دریافت و index ساخته می‌شود. برای اجرا بدون مدل زبانی، متغیرهای LLM را خالی بگذارید؛ retrieval انجام می‌شود اما برنامه به‌جای crash پیام کنترل‌شدهٔ تنظیم‌نبودن سرویس نمایش می‌دهد.

## پیکربندی

```dotenv
LLM_API_URL=
LLM_API_KEY=
LLM_MODEL=
LLM_TIMEOUT_SECONDS=30
RETRIEVAL_THRESHOLD=0.42
RETRIEVAL_TOP_K=5
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=900
CHUNK_OVERLAP=140
MIN_CHUNK_LENGTH=60
```

API client از قرارداد generic/OpenAI-compatible chat completions استفاده می‌کند. Provider در `llm/provider.py` جداست؛ برای تعویض سرویس فقط Provider یا Client را پیاده‌سازی/تزریق کنید، نه UI را. هیچ کلید یا URL واقعی در Repository قرار ندهید.

`RETRIEVAL_THRESHOLD` باید با پرسش‌های واقعی فارسی کالیبره شود. مقدار پیش‌فرض، نقطهٔ شروع است نه تضمین کیفیت برای هر مجموعه‌داده.

## پایگاه دانش و index

فقط PDFهای معتبر و مجاز را در `knowledge_base/` قرار دهید. PDF رمزگذاری‌شده، خراب، بزرگ‌تر از ۲۵MB، بیش از ۵۰۰ صفحه یا بیش از ۱۰۰ فایل پذیرفته نمی‌شود. manifest index شامل fingerprint محتوای PDFها، نام مدل embedding، بعد بردار، تنظیمات chunking و checksum فایل‌های index است. پیش از هر retrieval، تغییر PDF، افزوده/حذف‌شدن PDF یا تغییر مدل/تنظیمات باعث بازسازی index می‌شود.

## امنیت

- خروجی LLM و محتوای PDF قبل از نمایش HTML-escape می‌شوند.
- خروجی LLM باید JSON ساخت‌یافته با source-idهای بازیابی‌شده باشد؛ citation ساختگی یا پاسخ نامعتبر رد می‌شود.
- metadata index JSON و checksum‌دار است؛ فایل pickle بارگذاری نمی‌شود.
- timeout، اتصال، HTTP، authentication، rate-limit، پاسخ نامعتبر و پاسخ خالی API به خطای کنترل‌شده تبدیل می‌شوند.
- برای استقرار عمومی، HTTPS، authentication، rate limiting و permission محدود روی `knowledge_base/` و `data/` را در لایهٔ زیرساخت اعمال کنید.

## تست

```bash
python -B -m pytest -q
```

تست‌ها PDF loading، chunking، retrieval threshold، fallback، classifier، API client و خطاهای آن، هر سه نوع خروجی service و تشخیص stale index را پوشش می‌دهند.
