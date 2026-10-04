"""خطاهای سفارشی سیستم RAG — همه از یک پایه مشترک ارث می‌برند."""


class RAGSystemError(Exception):
    """پایهٔ تمام خطاهای پروژه."""


class ConfigurationError(RAGSystemError):
    """تنظیمات نامعتبر یا ناقص."""


class APIConnectionError(RAGSystemError):
    """عدم توانایی در اتصال به سرویس (مدل زبانی یا امبدینگ)."""


class APIResponseError(RAGSystemError):
    """پاسخ ناموفق یا نامعتبر از سرویس (کد 4xx/5xx یا ساختار خراب)."""


class DatasetError(RAGSystemError):
    """مشکل در خواندن یا اعتبارسنجی فایل دیتاست CSV."""


class IndexNotFoundError(RAGSystemError):
    """ایندکس FAISS هنوز ساخته یا بارگذاری نشده است."""
