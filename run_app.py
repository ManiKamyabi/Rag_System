"""
اجرای رابط کاربری Streamlit
============================
استفاده از ریشهٔ پروژه:

    python run_app.py

آرگومان‌های اضافی مستقیماً به streamlit پاس می‌شوند؛ مثلاً برای تغییر پورت:

    python run_app.py --server.port 8502
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        sys.exit("کتابخانه streamlit نصب نشده است. ابتدا اجرا کنید: pip install -r requirements.txt")

    sys.argv = ["streamlit", "run", str(ROOT / "src" / "ui" / "app.py")] + sys.argv[1:]
    sys.exit(streamlit_cli.main())
