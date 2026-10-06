import csv
from pathlib import Path
from pypdf import PdfReader

output = Path("data/data.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", encoding="utf-8", newline="") as out:
    writer = csv.writer(
        out,
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )

    writer.writerow(["filename", "content"])

    for pdf_path in sorted(Path("pdfs").glob("*.pdf")):
        print(f"در حال پردازش: {pdf_path.name}")

        try:
            reader = PdfReader(pdf_path)

            pages = [
                (page.extract_text() or "").strip()
                for page in reader.pages
            ]

            content = "\n\n".join(
                page for page in pages if page
            )

            writer.writerow([
                pdf_path.name,
                content
            ])

        except Exception as e:
            print(f"خطا در {pdf_path.name}: {e}")

print(f"دیتاست ساخته شد: {output}")