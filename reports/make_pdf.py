"""Genera el PDF de entrega a partir de reports/Entregable.md.

    python reports/make_pdf.py

Convierte el Markdown a HTML con el formato que exige la actividad (Arial 11 pt,
interlineado 1.5, márgenes de 2.5 cm) y lo imprime a PDF con Microsoft Edge en
modo headless (incluido en Windows; también acepta Chrome). El resultado queda en:

    reports/TC5061_Experimentacion_MLFlow_SalazarVictor.pdf
"""

import subprocess
import sys
from pathlib import Path

import markdown

REPORTS_DIR = Path(__file__).resolve().parent
SOURCE = REPORTS_DIR / "Entregable.md"
OUTPUT = REPORTS_DIR / "TC5061_Experimentacion_MLFlow_SalazarVictor.pdf"

# Formato exigido por la entrega: Arial/Times New Roman 11, interlineado 1.5
CSS = """
@page { size: Letter; margin: 2.5cm; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; line-height: 1.5; }
h1 { font-size: 15pt; } h2 { font-size: 13pt; } h3 { font-size: 11.5pt; }
img { max-width: 100%; }
code { font-family: Consolas, monospace; font-size: 9.5pt; }
pre { background: #f5f5f5; padding: 6px 10px; font-size: 9.5pt; line-height: 1.25; }
blockquote { color: #444; border-left: 3px solid #bbb; margin-left: 0; padding-left: 12px; }
a { color: #0b5394; }
"""

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for path in BROWSERS:
        if Path(path).exists():
            return path
    sys.exit("No se encontró Edge ni Chrome para imprimir el PDF.")


def main():
    body = markdown.markdown(
        SOURCE.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
    )
    html = f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'

    # El HTML intermedio vive junto al .md para que las rutas relativas de las
    # imágenes (mlflow_screenshots/...) resuelvan bien; se borra al final.
    html_path = REPORTS_DIR / "_entregable_tmp.html"
    html_path.write_text(html, encoding="utf-8")
    try:
        subprocess.run(
            [
                find_browser(),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={OUTPUT}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        html_path.unlink(missing_ok=True)

    print(f"PDF generado: {OUTPUT}")


if __name__ == "__main__":
    main()
