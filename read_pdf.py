"""Extract text from the soiling PDF for reference."""
import PyPDF2

reader = PyPDF2.PdfReader('soiling related data.pdf')
print(f'Total pages: {len(reader.pages)}')

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        print(f'\n{"="*60}')
        print(f'Page {i+1}')
        print(f'{"="*60}')
        print(text)
