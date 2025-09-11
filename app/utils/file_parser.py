from PyPDF2 import PdfReader
import docx

def parse_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def parse_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def parse_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return parse_pdf(file_path)
    elif file_path.endswith(".docx"):
        return parse_docx(file_path)
    else:
        raise ValueError("Unsupported file format")
