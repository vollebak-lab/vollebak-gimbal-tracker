import pypdf
import sys

def extract_text(pdf_path, txt_path):
    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Extraction complete.")

if __name__ == "__main__":
    extract_text(sys.argv[1], sys.argv[2])
