import fitz
def extracted_pdf (file_path):
    doc = fitz.open(file_path)
    full_text = ""
    metadata = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            full_text += text + "\n" 
            metadata.append({
                'page' : page_number,
                'text' : text.strip()
            })
        
        doc.close()
        return full_text.strip(), metadata  