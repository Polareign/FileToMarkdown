import sys
import os
import base64
from mistralai import Mistral

def FileTest(file):
    Accepted = ['.jpg', '.jpeg', '.png', '.pdf', '.docx']
    return os.path.splitext(file)[1].lower() in Accepted

def Mistral(file):
    try:
        with open(file, "rb") as file:
            return base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding file: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python File_Imput.py <file>")
        sys.exit(1)
    file = sys.argv[1]

    ext = os.path.splitext(file)[1].lower()
    base64_file = Mistral(file)
    if not base64_file:
        sys.exit(1)

    apikey = os.environ.get("MISTRALAPIKEY")
    if not apikey:
        print("Error: Please Imput MISTRALAPIKEY")
        sys.exit(1)

    client = Mistral(api_key=apikey)

    if ext in [".pdf", ".docx"]:
        doc_type = "document_url"
        mime = "application/pdf" if ext == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        url = f"data:{mime};base64,{base64_file}"
        doc_key = "document_url"
    else:
        doc_type = "image_url"
        mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        url = f"data:{mime};base64,{base64_file}"
        doc_key = "image_url"

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": doc_type,
            doc_key: url
        },
        include_image_base64=True
    )

    print(ocr_response)

    # Use python File_Imput.py testimage.png 