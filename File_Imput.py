import sys
import os
import base64
import re
from mistralai import Mistral
import openai
import markdown

print("[SCRIPT] Script loaded")

def file_is_accepted(file):
    accepted = ['.jpg', '.jpeg', '.png', '.pdf', '.docx']
    return os.path.splitext(file)[1].lower() in accepted

def encode_file_base64(filepath):
    print("[STEP] Encoding file to base64")
    try:
        with open(filepath, "rb") as f:
            print("[SUCCESS] File read and encoded")
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Failed to encode file: {e}")
        return None

def describe_image(base64_str, api_key):
    print("[STEP] Describing image using OpenAI GPT-4 Vision")
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image in markdown-friendly prose."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
            ]}
        ],
        max_tokens=300
    )
    print("[SUCCESS] Received image description")
    return response.choices[0].message['content'].strip()

def process_with_mistral_ocr(filepath, base64_file, mistral_api_key):
    print("[STEP] Preparing OCR request for Mistral")
    ext = os.path.splitext(filepath)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }.get(ext)

    doc_type = "document_url" if ext in [".pdf", ".docx"] else "image_url"
    doc_key = "document_url" if doc_type == "document_url" else "image_url"
    url = f"data:{mime};base64,{base64_file}"

    client = Mistral(api_key=mistral_api_key)
    print("[STEP] Sending OCR request to Mistral...")
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={ "type": doc_type, doc_key: url },
            include_image_base64=True
        )
        print("[SUCCESS] OCR completed")
        return ocr_response
    except Exception as e:
        print(f"[ERROR] OCR request failed: {repr(e)}")
        sys.exit(1)

def replace_images_with_descriptions(markdown_text, image_data_list, openai_api_key):
    print("[STEP] Adding AI descriptions under ![img-#.jpeg] tags")
    def replace(match):
        idx = int(match.group(1))
        if idx < len(image_data_list):
            base64_image = image_data_list[idx]
            description = describe_image(base64_image, openai_api_key)
            return f"{match.group(0)}\n\n**Image Description:** {description}\n"
        return match.group(0)

    return re.sub(r"!\[img-(\d+)\.jpeg\]\(img-\d+\.jpeg\)", replace, markdown_text)

if __name__ == "__main__":
    print("[MAIN] Starting file processing")

    if len(sys.argv) < 2:
        print("[ERROR] Usage: python File_Imput.py <file>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"[INPUT] Filepath received: {filepath}")

    if not file_is_accepted(filepath):
        print("[ERROR] Unsupported file type")
        sys.exit(1)

    mistral_api_key = os.environ.get("MISTRALAPIKEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not mistral_api_key or not openai_api_key:
        print("[ERROR] Missing API keys in environment variables")
        sys.exit(1)

    base64_file = encode_file_base64(filepath)
    if not base64_file:
        sys.exit(1)

    ocr_response = process_with_mistral_ocr(filepath, base64_file, mistral_api_key)

    # Extract markdown from all pages
    markdown_text = "\n\n".join(page.markdown for page in ocr_response.pages)
    image_data_list = []
    print("[DEBUG] Checking for images in OCR response...")
    for page in ocr_response.pages:
        if hasattr(page, 'images') and page.images:
            print(f"[DEBUG] Found {len(page.images)} images on this page")
            for img in page.images:
                print(f"[DEBUG] Image object type: {type(img)}")
                print(f"[DEBUG] Image attributes: {dir(img)}")
                if hasattr(img, 'image_base64') and img.image_base64:
                    print("[DEBUG] Adding image to list")
                    image_data_list.append(img.image_base64)
                elif hasattr(img, 'data'):
                    print("[DEBUG] Found img.data instead of img.image_base64")
                    image_data_list.append(img.data)
                elif hasattr(img, 'content'):
                    print("[DEBUG] Found img.content instead of img.image_base64")
                    image_data_list.append(img.content)
                else:
                    print(f"[DEBUG] Image found but no image_base64 data. Available attributes: {[attr for attr in dir(img) if not attr.startswith('_')]}")
        else:
            print("[DEBUG] No images found on this page")
    print(f"[DEBUG] Total images extracted: {len(image_data_list)}")

    if not markdown_text.strip():
        print("[ERROR] OCR returned no markdown")
        sys.exit(1)


    if image_data_list:
        print(f"[STEP] Processing {len(image_data_list)} images for descriptions")
        markdown_with_descriptions = replace_images_with_descriptions(markdown_text, image_data_list, openai_api_key)
    else:
        print("[INFO] No images found to describe")
        markdown_with_descriptions = markdown_text

    # Save markdown
    with open("output.md", "w", encoding="utf-8") as f:
        f.write(markdown_with_descriptions)
    print("[OUTPUT] Markdown saved to output.md")

    # Save HTML
    html_content = markdown.markdown(markdown_with_descriptions)
    with open("output.html", "w", encoding="utf-8") as f:
        f.write(f"<html><body>{html_content}</body></html>")
    print("[OUTPUT] HTML saved to output.html")

    print("[DONE] Processing complete. Open output.md or output.html to view the result.")
