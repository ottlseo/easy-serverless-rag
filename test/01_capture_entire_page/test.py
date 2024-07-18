import fitz
from PIL import Image
from pdf2image import convert_from_path
import os

### 페이지에 이미지나 테이블이 있는지 확인합니다 ###
def has_image_or_table(page):
    image_list = page.get_images(full=True)
    has_images = len(image_list) > 0
    tables = page.find_tables()
    has_tables = len(tables.tables) > 0 
    return has_images or has_tables
    
### 특정 페이지의 스크린샷을 저장합니다 ###
def save_page_screenshot(pdf_path, page_num, output_folder):
    images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
    if images:
        image = images[0]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        image.save(f"{output_folder}/page_{page_num+1}.png", "PNG")
        print(f"Page {page_num+1} screenshot saved.")

### PDF를 처리하고 이미지나 테이블이 있는 페이지의 스크린샷을 저장합니다 ### 
def process_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        if has_image_or_table(page):
            save_page_screenshot(pdf_path, page_num, output_folder)
    doc.close()

#### 사용 예 ####
pdf_path = "../school_edu_guide.pdf"
output_folder = "output/"
process_pdf(pdf_path, output_folder)