
import fitz
from PIL import Image
import io
import os

def save_image(page, img_index, output_folder, min_width=100, min_height=100):
    """
    페이지에서 이미지를 추출하고 저장합니다.
    최소 너비와 높이보다 큰 이미지만 저장합니다.
    """
    img = page.get_images()[img_index]
    xref = img[0]
    base_image = page.parent.extract_image(xref)
    image_bytes = base_image["image"]
    
    # 이미지 데이터를 PIL Image 객체로 변환
    image = Image.open(io.BytesIO(image_bytes))
    
    # 이미지 크기 확인
    width, height = image.size
    if width < min_width or height < min_height:
        print(f"Image on page {page.number + 1} is too small. Skipping.")
        return
    
    # 이미지 저장 폴더 생성
    image_folder = os.path.join(output_folder, "image")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    # 이미지 저장
    image.save(f"{image_folder}/page{page.number + 1}_{img_index + 1}.png")
    print(f"Image from page {page.number + 1} saved. Size: {width}x{height}")

def save_table(page, table, table_index, output_folder, min_width=100, min_height=50):
    """
    테이블 영역의 스크린샷을 저장합니다.
    최소 너비와 높이보다 큰 테이블만 저장합니다.
    """
    # 테이블 영역 좌표 가져오기
    rect = table.bbox
    
    # 테이블 크기 계산
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    
    if width < min_width or height < min_height:
        print(f"Table on page {page.number + 1} is too small. Skipping.")
        return
    
    # 해당 영역의 픽셀맵 생성
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
    
    # 테이블 저장 폴더 생성
    table_folder = os.path.join(output_folder, "table")
    if not os.path.exists(table_folder):
        os.makedirs(table_folder)
    
    # 테이블 이미지 저장
    pix.save(f"{table_folder}/page{page.number + 1}_{table_index + 1}.png")
    print(f"Table from page {page.number + 1} saved. Size: {width}x{height}")

def process_pdf(pdf_path, output_folder, min_image_width=100, min_image_height=100, min_table_width=100, min_table_height=50):
    """PDF를 처리하고 특정 크기 이상의 이미지와 테이블의 스크린샷을 저장합니다."""
    doc = fitz.open(pdf_path)
    
    for page in doc:
        # 이미지 처리
        images = page.get_images()
        for img_index in range(len(images)):
            save_image(page, img_index, output_folder, min_image_width, min_image_height)
        
        # 테이블 처리
        tables = page.find_tables()
        for table_index, table in enumerate(tables):
            save_table(page, table, table_index, output_folder, min_table_width, min_table_height)
    
    doc.close()

# 사용 예
pdf_path = "../school_edu_guide.pdf"
output_folder = "output/"
process_pdf(pdf_path, output_folder, min_image_width=200, min_image_height=50, min_table_width=30, min_table_height=30)
