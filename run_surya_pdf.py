import os
import glob
import json
import argparse
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
from dataclasses import dataclass, asdict

# Giữ nguyên cấu trúc dữ liệu OCRBox giống logic hiện tại
@dataclass
class OCRBox:
    text: str
    bbox: list
    confidence: float = 0.0
    page: int = 1
    column_order: int = -1
    engine: str = 'Surya_v0.17'

def _to_native(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))

def process_single_pdf(pdf_path, output_dir, det_model, det_processor, rec_model, rec_processor):
    from surya.ocr import run_ocr
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ Error: Không tìm thấy file {pdf_path}")
        return

    book_name = pdf_path.stem
    book_out_dir = Path(output_dir) / book_name
    book_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📖 Đang xử lý sách: {pdf_path.name}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        
        # Render trang PDF thành ảnh RGB
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), colorspace=fitz.csRGB, alpha=False)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        print(f"  ⏳ Đang chạy OCR trang {page_num + 1}/{total_pages}...")
        
        # Chạy dự đoán bằng Surya v0.17.x API
        # [[]] là biến truyền ngôn ngữ trống để model tự đoán, bạn có thể thay bằng [['vi']] nếu cần
        predictions = run_ocr([pil_img], [[]], det_model, det_processor, rec_model, rec_processor)
        
        boxes = []
        page_text_lines = []
        
        # Lấy từng dòng text và tọa độ từ kết quả
        for text_line in predictions[0].text_lines:
            text = text_line.text.strip()
            if not text:
                continue
                
            box = OCRBox(
                text=text,
                bbox=text_line.bbox,
                confidence=getattr(text_line, 'confidence', 0.9),
                page=page_num + 1
            )
            boxes.append(box)
            page_text_lines.append(text)
            
        # --- LƯU KẾT QUẢ ---
        json_path = book_out_dir / f"page_{page_num + 1:04d}.json"
        data = []
        for b in boxes:
            d = asdict(b)
            d['bbox'] = np.array(d['bbox'], float).flatten().tolist()
            data.append(d)
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_to_native)
            
        txt_path = book_out_dir / f"page_{page_num + 1:04d}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(page_text_lines))
            
    print(f"✅ Hoàn thành sách {book_name}. Kết quả lưu tại: {book_out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Script chạy Surya OCR (phiên bản v0.17) cho hàng loạt file PDF")
    parser.add_argument("input_path", type=str, help="Đường dẫn đến một file PDF hoặc THƯ MỤC chứa nhiều file PDF")
    parser.add_argument("--out", type=str, default="output", help="Thư mục lớn chứa kết quả OCR (mặc định: output)")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    
    pdf_files = []
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        pdf_files.append(input_path)
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf"))
    else:
        print("❌ Error: input_path không hợp lệ hoặc thư mục không có file PDF.")
        return

    if not pdf_files:
        print(f"❌ Không tìm thấy file PDF nào trong: {input_path}")
        return

    print(f"📚 Đã tìm thấy {len(pdf_files)} file PDF cần xử lý.")

    # Khởi tạo models PyTorch thuần túy (Không dùng Docker)
    print(f"🚀 Khởi tạo models Surya OCR v0.17.x (PyTorch Native)...")
    from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor

    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()

    for pdf in pdf_files:
        process_single_pdf(pdf, args.out, det_model, det_processor, rec_model, rec_processor)

if __name__ == "__main__":
    main()
