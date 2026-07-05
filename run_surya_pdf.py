import os
import glob
import json
import argparse
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
from dataclasses import dataclass, asdict
import re
import html

# Giữ nguyên cấu trúc dữ liệu OCRBox giống logic hiện tại
@dataclass
class OCRBox:
    text: str
    bbox: list
    confidence: float = 0.0
    page: int = 1
    column_order: int = -1
    engine: str = 'Surya_OCR'

def _to_native(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))

def _page_output_paths(book_out_dir, page_num):
    page_idx = page_num + 1
    return (
        book_out_dir / f"page_{page_idx:04d}.json",
        book_out_dir / f"page_{page_idx:04d}.txt",
    )

def _page_outputs_exist(book_out_dir, page_num):
    json_path, txt_path = _page_output_paths(book_out_dir, page_num)
    return json_path.exists() and txt_path.exists()

def process_single_pdf_v20(pdf_path, output_dir, rec_predictor):
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
    remaining_pages = [
        page_num for page_num in range(total_pages)
        if not _page_outputs_exist(book_out_dir, page_num)
    ]
    skipped_pages = total_pages - len(remaining_pages)

    if skipped_pages:
        print(f"  ⏭️  Bỏ qua {skipped_pages}/{total_pages} trang đã có output.")
    if not remaining_pages:
        print(f"✅ Bỏ qua sách {book_name}: tất cả trang đã có output tại {book_out_dir}")
        doc.close()
        return
    
    for page_num in remaining_pages:
        page = doc.load_page(page_num)
        
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), colorspace=fitz.csRGB, alpha=False)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        print(f"  ⏳ Đang chạy OCR trang {page_num + 1}/{total_pages} (API v0.20)...")
        
        predictions = rec_predictor([pil_img], full_page=True)
        boxes = []
        page_text_lines = []
        
        page_result = predictions[0]
        
        for block in page_result.blocks:
            if getattr(block, 'skipped', False) or getattr(block, 'error', False):
                continue
                
            text = html.unescape(re.sub(r'<[^>]+>', '', getattr(block, 'html', ''))).strip()
            if not text:
                continue
                
            conf = getattr(block, 'confidence', None)
            
            box = OCRBox(
                text=text,
                bbox=block.bbox,
                confidence=conf if conf is not None else 0.9,
                page=page_num + 1,
                column_order=getattr(block, 'reading_order', -1),
                engine='Surya_v0.20'
            )
            boxes.append(box)
            page_text_lines.append(text)
            
        json_path, txt_path = _page_output_paths(book_out_dir, page_num)
        data = []
        for b in boxes:
            d = asdict(b)
            d['bbox'] = np.array(d['bbox'], float).flatten().tolist()
            data.append(d)
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_to_native)
            
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(page_text_lines))
            
    doc.close()
    print(f"✅ Hoàn thành sách {book_name}. Kết quả lưu tại: {book_out_dir}")

def process_single_pdf_v17(pdf_path, output_dir, rec_predictor, det_predictor, TaskNames):
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
    remaining_pages = [
        page_num for page_num in range(total_pages)
        if not _page_outputs_exist(book_out_dir, page_num)
    ]
    skipped_pages = total_pages - len(remaining_pages)

    if skipped_pages:
        print(f"  ⏭️  Bỏ qua {skipped_pages}/{total_pages} trang đã có output.")
    if not remaining_pages:
        print(f"✅ Bỏ qua sách {book_name}: tất cả trang đã có output tại {book_out_dir}")
        doc.close()
        return
    
    for page_num in remaining_pages:
        page = doc.load_page(page_num)
        
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), colorspace=fitz.csRGB, alpha=False)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        print(f"  ⏳ Đang chạy OCR trang {page_num + 1}/{total_pages} (API v0.17)...")
        
        predictions = rec_predictor(
            [pil_img],
            task_names=[TaskNames.ocr_with_boxes],
            det_predictor=det_predictor,
            highres_images=[pil_img]
        )
        
        boxes = []
        page_text_lines = []
        
        for text_line in predictions[0].text_lines:
            text = text_line.text.strip()
            if not text:
                continue
                
            box = OCRBox(
                text=text,
                bbox=text_line.bbox,
                confidence=getattr(text_line, 'confidence', 0.9),
                page=page_num + 1,
                engine='Surya_v0.17'
            )
            boxes.append(box)
            page_text_lines.append(text)
            
        json_path, txt_path = _page_output_paths(book_out_dir, page_num)
        data = []
        for b in boxes:
            d = asdict(b)
            d['bbox'] = np.array(d['bbox'], float).flatten().tolist()
            data.append(d)
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_to_native)
            
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(page_text_lines))
            
    doc.close()
    print(f"✅ Hoàn thành sách {book_name}. Kết quả lưu tại: {book_out_dir}")

def process_single_pdf_v16(pdf_path, output_dir, det_model, det_processor, rec_model, rec_processor, run_ocr):
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
    remaining_pages = [
        page_num for page_num in range(total_pages)
        if not _page_outputs_exist(book_out_dir, page_num)
    ]
    skipped_pages = total_pages - len(remaining_pages)

    if skipped_pages:
        print(f"  ⏭️  Bỏ qua {skipped_pages}/{total_pages} trang đã có output.")
    if not remaining_pages:
        print(f"✅ Bỏ qua sách {book_name}: tất cả trang đã có output tại {book_out_dir}")
        doc.close()
        return
    
    for page_num in remaining_pages:
        page = doc.load_page(page_num)
        
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), colorspace=fitz.csRGB, alpha=False)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        print(f"  ⏳ Đang chạy OCR trang {page_num + 1}/{total_pages} (API v0.16)...")
        
        predictions = run_ocr([pil_img], [[]], det_model, det_processor, rec_model, rec_processor)
        
        boxes = []
        page_text_lines = []
        
        for text_line in predictions[0].text_lines:
            text = text_line.text.strip()
            if not text:
                continue
                
            box = OCRBox(
                text=text,
                bbox=text_line.bbox,
                confidence=getattr(text_line, 'confidence', 0.9),
                page=page_num + 1,
                engine='Surya_v0.16'
            )
            boxes.append(box)
            page_text_lines.append(text)
            
        json_path, txt_path = _page_output_paths(book_out_dir, page_num)
        data = []
        for b in boxes:
            d = asdict(b)
            d['bbox'] = np.array(d['bbox'], float).flatten().tolist()
            data.append(d)
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_to_native)
            
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(page_text_lines))
            
    doc.close()
    print(f"✅ Hoàn thành sách {book_name}. Kết quả lưu tại: {book_out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Script chạy Surya OCR đa phiên bản")
    parser.add_argument("input_path", type=str, help="Đường dẫn đến file PDF hoặc THƯ MỤC chứa file PDF")
    parser.add_argument("--out", type=str, default="output", help="Thư mục chứa kết quả")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    
    pdf_files = []
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        pdf_files.append(input_path)
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ Không tìm thấy file PDF nào trong: {input_path}")
        return

    print(f"📚 Đã tìm thấy {len(pdf_files)} file PDF.")

    API_VERSION = None
    
    try:
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
        API_VERSION = "0.20"
        print("🚀 Đang khởi tạo models Surya OCR (API v0.20)...")
        manager = SuryaInferenceManager()
        rec_predictor = RecognitionPredictor(manager)
        
        for pdf in pdf_files:
            process_single_pdf_v20(pdf, args.out, rec_predictor)
            
    except ImportError:
        try:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.common.surya.schema import TaskNames
            API_VERSION = "0.17"
            print("🚀 Đang khởi tạo models Surya OCR (API v0.17)...")
            foundation_predictor = FoundationPredictor()
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor(foundation_predictor)
            
            for pdf in pdf_files:
                process_single_pdf_v17(pdf, args.out, rec_predictor, det_predictor, TaskNames)
                
        except ImportError:
            try:
                from surya.ocr import run_ocr
                from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
                from surya.model.recognition.model import load_model as load_rec_model
                from surya.model.recognition.processor import load_processor as load_rec_processor
                API_VERSION = "0.16"
                print("🚀 Đang khởi tạo models Surya OCR (API v0.16)...")
                det_processor, det_model = load_det_processor(), load_det_model()
                rec_model, rec_processor = load_rec_model(), load_rec_processor()
                
                for pdf in pdf_files:
                    process_single_pdf_v16(pdf, args.out, det_model, det_processor, rec_model, rec_processor, run_ocr)
                    
            except ImportError as e:
                print(f"❌ Lỗi: {e}")
                print("❌ Môi trường hiện tại không thể load module của surya-ocr.")
                print("👉 VUI LÒNG CHẠY LỆNH SAU TRONG KAGGLE ĐỂ SỬA LỖI:")
                print("!pip install --upgrade surya-ocr==0.20.0")
                return

if __name__ == "__main__":
    main()
