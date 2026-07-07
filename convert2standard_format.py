import argparse
import json
import re
import shutil
from pathlib import Path


PAGE_RE = re.compile(r"^page_(\d{4})\.(txt|json)$")
UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def normalize_book_id(book_name: str) -> str:
    book_id = "_".join(book_name.split())
    book_id = UNSAFE_RE.sub("", book_id)
    if book_id == "越南汉文燕行文献集成_第1册":
        return "越南汉文燕行文献集成_第1册-1"
    return book_id


def collect_pages(book_dir: Path) -> list[int]:
    txt_pages = {}
    json_pages = {}

    for path in book_dir.iterdir():
        if not path.is_file():
            continue
        match = PAGE_RE.match(path.name)
        if not match:
            continue

        page_number = int(match.group(1))
        suffix = match.group(2)
        if suffix == "txt":
            txt_pages[page_number] = path
        else:
            json_pages[page_number] = path

    if not txt_pages and not json_pages:
        raise ValueError(f"No page_NNNN txt/json files found in {book_dir}")

    missing_txt = sorted(set(json_pages) - set(txt_pages))
    missing_json = sorted(set(txt_pages) - set(json_pages))
    if missing_txt or missing_json:
        details = []
        if missing_txt:
            details.append(f"missing txt for pages: {format_page_list(missing_txt)}")
        if missing_json:
            details.append(f"missing json for pages: {format_page_list(missing_json)}")
        raise ValueError(f"Incomplete page pairs in {book_dir}: {'; '.join(details)}")

    return sorted(txt_pages)


def format_page_list(pages: list[int], limit: int = 10) -> str:
    shown = ", ".join(f"{page:04d}" for page in pages[:limit])
    if len(pages) > limit:
        shown += f", ... ({len(pages)} total)"
    return shown


def write_manifest_row(manifest_file, row: dict) -> None:
    manifest_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    manifest_file.write("\n")


def replace_text_key_with_data(value):
    if isinstance(value, list):
        return [replace_text_key_with_data(item) for item in value]
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            target_key = "data" if key == "text" else key
            converted[target_key] = replace_text_key_with_data(item)
        return converted
    return value


def write_standard_json(source_json: Path, dest_json: Path) -> None:
    raw_json = json.loads(source_json.read_text(encoding="utf-8"))
    standard_json = replace_text_key_with_data(raw_json)
    dest_json.write_text(
        json.dumps(standard_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def convert_book(
    book_dir: Path,
    run_dir: Path,
    manifest_file,
    model: str,
    run_id: str,
    source_pdf_template: str,
    book_id_override: str | None,
) -> dict:
    book_id = book_id_override or normalize_book_id(book_dir.name)
    pages = collect_pages(book_dir)

    pages_text_dir = run_dir / "pages_text" / book_id
    pages_json_dir = run_dir / "pages_json" / book_id
    books_text_dir = run_dir / "books_text"
    logs_dir = run_dir / "logs" / book_id

    pages_text_dir.mkdir(parents=True, exist_ok=True)
    pages_json_dir.mkdir(parents=True, exist_ok=True)
    books_text_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    source_pdf = source_pdf_template.format(book_name=book_dir.name, book_id=book_id)
    ok_count = 0
    blank_count = 0
    book_text_path = books_text_dir / f"{book_id}.txt"

    with book_text_path.open("w", encoding="utf-8", newline="\n") as book_text_file:
        for page_number in pages:
            filename = f"page_{page_number:04d}"
            source_txt = book_dir / f"{filename}.txt"
            source_json = book_dir / f"{filename}.json"
            dest_txt = pages_text_dir / source_txt.name
            dest_json = pages_json_dir / source_json.name

            text = source_txt.read_text(encoding="utf-8")
            shutil.copy2(source_txt, dest_txt)
            write_standard_json(source_json, dest_json)

            status = "ok" if text else "blank"
            if status == "ok":
                ok_count += 1
            else:
                blank_count += 1

            if page_number != pages[0]:
                book_text_file.write("\n")
            book_text_file.write(f"=== {filename} ===\n")
            book_text_file.write(text)
            book_text_file.write("\n")

            row = {
                "model": model,
                "run_id": run_id,
                "book_id": book_id,
                "source_pdf": source_pdf,
                "page_number": page_number,
                "status": status,
                "text_path": str(dest_txt),
                "json_path": str(dest_json),
                "char_count": len(text),
                "wall_time_seconds": 0.0,
                "error": None,
            }
            write_manifest_row(manifest_file, row)

    return {
        "book_id": book_id,
        "pages": len(pages),
        "ok": ok_count,
        "blank": blank_count,
        "book_text_path": book_text_path,
    }


def convert_all(
    input_dir: Path,
    output_root: Path,
    model: str,
    run_id: str,
    source_pdf_template: str,
    book_id_override: str | None,
) -> list[dict]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    book_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not book_dirs:
        raise ValueError(f"No book folders found in input folder: {input_dir}")
    if book_id_override and len(book_dirs) != 1:
        raise ValueError("--book-id can only be used when input contains exactly one book folder")

    run_dir = output_root / "ocr_runs" / model / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.jsonl"

    summaries = []
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_file:
        for book_dir in book_dirs:
            summaries.append(
                convert_book(
                    book_dir=book_dir,
                    run_dir=run_dir,
                    manifest_file=manifest_file,
                    model=model,
                    run_id=run_id,
                    source_pdf_template=source_pdf_template,
                    book_id_override=book_id_override,
                )
            )

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Surya OCR output/ folders to the standard OCR run format."
    )
    parser.add_argument("input_dir", nargs="?", default="output", help="Folder containing book output folders")
    parser.add_argument("--out", default="formated_output", help="Formatted output root folder")
    parser.add_argument("--model", default="surya_ocr", help="Standard model name")
    parser.add_argument("--run-id", default="surya_ocr-full-001", help="Standard run id")
    parser.add_argument("--book-id", default=None, help="Override book_id; only valid for one input book")
    parser.add_argument(
        "--source-pdf-template",
        default="books/{book_name}.pdf",
        help="Manifest source_pdf template; supports {book_name} and {book_id}",
    )
    args = parser.parse_args()

    summaries = convert_all(
        input_dir=Path(args.input_dir),
        output_root=Path(args.out),
        model=args.model,
        run_id=args.run_id,
        source_pdf_template=args.source_pdf_template,
        book_id_override=args.book_id,
    )

    for summary in summaries:
        print(
            f"{summary['book_id']}: {summary['pages']} pages, "
            f"{summary['ok']} ok, {summary['blank']} blank -> {summary['book_text_path']}"
        )


if __name__ == "__main__":
    main()
