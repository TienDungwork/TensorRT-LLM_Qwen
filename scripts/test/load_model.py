import argparse
import json
import re
import shutil
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)

VALID_LABELS = {"positive", "neutral", "negative"}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


PROMPT = """You are an expert in facial emotion analysis.

Classify the person's facial emotion in the image into exactly one of these labels:
- positive
- neutral
- negative

Return the result in valid JSON format:
{"label": "...", "confidence": "high|medium|low"}

Return JSON only, with no extra text."""


def classify_emotion(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {"label": "unknown", "confidence": "low"}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"label": "unknown", "confidence": "low"}


def interactive_mode() -> None:
    print("Nhap duong dan anh de test cam xuc. Nhap 'q' de thoat.")

    while True:
        image_path = input("\nDuong dan anh: ").strip()
        if image_path.lower() in {"q", "quit", "exit"}:
            print("Da thoat.")
            break

        if not image_path:
            print("Vui long nhap duong dan anh.")
            continue

        path = Path(image_path)
        if not path.is_file():
            print("Khong tim thay file anh.")
            continue

        try:
            result = classify_emotion(str(path))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "label": "error",
                        "confidence": "low",
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


def prepare_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in sorted(VALID_LABELS):
        label_dir = output_dir / label
        if label_dir.exists():
            shutil.rmtree(label_dir)
        label_dir.mkdir(parents=True, exist_ok=True)


def process_face_folder(image_dir: str, output_dir: str) -> None:
    src_dir = Path(image_dir)
    dst_dir = Path(output_dir)

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay thu muc anh: {src_dir}")

    prepare_output_dirs(dst_dir)

    image_paths = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    total = len(image_paths)
    results = []
    saved_high = 0

    if total == 0:
        print("Khong co anh nao trong thu muc nguon.")
        return

    for index, path in enumerate(image_paths, start=1):
        print(f"[{index}/{total}] {path.name}")
        try:
            result = classify_emotion(str(path))
        except Exception as exc:
            result = {"label": "error", "confidence": "low", "error": str(exc)}

        label = str(result.get("label", "")).lower()
        confidence = str(result.get("confidence", "")).lower()

        record = {
            "image": path.name,
            "label": label,
            "confidence": confidence,
        }
        if "error" in result:
            record["error"] = result["error"]
        results.append(record)

        if label in VALID_LABELS and confidence == "high":
            shutil.copyfile(path, dst_dir / label / path.name)
            saved_high += 1

    results_path = dst_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nTong anh: {total}")
    print(f"Anh duoc luu voi confidence high: {saved_high}")
    print(f"Ket qua chi tiet: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phan loai cam xuc anh bang Qwen2.5-VL.")
    parser.add_argument("--interactive", action="store_true", help="Test tung anh bang tay")
    parser.add_argument("--image-dir", default="face", help="Thu muc chua anh nguon")
    parser.add_argument("--output-dir", default="emotion", help="Thu muc luu ket qua")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    process_face_folder(args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
