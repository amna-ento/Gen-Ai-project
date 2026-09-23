import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_words(xml_path: str) -> list[dict]:
    root = ET.parse(xml_path).getroot()

    words = []

    for element in root.findall("w"):
        if element.get("punc") == "true":
            continue

        text = (element.text or "").strip()

        if not text:
            continue

        start = float(element.get("starttime", "0"))
        end = float(element.get("endtime", "0"))

        words.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    return words


def create_reference(xml_paths: list[str], output_path: str) -> dict:
    all_words = []

    for xml_path in xml_paths:
        words = extract_words(xml_path)
        all_words.extend(words)

    all_words.sort(key=lambda word: (word["start"], word["end"]))

    transcript = " ".join(word["text"] for word in all_words)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(transcript, encoding="utf-8")

    return {
        "output_path": str(output),
        "xml_files": len(xml_paths),
        "reference_words": len(all_words),
        "first_word": all_words[0]["text"] if all_words else None,
        "last_word": all_words[-1]["text"] if all_words else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create a ground-truth reference transcript from AMI word annotations."
    )

    parser.add_argument(
        "--xml-dir",
        required=True,
        help="Directory containing AMI .words.xml files.",
    )

    parser.add_argument(
        "--meeting-id",
        required=True,
        help="AMI meeting ID.",
    )

    parser.add_argument(
        "--output-meeting-id",
        required=True,
        help="Project meeting ID for the output path.",
    )

    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)

    xml_paths = sorted(
        xml_dir.glob(f"{args.meeting_id}.?.words.xml")
    )

    if not xml_paths:
        raise FileNotFoundError(
            f"No word annotation files found for meeting: {args.meeting_id}"
        )

    output_path = (
        Path("data")
        / "meetings"
        / args.output_meeting_id
        / "reference"
        / "reference.txt"
    )

    result = create_reference(
        [str(path) for path in xml_paths],
        str(output_path),
    )

    print(result)


if __name__ == "__main__":
    main()