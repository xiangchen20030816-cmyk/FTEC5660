#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."

#读取env内容
def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


#扫描文件夹返回合适图片
def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

#把本地图片转成 base64 data URL让模型操作
def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    from langchain_core.output_parsers import JsonOutputParser

    #env -》 key
    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
    )

    parser = JsonOutputParser()
    
    # context + picture 
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert at reading supermarket receipts and extracting payment information.

                Your task is to extract exactly three pieces of information from each receipt.

                1. paid
                - The final amount actually paid after ROUNDING.
                - Use the final payment amount shown on the receipt.
                - This is usually the amount associated with the final payment method
                (for example, OCTOPUS, CASH, CARD, etc.).

                2. subtotal
                - The receipt's SUBTOTAL after discounts but before ROUNDING.
                - Use the value explicitly shown as SUBTOTAL on the receipt.
                - Do not reconstruct SUBTOTAL from item prices if an explicit SUBTOTAL is shown.

                3. discounts
                - Extract every actual discount, promotion, coupon, or deduction line.
                - Return each discount as a positive number in a list.
                - Each value in the list must correspond to one actual deduction from the bill.
                - Discounts may include OFF, SAVE, DISCOUNT, COUPON, PROMOTION,
                APP discounts, member discounts, percentage discounts,
                packaging-damage discounts, and similar promotional deductions.

                Important rules for identifying discounts:
                - Only include actual deductions from the bill.
                - Do NOT include ROUNDING. ROUNDING is not a discount.
                - Do NOT include normal positive item prices.
                - Do NOT include SUBTOTAL or the final payment amount.
                - A normal positive item price must never be treated as a discount.
                - Every extracted discount must be supported by an actual deduction line
                or its corresponding promotion information on the receipt.

                Cross-check each discount carefully:
                - Read both the promotion description and the printed negative deduction amount.
                - When the printed negative deduction is clear, use that monetary value.
                - Use the promotion description as additional evidence to verify the deduction.
                - If the printed deduction amount is visually ambiguous, use the corresponding
                promotion description to resolve the ambiguity.
                - For example, if the promotion says "Save $6" and the nearby deduction is
                visually ambiguous between -$5.00 and -$6.00, extract 6.00.
                - If a promotion description contains a monetary value, compare it carefully
                with the nearby printed deduction amount.
                - If a discount is described as a percentage, return the actual monetary
                deduction shown on the receipt, not the percentage.
                - Pay special attention to visually similar digits.
                - Never guess an ambiguous monetary value without checking the corresponding
                promotion description and surrounding receipt context.
                - If the description and the deduction appear inconsistent, re-read both
                carefully before deciding.
                - Do not infer a discount from a normal positive item price.
                - Do not omit or duplicate any discount line.

                Before returning the result, verify:
                - paid is the final amount after ROUNDING.
                - subtotal is the explicit SUBTOTAL before ROUNDING.
                - discounts contains only real discounts/promotions/coupons.
                - ROUNDING does not appear in discounts.
                - No normal item price appears in discounts.
                - Each discount appears exactly once.
                - Every value in discounts is supported by the receipt.

                For example:

                If the receipt contains:

                5% OFF: -5.39
                SUBTOTAL: 102.31
                ROUNDING: -0.01
                OCTOPUS: 102.30

                Return:

                {{
                    "paid": 102.30,
                    "subtotal": 102.31,
                    "discounts": [5.39]
                }}

                Output rules:
                - Return only one valid JSON object.
                - Use numeric values only.
                - All values inside "discounts" must be positive numbers.
                - If there are no discounts, return an empty list: [].
                - Do not include any additional fields.
                - Do not include explanations.
                - Do not include HK$ or any currency symbol.
                - Do not include markdown or code fences.
                - Do not include percentages.
                - Do not include item counts.
                """
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            "Read this receipt carefully and extract the final amount paid, "
                            "the SUBTOTAL, and every actual discount deduction."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "{image_url}",
                            "detail": "original",
                        },
                    },
                ],
            ),
        ]
    )

    chain = prompt | llm | parser

    return chain
    # return None


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    
    #[Path(a),Path(b),,,]
    inputs = [
        {"image_url": image_data_url(image)} for image in images
    ]

    results = chain.batch(inputs)

    print("Parsed results:")
    
    print(json.dumps(results, indent=2))

    paid_total = Decimal("0.00")
    paid_without_discount_total = Decimal("0.00")

    for result in results:
        paid = Decimal(str(result["paid"]))
        subtotal = Decimal(str(result["subtotal"]))

        discounts = [
            Decimal(str(value)) for value in result["discounts"]
        ]

        discount_total = sum(discounts, Decimal("0.00"))

        paid_total += paid
        paid_without_discount_total += subtotal + discount_total


    return {
        QUERY_1: f"HK${paid_total:.2f}",
        QUERY_2: f"HK${paid_without_discount_total:.2f}",
    }
    # _ = (chain, images)
    # return {QUERY_1: DUMMY_RESPONSE, QUERY_2: DUMMY_RESPONSE}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)

#把 LangChain 返回的Message，字符串等统一转成普通文字
def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()

#最终输出中检查金额。答案中只能出现一个数字
def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}

#比较 答案 和 ground truth
def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
