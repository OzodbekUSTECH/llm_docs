import logging
from typing import List
from openai import AsyncOpenAI
from app.dto.ai_models import ContractSectionsOutput, InvoiceFieldsOutput


logger = logging.getLogger(__name__)


class ContractSectionExtractor:
    """Extracts structured CONTRACT sections using GPT structured outputs."""

    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client

    DEFAULT_ORDERED_TITLES: List[str] = [
        "SELLER",
        "BUYER",
        "PRODUCT",
        "DURATION",
        "QUANTITY",
        "QUALITY",
        "DELIVERY",
        "NOMINATION",
        "PRICE",
        "PAYMENT",
        "REACH",
        "DETERMINATION OF QUANTITY/QUALITY",
        "RISK AND PROPERTY",
        "LAYTIME AND DEMURRAGE",
        "FORCE MAJEURE",
        "ASSIGNMENT",
        "LAW AND JURISDICTION",
        "LIABILITY",
        "WAIVER",
        "ENTIRE AGREEMENT",
        "TAXES, DUTIES AND CHARGES",
        "V.A.T. AND EXCISE DUTY OR MINERAL OIL TAX",
        "LIQUIDATION",
        "ISPS CODE COMPLIANCE CLAUSES",
        "OTHER TERMS",
        "CONTACTS",
        "TABLES",
        "ADDITIONAL INFORMATION",
    ]

    # Reasonable default invoice field order for readability (does not enforce presence)
    DEFAULT_INVOICE_FIELDS: List[str] = [
        "INVOICE NO",
        "DATE",
        "SELLER",
        "BUYER",
        "CONTRACT NO",
        "VESSEL NAME",
        "LOADING PORT",
        "DISCHARGE PORT",
        "B/L NO",
        "COMMODITY",
        "QUANTITY",
        "UNIT PRICE",
        "TOTAL AMOUNT",
        "PAYMENT TERMS",
        "BANK DETAILS",
        "CURRENCY",
    ]

    async def extract(self, content: str) -> List[dict]:
        """
        Return a list of {title, content} sections, splitting each section into smaller logically coherent chunks
        if the content is too large, while avoiding overly small or excessively large segments.
        """

        ordered_titles = "\n".join([f"- {t}" for t in self.DEFAULT_ORDERED_TITLES])

        # NEW SYSTEM INSTRUCTION IN ENGLISH, fulfilling user's request:
        system = (
            "You are an expert contract parser. "
            "Split the contract text into logically distinct sections and chunks as follows:\n"
            "1. Use the provided ordered list of expected section headings. Match headings robustly (case-insensitive, support numbering and punctuation). Keep the original order.\n"
            "2. For EACH expected heading, extract ALL relevant content from ANYWHERE in the contract, including repeated and scattered occurrences. Merge them in original text order.\n"
            "3. For each section, if the content is long, SPLIT it into smaller, logically coherent chunks. "
            "Chunks must NOT be so small that they are meaningless, and NOT so large as to be unwieldy (aim for ~100–300 words or 500–1500 tokens per chunk, but keep sentences and meaning intact; never break in the middle of a sentence or logical item). "
            "Split at logical boundaries: paragraph, sub-clause, numbered list, sentence. Avoid cutting a logical point, even if near a soft token/length limit.\n"
            "4. The output must be a JSON array of objects, each having:\n"
            "  - 'title': exactly as from the headings list, indicating the section to which the chunk belongs\n"
            "  - 'content': a chunk of original contract text or markdown, not paraphrased or summarized\n"
            "If a section yields multiple chunks, repeat the 'title' for each chunked segment, preserving their order within the contract.\n"
            "5. For 'ISPS CODE COMPLIANCE CLAUSES' and 'CONTACTS', split as listed in the headings; do not merge them.\n"
            "6. Include tables as markdown under the most relevant chunk.\n"
            "7. Do NOT paraphrase, summarize, or invent content. "
            "Minimal normalization is allowed for whitespace, line breaks, and consistent list/table format.\n"
            "8. Omit sections with no meaningful content. If important content does not fit any predefined section, place it under 'ADDITIONAL INFORMATION'.\n"
            "IMPORTANT: Each chunk must be long enough to be meaningful, but not overly long—never break in the middle of a sentence or logical point."
        )

        user = (
            f"Expected headings in order:\n{ordered_titles}\n\n"
            "For each heading, extract ALL related content across the contract. If a heading appears or is relevant in multiple places, MERGE all parts accordingly, preserving their order.\n"
            "If a section's total content is too lengthy or diverse, SPLIT it by logical paragraphs, list items, or at complete sentence boundaries. Keep each chunk useful and readable—do not make them too short or excessively large.\n"
            "Return JSON as an array named 'sections': each object must have 'title' (from the list above) and 'content' (a non-empty, meaningful chunk of contract verbatim text or markdown). "
            "If splitting yields multiple content parts for one title, make separate array elements sharing the same 'title', ordered as found. "
            "Content must be direct from the contract (never invented or summarized), only minimally normalized for clean structure. "
            "Do not include empty or trivial chunks. "
            "Return AT MOST one JSON object per chunk, always associating it with its section 'title'.\n\n"
            f"CONTRACT TEXT:\n{content}"
        )

        response = await self.openai_client.responses.parse(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=ContractSectionsOutput,
        )

        output: ContractSectionsOutput = response.output_parsed
        # Note: Now, output.sections may include multiple chunks per title
        items: List[dict] = []
        for section in output.sections:
            title = section.title.strip()
            content_text = section.content.strip()
            if not title or not content_text:
                continue
            items.append({"title": title, "content": content_text})

        # Strictly order items by DEFAULT_ORDERED_TITLES, preserving chunked order for repeated titles
        order_index = {t: i for i, t in enumerate(self.DEFAULT_ORDERED_TITLES)}
        items = [it for it in items if it["title"] in order_index]
        # Since chunks may repeat the same title, secondary sort is not needed
        items.sort(key=lambda x: order_index[x["title"]])

        logger.info(f"Extracted {len(items)} contract section chunks")
        return items

    async def extract_invoice_fields(self, content: str) -> List[dict]:
        """Return list of {title, value} pairs suitable for chunking and search."""

        ordered_fields = "\n".join([f"- {t}" for t in self.DEFAULT_INVOICE_FIELDS])

        system = (
            "You are an expert invoice parser. Identify key invoice fields and extract them as title/value pairs. "
            "Match headings robustly (case-insensitive, tolerate punctuation and colon). "
            "Preserve wording of values (no paraphrasing); you may minimally normalize spacing and list/table formatting for readability. "
            "Combine signals from tables and free text. If a field appears multiple times, merge logically. "
            "Avoid overly short, low-signal chunks: if a value would be very short (e.g., just 'USD' or a single code), "
            "merge it with the most related fields into one meaningful value block under a composite title (e.g., 'PRICING', 'LOGISTICS', 'BANKING'). "
            "Prefer chunks that are informative on their own (roughly ≥ 30–40 characters) without being verbose. "
            "Do not invent values; include only what appears in the invoice."
        )

        user = (
            f"Common field order for readability (not all must appear):\n{ordered_fields}\n\n"
            "Extract fields like INVOICE NO, DATE, SELLER, BUYER, CONTRACT NO, VESSEL NAME, LOADING PORT, DISCHARGE PORT, B/L NO, COMMODITY, QUANTITY, UNIT PRICE, TOTAL AMOUNT, PAYMENT TERMS, BANK DETAILS, CURRENCY, and any other clearly labeled key fields. "
            "Return JSON with an array 'fields', each item has 'title' and 'value'. "
            "Title should be human-readable uppercase heading. Value should be verbatim/minimally normalized text; for tabular values, include a compact markdown table if helpful. "
            "If a standalone field would be too short to be useful, MERGE related fields into one chunk with a composite title (e.g., 'PRICING' may include UNIT PRICE, QUANTITY, TOTAL, CURRENCY; 'LOGISTICS' may include VESSEL NAME, LOADING/DISCHARGE PORTS, B/L NO). "
            "Return at most one item per unique title. Omit empty values.\n\n"
            f"INVOICE TEXT:\n{content}"
        )

        response = await self.openai_client.responses.parse(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=InvoiceFieldsOutput,
        )

        output: InvoiceFieldsOutput = response.output_parsed
        items: List[dict] = []
        for field in output.fields:
            title = field.title.strip().upper()
            value_text = field.value.strip()
            if not title or not value_text:
                continue
            items.append({"title": title, "value": value_text})

        # Reorder to default order first, then append unknowns at the end keeping model order
        order_index = {t: i for i, t in enumerate(self.DEFAULT_INVOICE_FIELDS)}
        known = [it for it in items if it["title"] in order_index]
        known.sort(key=lambda x: order_index[x["title"]])
        unknown = [it for it in items if it["title"] not in order_index]
        items = known + unknown

        logger.info(f"Extracted {len(items)} invoice fields")
        return items


