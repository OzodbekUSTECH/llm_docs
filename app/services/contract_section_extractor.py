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
        """Return list of {title, content} in the specified order; skip empty."""

        ordered_titles = "\n".join([f"- {t}" for t in self.DEFAULT_ORDERED_TITLES])

        system = (
            "You are an expert contract parser. Split the contract text into logically distinct "
            "sections using the provided ordered list of expected headings. Match headings "
            "robustly (case-insensitive, tolerate numbering and punctuation). Keep the original order. "
            "Include a section only if meaningful content exists. For 'ISPS' and 'CONTACTS' split into the two sub-sections as listed. "
            "If there are tables relevant to any section, include their markdown under that section content. "
            "Critically: for EACH expected heading, COLLECT ALL content that belongs to it from the ENTIRE contract. "
            "This includes multiple occurrences throughout the document (e.g., repeated PRICE clauses); MERGE them in original order. "
            "Do NOT paraphrase or reword the source text; preserve the original wording. "
            "You MAY minimally normalize spacing for readability (collapse excessive blank lines, fix list bullets/indentation), and keep tables as markdown if present. "
            "Maintain original clauses, numbering, and structure; no summaries. "
            "If any important information does not clearly belong to the predefined sections, place it under 'ADDITIONAL INFORMATION'."
        )

        user = (
            f"Expected headings in order:\n{ordered_titles}\n\n"
            "For each heading, extract ALL related content from the whole contract. If a heading appears multiple times or content is scattered, MERGE all parts keeping the original text order yourself. "
            "Include everything until the next heading begins each time (paragraphs, lists, sub-clauses, tables as markdown). "
            "Return JSON with an array 'sections', each item has 'title' and 'content'. "
            "Title must be exactly from the expected list above. Content must be from the contract (plain text/markdown), not invented, not paraphrased, not summarized. "
            "You may only normalize spacing and list/table formatting minimally for readability; preserve wording. "
            "Return AT MOST ONE item per title. If no content for a title, omit that title.\n\n"
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
        # Convert to list of dicts; GPT must already aggregate per title
        items: List[dict] = []
        for section in output.sections:
            title = section.title.strip()
            content_text = section.content.strip()
            if not title or not content_text:
                continue
            items.append({"title": title, "content": content_text})

        # Reorder strictly by DEFAULT_ORDERED_TITLES, dropping unknowns
        order_index = {t: i for i, t in enumerate(self.DEFAULT_ORDERED_TITLES)}
        items = [it for it in items if it["title"] in order_index]
        items.sort(key=lambda x: order_index[x["title"]])

        logger.info(f"Extracted {len(items)} contract sections")
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


