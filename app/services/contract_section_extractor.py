import logging
from typing import List
from openai import AsyncOpenAI
from app.dto.ai_models import ContractSectionsOutput


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
        "ISPS CODE COMPLIANCE CLAUSES — A. SELLERS' OBLIGATIONS",
        "ISPS CODE COMPLIANCE CLAUSES — B. BUYER'S OBLIGATIONS",
        "OTHER TERMS",
        "CONTACTS — FOR THE BUYER",
        "CONTACTS — FOR THE SELLER",
        "TABLES",
        "ADDITIONAL INFORMATION",
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
            "If any important information does not clearly belong to the predefined sections, place it under 'ADDITIONAL INFORMATION'."
        )

        user = (
            f"Expected headings in order:\n{ordered_titles}\n\n"
            "Return JSON with an array 'sections', each item has 'title' and 'content'. "
            "Title must be exactly from the expected list above. Content is plain text/markdown gathered from the contract, not invented.\n\n"
            f"CONTRACT TEXT:\n{content}"
        )

        response = await self.openai_client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=ContractSectionsOutput,
        )

        output: ContractSectionsOutput = response.output_parsed
        # Convert to list of dicts
        items = []
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


