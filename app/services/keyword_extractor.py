import logging
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
from app.utils.enums import DocumentType
from app.core.config import settings
from app.dto.keyword_models import ExtractedKeywords

logger = logging.getLogger(__name__)

class KeywordExtractor:
    """Service for extracting keywords from documents using OpenAI"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        
    # Keyword templates for each document type
    KEYWORD_TEMPLATES = {
        DocumentType.INVOICE: {
            "vessel": "Name of the vessel carrying the cargo",
            "invoice_number": "Invoice number",
            "invoice_date": "Invoice date",
            "total_amount": "Total amount payable",
            "currency": "Payment currency",
            "payment_terms": "Payment terms",
            "due_date": "Due date",
            "seller": "Seller/supplier",
            "buyer": "Buyer",
            "port_of_loading": "Port of loading",
            "port_of_discharge": "Port of discharge",
            "commodity": "Goods/cargo",
            "quantity": "Quantity of goods",
            "unit_price": "Unit price",
            "net_weight": "Net weight",
            "gross_weight": "Gross weight",
            "packages": "Number of packages"
        },
        DocumentType.CONTRACT: {
            "contract_number": "Contract number",
            "contract_date": "Contract date",
            "parties": "Contract parties",
            "subject": "Subject of the contract",
            "price": "Price/cost",
            "currency": "Currency",
            "delivery_terms": "Delivery terms",
            "payment_terms": "Payment terms",
            "validity_period": "Validity period",
            "termination_conditions": "Termination conditions",
            "force_majeure": "Force majeure",
            "governing_law": "Governing law",
            "jurisdiction": "Jurisdiction"
        },
        DocumentType.COO: {
            "exporter": "Exporter",
            "importer": "Importer",
            "consignee": "Consignee",
            "country_of_origin": "Country of origin",
            "commodity": "Goods",
            "hs_code": "HS code",
            "quantity": "Quantity",
            "net_weight": "Net weight",
            "gross_weight": "Gross weight",
            "packages": "Number of packages",
            "vessel": "Vessel name",
            "port_of_loading": "Port of loading",
            "port_of_discharge": "Port of discharge",
            "loading_date": "Loading date",
            "certificate_number": "Certificate number",
            "issuing_authority": "Issuing authority"
        },
        DocumentType.COA: {
            "product_name": "Product name",
            "batch_number": "Batch number",
            "manufacturing_date": "Manufacturing date",
            "expiry_date": "Expiry date",
            "test_results": "Test results",
            "specifications": "Specifications",
            "quality_parameters": "Quality parameters",
            "laboratory": "Laboratory",
            "analyst": "Analyst",
            "test_date": "Test date",
            "certificate_number": "Certificate number",
            "standards": "Quality standards"
        },
        DocumentType.COW: {
            "vessel": "Vessel name",
            "port_of_loading": "Port of loading",
            "port_of_discharge": "Port of discharge",
            "loading_date": "Loading date",
            "commodity": "Goods",
            "net_weight": "Net weight",
            "gross_weight": "Gross weight",
            "tare_weight": "Tare weight",
            "packages": "Number of packages",
            "weighing_method": "Weighing method",
            "weighing_location": "Weighing location",
            "certificate_number": "Certificate number",
            "weighing_date": "Weighing date"
        },
        DocumentType.COQ: {
            "product_name": "Product name",
            "batch_number": "Batch number",
            "quality_grade": "Quality grade",
            "quality_parameters": "Quality parameters",
            "inspection_date": "Inspection date",
            "inspector": "Inspector",
            "inspection_company": "Inspection company",
            "quality_standards": "Quality standards",
            "certificate_number": "Certificate number",
            "validity_period": "Certificate validity period"
        },
        DocumentType.BL: {
            "vessel": "Vessel name",
            "voyage_number": "Voyage number",
            "bl_number": "Bill of lading number",
            "shipper": "Shipper",
            "consignee": "Consignee",
            "notify_party": "Notify party",
            "port_of_loading": "Port of loading",
            "port_of_discharge": "Port of discharge",
            "loading_date": "Loading date",
            "commodity": "Cargo description",
            "packages": "Number of packages",
            "net_weight": "Net weight",
            "gross_weight": "Gross weight",
            "freight_terms": "Freight terms",
            "freight_amount": "Freight amount",
            "currency": "Currency"
        },
        DocumentType.FINANCIAL: {
            "company_name": "Company name",
            "reporting_period": "Reporting period",
            "revenue": "Revenue",
            "profit": "Profit",
            "assets": "Assets",
            "liabilities": "Liabilities",
            "equity": "Equity",
            "cash_flow": "Cash flow",
            "currency": "Reporting currency",
            "auditor": "Auditor",
            "audit_date": "Audit date",
            "fiscal_year": "Fiscal year"
        },
        DocumentType.LC: {
            "lc_number": "LC number",
            "lc_date": "LC date",
            "lc_amount": "LC amount",
            "lc_currency": "LC currency",
            "lc_bank": "LC bank",
        },
        DocumentType.OTHER: {
            "document_type": "Document type",
            "key_entities": "Key entities",
            "important_dates": "Important dates",
            "amounts": "Amounts and numbers",
            "locations": "Locations",
            "organizations": "Organizations",
            "people": "People"
        }
    }
    
    def _get_keyword_template(self, document_type: DocumentType) -> Dict[str, str]:
        """Get the keyword template for the document type"""
        return self.KEYWORD_TEMPLATES.get(document_type, self.KEYWORD_TEMPLATES[DocumentType.OTHER])
    
    def _split_content_into_chunks(self, content: str, max_chunk_size: int = 8000) -> List[str]:
        """Split content into chunks for OpenAI processing"""
        if len(content) <= max_chunk_size:
            return [content]
        
        # Split by paragraphs, then by sentences if needed
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def extract_keywords_from_chunk(
        self, 
        content_chunk: str, 
        document_type: DocumentType,
        chunk_index: int = 0,
        total_chunks: int = 1
    ) -> Dict[str, Any]:
        """Extract keywords from a content chunk using Structured Outputs"""
        
        keyword_template = self._get_keyword_template(document_type)
        
        # Create prompt for keyword extraction
        template_list = "\n".join([f"- {key}: {desc}" for key, desc in keyword_template.items()])
        
        prompt = f"""Extract keywords from the following document text of type "{document_type.value}".

Document type: {document_type.value}
Part {chunk_index + 1} of {total_chunks}

INSTRUCTIONS:
1. Find and extract values for the following keywords (if present in the text):
{template_list}

2. You can add ONLY 2-3 additional keywords that are truly important and relevant to the document type. Do NOT extract every single detail.

3. For each keyword, provide only the extracted value (no context needed).

4. If a keyword is not found, do not include it in the result.

5. Extract only facts from the text, do not make anything up.

6. Use clear, descriptive names for keywords (e.g., vessel, invoice_number, total_amount, etc.).

7. Focus on the most critical information that would be useful for searching and categorizing this document.

8. If you extract any dates or datetimes, always return them in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).

9. Return the result as a JSON object with key-value pairs in the 'keywords' field:
{{
  "keywords": {{
    "vessel": "MV OCEAN STAR",
    "bl_number": "BL-2024-001",
    "port_of_loading": "MAPUTO, MOZAMBIQUE",
    "invoice_date": "2024-05-01"
  }}
}}

DOCUMENT TEXT:
{content_chunk}"""

        # Use Structured Outputs with Pydantic model using the new responses API
        response = await self.openai_client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": "You are an expert in extracting keywords from documents. Extract only the most important and relevant facts from the text. Be selective and focus on critical information that would be useful for searching and categorizing documents. Do not extract every minor detail."},
                {"role": "user", "content": prompt}
            ],
            text_format=ExtractedKeywords,
        )
        
        # Get the parsed response
        result = response.output_parsed
        
        # Convert to dict format - используем to_dict() для получения всех полей
        keywords_dict = result.to_dict()
        logger.info(f"Successfully extracted keywords from chunk {chunk_index + 1}/{total_chunks}")
        return keywords_dict
    
    async def extract_keywords(self, content: str, document_type: DocumentType) -> Dict[str, Any]:
        """Extract keywords from the entire document"""
        
        # Split content into chunks
        chunks = self._split_content_into_chunks(content)
        logger.info(f"Processing {len(chunks)} chunks for keyword extraction")
        
        all_keywords = {}
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_keywords = await self.extract_keywords_from_chunk(
                chunk, document_type, i, len(chunks)
            )
            
            # Merge results
            for key, value in chunk_keywords.items():
                if key not in all_keywords:
                    all_keywords[key] = value
                else:
                    # If the key already exists and values are different, keep both
                    if value != all_keywords[key]:
                        if f"{key}_alternative" not in all_keywords:
                            all_keywords[f"{key}_alternative"] = all_keywords[key]
                        all_keywords[f"{key}_alternative_2"] = value
        
        logger.info(f"Extracted {len(all_keywords)} unique keywords")
        return all_keywords
    
    def get_keyword_value(self, keywords: Dict[str, Any], keyword: str) -> Optional[str]:
        """Get the value of a keyword from the extracted data"""
        if keyword not in keywords:
            return None
        
        return str(keywords[keyword])
    
    def get_keyword_context(self, keywords: Dict[str, Any], keyword: str) -> Optional[str]:
        """Get the context of a keyword from the extracted data (not available in simplified format)"""
        return None
