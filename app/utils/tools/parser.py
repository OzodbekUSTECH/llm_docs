"""
Parser-only tools (runtime implementations) not shared with chat.
"""
from typing import Dict, List


def get_document_clauses(document_type: str) -> Dict[str, List[str]]:
    """Return canonical clause titles for the given document type.

    These titles guide segmentation. LLM should map semantically similar headings
    to the closest canonical clause. Keep list concise and practical.
    """
    dt = (document_type or "OTHER").upper()

    contract_clauses = [
        "Parties and Definitions",
        "Subject Matter / Scope",
        "Term and Termination",
        "Price and Payment Terms",
        "Delivery / Performance",
        "Quality and Inspection",
        "Warranties and Liability",
        "Confidentiality",
        "Force Majeure",
        "Governing Law and Dispute Resolution",
        "Notices",
        "Miscellaneous",
        "Signatures",
    ]

    invoice_clauses = [
        "Header and Parties",
        "Invoice Details",
        "Ship To / Bill To",
        "Line Items",
        "Totals and Taxes",
        "Banking Details",
        "Terms and Notes",
    ]

    bl_clauses = [
        "Shipper and Consignee",
        "Vessel and Voyage",
        "Ports and Dates",
        "Cargo Description",
        "Marks and Numbers",
        "Weights and Measurements",
        "Freight and Charges",
        "Conditions and Clauses",
    ]

    lc_clauses = [
        "Parties",
        "LC Details",
        "Amounts and Currencies",
        "Shipment Terms",
        "Documents Required",
        "Conditions",
        "Charges and Reimbursements",
    ]

    certificate_common = [
        "Header and Identifiers",
        "Parties and Authorities",
        "Goods Description",
        "Measurements and Results",
        "Conclusions and Certifications",
        "Notes and References",
    ]

    financial_clauses = [
        "Executive Summary",
        "Balance Sheet",
        "Income Statement",
        "Cash Flow Statement",
        "Notes to Financial Statements",
        "Management Discussion",
    ]

    mapping = {
        "CONTRACT": contract_clauses,
        "INVOICE": invoice_clauses,
        "BL": bl_clauses,
        "LC": lc_clauses,
        "COO": certificate_common,
        "COA": certificate_common,
        "COW": certificate_common,
        "COQ": certificate_common,
        "FINANCIAL": financial_clauses,
        "OTHER": [
            "Title and Metadata",
            "Overview",
            "Main Content",
            "Details",
            "Appendices / Tables",
        ],
    }

    return {"document_type": dt, "clauses": mapping.get(dt, mapping["OTHER"])}


