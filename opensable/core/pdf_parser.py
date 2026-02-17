"""
PDF Parser for SableCore
Extract text, images, and tables from PDF documents
"""
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not installed. Run: pip install PyPDF2")

@dataclass
class PDFPage:
    """Single PDF page"""
    page_number: int
    text: str

@dataclass
class PDFDocument:
    """PDF document"""
    filename: str
    num_pages: int
    pages: List[PDFPage]
    
    def get_full_text(self) -> str:
        return "\n\n".join([p.text for p in self.pages])

class PDFParser:
    """Simple PDF parser"""
    
    def __init__(self, config=None):
        self.config = config
        if not PDF_AVAILABLE:
            logger.warning("PyPDF2 not installed - PDF parsing will be limited")
    
    async def parse(self, pdf_source: Union[str, Path, bytes]) -> PDFDocument:
        """Parse PDF"""
        import PyPDF2
        import io
        
        if isinstance(pdf_source, bytes):
            pdf = PyPDF2.PdfReader(io.BytesIO(pdf_source))
        else:
            pdf = PyPDF2.PdfReader(str(pdf_source))
        
        pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(PDFPage(page_number=i+1, text=text))
        
        return PDFDocument(
            filename=str(pdf_source),
            num_pages=len(pages),
            pages=pages
        )
