"""PDF to markdown conversion using docling."""

import logging
from pathlib import Path
from typing import List, Dict
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.converter = DocumentConverter()
        logger.info("DocumentProcessor initialized")
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, str]:
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            result = self.converter.convert(str(pdf_path))
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Successfully converted {pdf_path.name}")
            
            return {
                "filename": pdf_path.stem,
                "source_path": str(pdf_path),
                "content": markdown_content,
            }
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            raise
    
    def process_directory(self, directory: Path) -> List[Dict[str, str]]:
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        documents = []
        for pdf_path in pdf_files:
            try:
                documents.append(self.process_pdf(pdf_path))
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {str(e)}")
        
        logger.info(f"Processed {len(documents)}/{len(pdf_files)} documents")
        return documents
    
    def save_markdown(self, document: Dict[str, str], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{document['filename']}.md"
        output_path.write_text(document['content'], encoding='utf-8')
        logger.info(f"Saved {output_path}")
        return output_path


