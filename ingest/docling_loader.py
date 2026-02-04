# ingest/docling_loader.py
import os
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (PdfPipelineOptions, RapidOcrOptions, smolvlm_picture_description)
from docling.document_converter import DocumentConverter, PdfFormatOption


class CustomDocumentLoader:
    """
    Docling Loader configured to use RapidOCR with custom ONNX models.
    Converts a PDF into a Docling Document with OCR.
    """

    def __init__(self):
        self.image_placeholder = "[IMAGE]"
        self.page_break_placeholder = "[PAGE_BREAK]"
        self.converter = self._configure_ocr_converter()



    def process_document(self, pdf_path: str = None, n_pages : int = -1) -> str:
        """Process the PDF and return the text with image annotations."""

        document = self.convert(pdf_path)
        annotations = self._extract_image_annotations(document)

        text = document.export_to_markdown(
            image_placeholder=self.image_placeholder,
            page_break_placeholder=self.page_break_placeholder
        )

        # Replace each image_placeholder with the corresponding image annotation
        text = self._replace_occurrences(text, self.image_placeholder, annotations)

        if n_pages == -1:
            return text
        
        return self.page_break_placeholder.join(text.split(self.page_break_placeholder)[:n_pages])  


    def convert(self, pdf_path: str = None):
        """Convert the PDF and return a Docling Document object"""
        conversion_result = self.converter.convert(pdf_path)
        return conversion_result.document



    @staticmethod
    def _extract_image_annotations(document) -> list[str]:
        """
        Extract text annotations from all images in the document.
        If multiple annotations exist for an image, take the first one.
        If no annotation exists, store None.
        """
        annotations = []
        for picture in document.pictures:
            if picture.annotations:
                annotations.append(picture.annotations[0].text)
            else:
                annotations.append(None)  # No description for this image
        return annotations

    @staticmethod
    def _replace_occurrences(text: str, to_replace: str, replacements: list[str]) -> str:
        """
        Replace occurrences of image placeholder with image description.
        If no description exists, remove the placeholder.
        """
        parts = text.split(to_replace)
        if len(parts) == 1:
            # No placeholders found
            return text

        new_text = parts[0]
        for i, replacement in enumerate(replacements):
            if i + 1 < len(parts):
                desc = replacement if replacement else ""
                new_text += desc + parts[i + 1]

        if len(replacements) < len(parts) - 1:
            new_text += "".join(parts[len(replacements)+1:])

        return new_text
    

    @staticmethod
    def _configure_ocr_converter() -> DocumentConverter:
        """
        Configure Docling pipeline to use RapidOCR with ONNX models.
        Downloads RapidOCR models if needed.
        """

        # PDF pipeline options
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            generate_page_images=True,
            images_scale=1.0, 
            do_picture_description=True,
            picture_description_options=smolvlm_picture_description
        )
        pipeline_options.ocr_options = RapidOcrOptions(backend="torch")

        # Document converter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        return converter
    

    @staticmethod
    def chunk_text(markdown_text: str, page_break_placeholder: str = "[PAGE_BREAK]") -> list[dict]:
        """
        Split markdown text into chunks using page breaks.
        """
        raw_chunks = markdown_text.split(page_break_placeholder)
        chunks = []

        for idx, content in enumerate(raw_chunks):
            content = content.strip()
            if not content:
                continue  # skip empty pages
            chunks.append({
                "chunk_idx": idx,
                "page_number": idx + 1,
                "content": content
            })

        return chunks