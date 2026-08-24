# Recovered Langflow component
# type: MarkdownToPDF
# class: MarkdownToPDF
# used in 1 flow(s): 280 Pages
# json path: node.data.node.template.code.value

from langflow.custom import Component
from langflow.io import HandleInput, Output, MessageTextInput
from langflow.schema import Data
import re

class MarkdownToPDF(Component):
    display_name = "Markdown to PDF (fpdf2)"
    description = "Converts Markdown to PDF. Note: requires 'pip install fpdf2'"
    icon = "file-pdf"

    inputs = [
        HandleInput(
            name="markdown_text",
            display_name="Markdown Text",
            info="Connect the merged string here.",
            input_types=["Data", "Message", "str"],
        ),
        MessageTextInput(
            name="file_path",
            display_name="Output File Path",
            value="AIT_Consolidated_Report.pdf"
        ),
    ]

    outputs = [
        Output(display_name="File Path", name="file_path_out", method="convert"),
    ]

    def convert(self) -> str:
        try:
            from fpdf import FPDF
        except ImportError:
            return "Error: Please run 'pip install fpdf2' in your environment."

        # Extracting the string from the input regardless of object type
        raw_input = self.markdown_text
        if isinstance(raw_input, Data):
            text_to_process = raw_input.data.get("text", str(raw_input))
        elif hasattr(raw_input, "text"): # If it's a Message object
            text_to_process = raw_input.text
        else:
            text_to_process = str(raw_input)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)

        lines = text_to_process.split('\n')
        
        for line in lines:
            # Handle Page Headers
            if line.startswith('# PAGE:'):
                pdf.add_page()
                pdf.set_font("Helvetica", 'B', 16)
                pdf.set_text_color(0, 51, 102)
                pdf.multi_cell(0, 10, line.replace('# PAGE:', '').strip())
                pdf.ln(2)
            
            # Handle Section Headers
            elif line.startswith('##'):
                pdf.set_font("Helvetica", 'B', 13)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(4)
                pdf.multi_cell(0, 8, line.replace('##', '').strip())
            
            # Handle Sub Headers
            elif line.startswith('###'):
                pdf.set_font("Helvetica", 'B', 11)
                pdf.ln(2)
                pdf.multi_cell(0, 7, line.replace('###', '').strip())

            # Handle Links
            elif '[' in line and '](' in line:
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(0, 0, 255)
                clean_line = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', line)
                pdf.multi_cell(0, 6, clean_line)
                pdf.set_text_color(0, 0, 0)

            # Normal Text
            else:
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(0, 0, 0)
                if line.strip():
                    pdf.multi_cell(0, 6, line.strip())

        pdf.output(self.file_path)
        self.status = f"Saved to {self.file_path}"
        return self.file_path