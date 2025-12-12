"""
Script to convert PROJECT_REPORT.md to PDF format
Uses reportlab for PDF generation
"""

import re
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import os
from datetime import datetime

class PDFReportGenerator:
    def __init__(self, markdown_file, output_pdf):
        self.markdown_file = markdown_file
        self.output_pdf = output_pdf
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        self.content = []
        
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Heading 1
        self.styles.add(ParagraphStyle(
            name='CustomH1',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Heading 2
        self.styles.add(ParagraphStyle(
            name='CustomH2',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#34495e'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        # Heading 3
        self.styles.add(ParagraphStyle(
            name='CustomH3',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=HexColor('#34495e'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Normal text
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=HexColor('#2c3e50'),
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            leading=14
        ))
        
        # Abstract style
        self.styles.add(ParagraphStyle(
            name='Abstract',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=HexColor('#2c3e50'),
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leftIndent=20,
            rightIndent=20,
            leading=14,
            fontName='Helvetica-Oblique'
        ))
        
    def parse_markdown(self):
        """Parse markdown file and convert to PDF elements"""
        with open(self.markdown_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Title (first # heading)
            if line.startswith('# ') and i < 5:
                title = line[2:].strip()
                self.content.append(Spacer(1, 0.5*inch))
                self.content.append(Paragraph(title, self.styles['CustomTitle']))
                self.content.append(Spacer(1, 0.3*inch))
                i += 1
                continue
            
            # Heading 1
            if line.startswith('# '):
                heading = line[2:].strip()
                self.content.append(Spacer(1, 0.2*inch))
                self.content.append(Paragraph(heading, self.styles['CustomH1']))
                i += 1
                continue
            
            # Heading 2
            if line.startswith('## '):
                heading = line[3:].strip()
                self.content.append(Paragraph(heading, self.styles['CustomH2']))
                i += 1
                continue
            
            # Heading 3
            if line.startswith('### '):
                heading = line[4:].strip()
                self.content.append(Paragraph(heading, self.styles['CustomH3']))
                i += 1
                continue
            
            # Horizontal rule
            if line.startswith('---'):
                self.content.append(Spacer(1, 0.1*inch))
                i += 1
                continue
            
            # Abstract section
            if 'Abstract' in line and i < 10:
                i += 1
                abstract_text = ""
                while i < len(lines) and not lines[i].strip().startswith('##'):
                    if lines[i].strip():
                        abstract_text += lines[i].strip() + " "
                    i += 1
                self.content.append(Paragraph(abstract_text.strip(), self.styles['Abstract']))
                self.content.append(Spacer(1, 0.2*inch))
                continue
            
            # Table detection (improved)
            if '|' in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if '---' in next_line or '|' in next_line:
                    table_data = []
                    # Read table header
                    if '|' in line:
                        row = [cell.strip() for cell in line.split('|')[1:-1]]
                        table_data.append(row)
                    i += 1
                    # Skip separator line
                    if '---' in lines[i]:
                        i += 1
                    # Read table rows
                    while i < len(lines) and '|' in lines[i] and '---' not in lines[i]:
                        row = [cell.strip() for cell in lines[i].split('|')[1:-1]]
                        if row:  # Only add non-empty rows
                            table_data.append(row)
                        i += 1
                    
                    if table_data and len(table_data) > 1:
                        # Create table with proper formatting
                        formatted_table_data = []
                        for row_idx, row in enumerate(table_data):
                            formatted_row = []
                            for cell in row:
                                # Clean markdown formatting
                                cell = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', cell)
                                cell = re.sub(r'`(.*?)`', r'<i>\1</i>', cell)
                                formatted_row.append(Paragraph(cell, self.styles['CustomBody'] if row_idx > 0 else ParagraphStyle(
                                    name='TableHeader',
                                    parent=self.styles['CustomBody'],
                                    fontName='Helvetica-Bold',
                                    fontSize=10
                                )))
                            formatted_table_data.append(formatted_row)
                        
                        # Create table
                        col_widths = [2*inch] * len(formatted_table_data[0])
                        table = Table(formatted_table_data, colWidths=col_widths)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3498db')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('TOPPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ecf0f1')),
                            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                            ('FONTSIZE', (0, 1), (-1, -1), 9),
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f8f9fa')]),
                        ]))
                        self.content.append(table)
                        self.content.append(Spacer(1, 0.2*inch))
                    continue
            
            # Code blocks (format as monospace)
            if line.startswith('```'):
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i].rstrip())
                    i += 1
                i += 1
                if code_lines:
                    code_text = '\n'.join(code_lines)
                    code_style = ParagraphStyle(
                        name='Code',
                        parent=self.styles['Normal'],
                        fontSize=9,
                        fontName='Courier',
                        textColor=HexColor('#2c3e50'),
                        leftIndent=20,
                        rightIndent=20,
                        backColor=HexColor('#f4f4f4'),
                        borderPadding=5,
                        leading=11
                    )
                    self.content.append(Paragraph(code_text.replace(' ', '&nbsp;'), code_style))
                    self.content.append(Spacer(1, 0.1*inch))
                continue
            
            # Bullet points
            if line.startswith('- ') or line.startswith('* '):
                text = line[2:].strip()
                # Remove markdown formatting
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                text = re.sub(r'`(.*?)`', r'<i>\1</i>', text)
                self.content.append(Paragraph(f"• {text}", self.styles['CustomBody']))
                i += 1
                continue
            
            # Numbered lists
            if re.match(r'^\d+\.\s', line):
                text = re.sub(r'^\d+\.\s', '', line)
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                text = re.sub(r'`(.*?)`', r'<i>\1</i>', text)
                self.content.append(Paragraph(text, self.styles['CustomBody']))
                i += 1
                continue
            
            # Regular paragraph
            paragraph_text = line
            # Collect continuation lines
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('#') and not lines[i].strip().startswith('-') and not lines[i].strip().startswith('|') and not lines[i].strip().startswith('```'):
                paragraph_text += " " + lines[i].strip()
                i += 1
            
            # Clean markdown formatting
            paragraph_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', paragraph_text)
            paragraph_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', paragraph_text)
            paragraph_text = re.sub(r'`(.*?)`', r'<i>\1</i>', paragraph_text)
            paragraph_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', paragraph_text)  # Remove links
            
            # Handle special sections
            if '**Keywords:**' in paragraph_text:
                self.content.append(Spacer(1, 0.1*inch))
            
            self.content.append(Paragraph(paragraph_text, self.styles['CustomBody']))
        
    def add_title_page(self):
        """Add a title page"""
        title_elements = []
        title_elements.append(Spacer(1, 2*inch))
        title_elements.append(Paragraph("Student Stress Detection System", self.styles['CustomTitle']))
        title_elements.append(Spacer(1, 0.3*inch))
        title_elements.append(Paragraph("Using Ensemble Machine Learning", ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=HexColor('#7f8c8d'),
            alignment=TA_CENTER,
            spaceAfter=30
        )))
        title_elements.append(Spacer(1, 1*inch))
        title_elements.append(Paragraph("A Comprehensive Machine Learning Project Report", ParagraphStyle(
            name='Subtitle2',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=HexColor('#95a5a6'),
            alignment=TA_CENTER
        )))
        title_elements.append(Spacer(1, 2*inch))
        title_elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", ParagraphStyle(
            name='Date',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=HexColor('#95a5a6'),
            alignment=TA_CENTER
        )))
        return title_elements
    
    def generate_pdf(self):
        """Generate the PDF document"""
        # Create PDF document
        doc = SimpleDocTemplate(
            self.output_pdf,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Add title page
        title_page = self.add_title_page()
        title_page.append(PageBreak())
        
        # Parse markdown and add content
        self.parse_markdown()
        
        # Combine all content
        all_content = title_page + self.content
        
        # Build PDF
        doc.build(all_content)
        print(f"PDF report generated successfully: {self.output_pdf}")

def main():
    # File paths
    markdown_file = '/home/sahu/Desktop/Student_S_detection/PROJECT_REPORT.md'
    output_pdf = '/home/sahu/Desktop/Student_S_detection/PROJECT_REPORT.pdf'
    
    # Check if markdown file exists
    if not os.path.exists(markdown_file):
        print(f"Error: Markdown file not found: {markdown_file}")
        return
    
    # Generate PDF
    print("Generating PDF report...")
    generator = PDFReportGenerator(markdown_file, output_pdf)
    generator.generate_pdf()
    print("Done!")

if __name__ == "__main__":
    main()

