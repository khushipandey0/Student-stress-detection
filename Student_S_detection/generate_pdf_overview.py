"""
Generate a concise, professional PDF overview for the Student Stress
Detection System. Includes: overview, abstract, problem statement,
chosen model and rationale, key results with visuals, and conclusion.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import os
from datetime import datetime


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            leading=28,
            textColor=HexColor("#2c3e50"),
            alignment=1,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CustomSection",
            parent=styles["Heading2"],
            fontSize=16,
            leading=19,
            textColor=HexColor("#34495e"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CustomBody",
            parent=styles["Normal"],
            fontSize=11.5,
            leading=15,
            textColor=HexColor("#2c3e50"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CustomBullet",
            parent=styles["Normal"],
            fontSize=11.5,
            leading=15,
            textColor=HexColor("#2c3e50"),
            leftIndent=12,
            bulletIndent=0,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CustomItalic",
            parent=styles["Italic"],
            fontSize=10.5,
            leading=14,
            textColor=HexColor("#7f8c8d"),
        )
    )
    return styles


def add_section_heading(content, styles, text):
    content.append(Paragraph(text, styles["CustomSection"]))


def add_bullets(content, styles, items):
    for item in items:
        content.append(Paragraph(f"• {item}", styles["CustomBullet"]))


def add_image_if_exists(content, styles, path, width=5.8 * inch):
    if os.path.exists(path):
        img = Image(path)
        img.drawHeight = img.drawHeight * (width / img.drawWidth)
        img.drawWidth = width
        content.append(img)
        content.append(Spacer(1, 0.2 * inch))
    else:
        content.append(
            Paragraph(f"(visual not found at {path})", styles["CustomItalic"])
        )


def build_pdf(output_path="/home/sahu/Desktop/Student_S_detection/PROJECT_OVERVIEW.pdf"):
    styles = build_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    content = []

    # Title
    title = "Student Stress Detection System — Executive Overview"
    subtitle = f"Concise report | {datetime.now().strftime('%B %d, %Y')}"
    content.append(Paragraph(title, styles["CustomTitle"]))
    content.append(Paragraph(subtitle, styles["CustomBody"]))
    content.append(Spacer(1, 0.2 * inch))

    # Overview / Abstract
    add_section_heading(content, styles, "Abstract (Overview)")
    content.append(
        Paragraph(
            "A lightweight machine learning system predicts student stress levels "
            "(low / medium / high) using an ensemble of Logistic Regression, SVM, "
            "Naive Bayes, and Random Forest. The pipeline spans synthetic data "
            "generation (10,000 records), cleaning, SMOTE balancing, scaling, model "
            "training, evaluation, and a Flask web UI for real-time predictions and "
            "motivational suggestions.",
            styles["CustomBody"],
        )
    )

    # Problem Statement
    add_section_heading(content, styles, "Problem Statement")
    add_bullets(
        content,
        styles,
        [
            "Stress impacts student well-being and academic outcomes; early detection is essential.",
            "Manual assessments are slow, subjective, and not scalable.",
            "Need an automated, data-driven, and actionable prediction tool.",
        ],
    )

    # Chosen Model & Rationale
    add_section_heading(content, styles, "Chosen Model & Rationale")
    content.append(
        Paragraph(
            "Selected Model: Voting Ensemble (soft voting) combining Logistic Regression, "
            "SVM, Naive Bayes, and Random Forest.",
            styles["CustomBody"],
        )
    )
    add_bullets(
        content,
        styles,
        [
            "Combines diverse learners to balance bias/variance and improve robustness.",
            "Soft voting leverages calibrated class probabilities for better uncertainty handling.",
            "Generalizes better than individual models and reduces overfitting risk.",
        ],
    )

    # Results with Visuals
    add_section_heading(content, styles, "Results (Overview)")
    add_bullets(
        content,
        styles,
        [
            "Accuracy (test): Ensemble 91.36%; strongest individual model (Random Forest) 93.20%.",
            "Low vs. Medium well separated; some Medium/High confusion (area to tune).",
            "Outputs class probabilities to gauge prediction confidence.",
        ],
    )
    # Add visuals if available
    img_base = "/home/sahu/Desktop/Student_S_detection/evaluation"
    add_image_if_exists(content, styles, os.path.join(img_base, "accuracy_comparison.png"))
    add_image_if_exists(content, styles, os.path.join(img_base, "confusion_matrices.png"))

    # Conclusion
    add_section_heading(content, styles, "Conclusion")
    add_bullets(
        content,
        styles,
        [
            "End-to-end, reproducible pipeline ready for pilot use.",
            "Ensemble model offers strong, balanced performance with probability outputs.",
            "Next focus: threshold tuning for High-stress recall, real-world validation, monitoring.",
        ],
    )

    doc.build(content)
    print(f"Overview PDF generated: {output_path}")


if __name__ == "__main__":
    build_pdf()

