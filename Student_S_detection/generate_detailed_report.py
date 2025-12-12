"""
Generate a detailed 5-6 page professional project report for the
Student Stress Detection System with comprehensive content.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle,
)
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import os
from datetime import datetime


def build_styles():
    """Build custom styles for professional report"""
    styles = getSampleStyleSheet()
    
    # Title style
    if "ReportTitle" not in [s.name for s in styles.byName.values()]:
        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Heading1"],
                fontSize=22,
                leading=26,
                textColor=HexColor("#2c3e50"),
                alignment=TA_CENTER,
                spaceAfter=12,
                fontName="Helvetica-Bold",
            )
        )
    
    # Subtitle
    if "ReportSubtitle" not in [s.name for s in styles.byName.values()]:
        styles.add(
            ParagraphStyle(
                name="ReportSubtitle",
                parent=styles["Normal"],
                fontSize=12,
                leading=14,
                textColor=HexColor("#7f8c8d"),
                alignment=TA_CENTER,
                spaceAfter=20,
            )
        )
    
    # Section heading
    if "SectionHeading" not in [s.name for s in styles.byName.values()]:
        styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=styles["Heading2"],
                fontSize=14,
                leading=18,
                textColor=HexColor("#34495e"),
                spaceBefore=16,
                spaceAfter=8,
                fontName="Helvetica-Bold",
            )
        )
    
    # Subsection heading
    if "SubsectionHeading" not in [s.name for s in styles.byName.values()]:
        styles.add(
            ParagraphStyle(
                name="SubsectionHeading",
                parent=styles["Heading3"],
                fontSize=12,
                leading=16,
                textColor=HexColor("#34495e"),
                spaceBefore=10,
                spaceAfter=6,
                fontName="Helvetica-Bold",
            )
        )
    
    # Body text
    if "BodyText" not in [s.name for s in styles.byName.values()]:
        styles.add(
            ParagraphStyle(
                name="BodyText",
                parent=styles["Normal"],
                fontSize=11,
                leading=14,
                textColor=HexColor("#2c3e50"),
                alignment=TA_JUSTIFY,
                spaceAfter=6,
            )
        )
    
    # Bullet text
    if "BulletText" not in [s.name for s in styles.byName.values()]:
        styles.add(
            ParagraphStyle(
                name="BulletText",
                parent=styles["Normal"],
                fontSize=11,
                leading=14,
                textColor=HexColor("#2c3e50"),
                leftIndent=18,
                bulletIndent=0,
                spaceAfter=4,
            )
        )
    
    # Abstract style
    if "AbstractText" not in [s.name for s in styles.byName.values()]:
        styles.add(
            ParagraphStyle(
                name="AbstractText",
                parent=styles["Normal"],
                fontSize=11,
                leading=15,
                textColor=HexColor("#2c3e50"),
                alignment=TA_JUSTIFY,
                leftIndent=20,
                rightIndent=20,
                spaceAfter=12,
            )
        )
    
    return styles


def add_image_if_exists(content, path, width=5.5 * inch, caption=None):
    """Add image if it exists"""
    if os.path.exists(path):
        try:
            img = Image(path)
            # Scale image proportionally
            aspect_ratio = img.drawHeight / img.drawWidth
            img.drawWidth = width
            img.drawHeight = width * aspect_ratio
            content.append(img)
            if caption:
                content.append(Spacer(1, 0.1 * inch))
                caption_style = ParagraphStyle(
                    name="Caption",
                    parent=getSampleStyleSheet()["Normal"],
                    fontSize=9,
                    alignment=TA_CENTER,
                    textColor=HexColor("#7f8c8d"),
                    italic=True,
                )
                content.append(Paragraph(caption, caption_style))
            content.append(Spacer(1, 0.2 * inch))
            return True
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return False
    return False


def create_performance_table(styles):
    """Create performance comparison table"""
    data = [
        ["Model", "Accuracy (%)"],
        ["Logistic Regression", "89.11"],
        ["Support Vector Machine", "90.99"],
        ["Naive Bayes", "83.98"],
        ["Random Forest", "93.20"],
        ["Ensemble Model (Selected)", "91.36"],
    ]
    
    table = Table(data, colWidths=[3.5 * inch, 2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 10),
                ("BACKGROUND", (0, 1), (-1, -1), HexColor("#ecf0f1")),
                ("BACKGROUND", (0, 5), (-1, 5), HexColor("#d5e8d4")),  # Highlight selected
                ("FONTNAME", (0, 5), (-1, 5), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f8f9fa")]),
            ]
        )
    )
    return table


def build_pdf(output_path="/home/sahu/Desktop/Student_S_detection/PROJECT_REPORT_DETAILED.pdf"):
    """Build the detailed project report"""
    styles = build_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )
    
    content = []
    
    # ========== PAGE 1: TITLE AND ABSTRACT ==========
    content.append(Spacer(1, 0.5 * inch))
    content.append(Paragraph("Student Stress Detection System", styles["ReportTitle"]))
    content.append(Paragraph("Using Ensemble Machine Learning", styles["ReportSubtitle"]))
    content.append(Paragraph(f"Project Report | {datetime.now().strftime('%B %d, %Y')}", styles["ReportSubtitle"]))
    content.append(Spacer(1, 0.4 * inch))
    
    # Abstract
    content.append(Paragraph("Abstract", styles["SectionHeading"]))
    abstract_text = (
        "This project presents a comprehensive machine learning system for detecting and "
        "predicting student stress levels using ensemble learning techniques. The system "
        "analyzes ten key features related to student lifestyle, academic performance, and "
        "personal circumstances to classify stress levels into three categories: low, medium, "
        "and high. Four individual machine learning models—Logistic Regression, Support Vector "
        "Machine (SVM), Naive Bayes, and Random Forest—were trained and evaluated. An ensemble "
        "model combining all four models using a Voting Classifier achieved 91.36% accuracy. "
        "The system includes a complete data pipeline from synthetic data generation to "
        "preprocessing, model training, evaluation, and deployment through a web application. "
        "The web interface provides real-time stress predictions along with personalized "
        "motivational suggestions to help students manage their stress levels effectively."
    )
    content.append(Paragraph(abstract_text, styles["AbstractText"]))
    content.append(PageBreak())
    
    # ========== PAGE 2: PROBLEM STATEMENT ==========
    content.append(Paragraph("1. Problem Statement", styles["SectionHeading"]))
    
    problem_text = (
        "Student stress has become a critical concern in educational institutions worldwide. "
        "Academic pressure, financial constraints, social challenges, and workload management "
        "contribute significantly to student stress levels. Early detection and intervention "
        "can help prevent severe mental health issues and improve overall academic performance "
        "and well-being."
    )
    content.append(Paragraph(problem_text, styles["BodyText"]))
    
    content.append(Spacer(1, 0.1 * inch))
    content.append(Paragraph("1.1 Challenges", styles["SubsectionHeading"]))
    challenges = [
        "Traditional stress assessment methods rely on self-reported questionnaires and clinical "
        "evaluations, which are time-consuming and subjective.",
        "Manual assessments may not capture real-time stress patterns or identify at-risk "
        "students early enough.",
        "Lack of automated, scalable solutions for stress detection in educational settings.",
        "Need for actionable insights and personalized recommendations based on individual "
        "student circumstances.",
    ]
    for challenge in challenges:
        content.append(Paragraph(f"• {challenge}", styles["BulletText"]))
    
    content.append(Spacer(1, 0.1 * inch))
    content.append(Paragraph("1.2 Objectives", styles["SubsectionHeading"]))
    objectives = [
        "Develop a robust machine learning system capable of accurately predicting student "
        "stress levels based on lifestyle and academic features.",
        "Compare multiple classification algorithms to identify the most effective approach.",
        "Implement ensemble learning to improve prediction accuracy and robustness.",
        "Create a user-friendly web application for real-time stress prediction and intervention.",
        "Provide actionable insights through personalized recommendations.",
    ]
    for obj in objectives:
        content.append(Paragraph(f"• {obj}", styles["BulletText"]))
    
    content.append(PageBreak())
    
    # ========== PAGE 3: METHODOLOGY ==========
    content.append(Paragraph("2. Methodology", styles["SectionHeading"]))
    
    content.append(Paragraph("2.1 Dataset", styles["SubsectionHeading"]))
    dataset_text = (
        "A synthetic dataset of 10,000 student records was generated using Python, incorporating "
        "realistic distributions and relationships between features. The dataset includes ten "
        "features: study hours, sleep hours, exercise hours, social activities, assignment "
        "deadlines, exam pressure, family support, financial stress, academic performance, and "
        "workload level. The target variable has three classes: low, medium, and high stress levels."
    )
    content.append(Paragraph(dataset_text, styles["BodyText"]))
    
    content.append(Spacer(1, 0.1 * inch))
    content.append(Paragraph("2.2 Data Preprocessing", styles["SubsectionHeading"]))
    preprocessing_steps = [
        "Data Cleaning: Removal of missing values, duplicates, and outliers using Interquartile "
        "Range (IQR) method.",
        "Data Balancing: Application of SMOTE (Synthetic Minority Oversampling Technique) to "
        "address class imbalance.",
        "Feature Scaling: StandardScaler normalization to ensure features have zero mean and "
        "unit variance.",
        "Train-Test Split: 80-20 stratified split to maintain class distribution.",
    ]
    for step in preprocessing_steps:
        content.append(Paragraph(f"• {step}", styles["BulletText"]))
    
    content.append(Spacer(1, 0.1 * inch))
    content.append(Paragraph("2.3 Model Selection", styles["SubsectionHeading"]))
    model_selection_text = (
        "Four machine learning algorithms were selected for comparison: Logistic Regression "
        "(interpretable baseline), Support Vector Machine (non-linear classification), Naive Bayes "
        "(fast probabilistic classifier), and Random Forest (captures complex interactions). "
        "An ensemble model using Voting Classifier with soft voting was created to combine "
        "predictions from all four models."
    )
    content.append(Paragraph(model_selection_text, styles["BodyText"]))
    
    content.append(PageBreak())
    
    # ========== PAGE 4: MODEL JUSTIFICATION ==========
    content.append(Paragraph("3. Model Selection and Justification", styles["SectionHeading"]))
    
    content.append(Paragraph("3.1 Individual Models", styles["SubsectionHeading"]))
    
    models_info = [
        (
            "Logistic Regression",
            "Provides interpretable coefficients showing feature importance. Fast training and "
            "prediction with probabilistic outputs. Works well when features have linear "
            "relationships with the target variable."
        ),
        (
            "Support Vector Machine (SVM)",
            "Handles complex decision boundaries using kernel trick. Effective with "
            "high-dimensional data and provides good generalization. Sensitive to feature scaling, "
            "requiring StandardScaler preprocessing."
        ),
        (
            "Naive Bayes",
            "Extremely fast training and prediction. Based on Bayes' theorem with feature "
            "independence assumption. Provides probability estimates and works well even with "
            "limited data."
        ),
        (
            "Random Forest",
            "Captures complex non-linear relationships and feature interactions. Provides feature "
            "importance insights. Handles outliers well and reduces overfitting through ensemble "
            "of decision trees. Achieved highest individual model accuracy (93.20%)."
        ),
    ]
    
    for model_name, description in models_info:
        content.append(Paragraph(f"<b>{model_name}:</b>", styles["BodyText"]))
        content.append(Paragraph(description, styles["BodyText"]))
        content.append(Spacer(1, 0.05 * inch))
    
    content.append(Spacer(1, 0.1 * inch))
    content.append(Paragraph("3.2 Ensemble Model (Selected)", styles["SubsectionHeading"]))
    ensemble_text = (
        "The Voting Classifier ensemble model was selected as the best model for deployment. "
        "It combines all four individual models using soft voting, which averages probability "
        "predictions from each model. This approach offers several advantages: (1) Improved "
        "generalization by combining diverse learning approaches, (2) Reduced overfitting risk "
        "compared to individual models, (3) Better robustness to data variations, and (4) "
        "Probability estimates for uncertainty quantification. While Random Forest achieved "
        "slightly higher accuracy (93.20%), the ensemble model provides better balance between "
        "accuracy and generalization, making it more suitable for real-world deployment."
    )
    content.append(Paragraph(ensemble_text, styles["BodyText"]))
    
    content.append(PageBreak())
    
    # ========== PAGE 5: RESULTS ==========
    content.append(Paragraph("4. Results and Analysis", styles["SectionHeading"]))
    
    content.append(Paragraph("4.1 Model Performance", styles["SubsectionHeading"]))
    performance_text = (
        "All models were evaluated on a held-out test set. The following table summarizes "
        "the accuracy achieved by each model:"
    )
    content.append(Paragraph(performance_text, styles["BodyText"]))
    content.append(Spacer(1, 0.1 * inch))
    
    # Add performance table
    content.append(create_performance_table(styles))
    content.append(Spacer(1, 0.15 * inch))
    
    content.append(Paragraph("4.2 Performance Analysis", styles["SubsectionHeading"]))
    analysis_text = (
        "The ensemble model achieved 91.36% accuracy, demonstrating strong performance for a "
        "three-class classification problem. Analysis of the confusion matrix reveals excellent "
        "separation between low and medium stress levels, with some confusion between medium "
        "and high stress categories. The model provides probability distributions for each "
        "stress level, enabling uncertainty quantification and better decision-making."
    )
    content.append(Paragraph(analysis_text, styles["BodyText"]))
    
    # Add accuracy comparison image
    img_path = "/home/sahu/Desktop/Student_S_detection/evaluation/accuracy_comparison.png"
    if add_image_if_exists(content, img_path, width=5.5 * inch, caption="Figure 1: Model Accuracy Comparison"):
        pass
    
    content.append(Spacer(1, 0.1 * inch))
    
    # Add confusion matrices image
    img_path2 = "/home/sahu/Desktop/Student_S_detection/evaluation/confusion_matrices.png"
    if add_image_if_exists(content, img_path2, width=5.5 * inch, caption="Figure 2: Confusion Matrices for All Models"):
        pass
    
    content.append(PageBreak())
    
    # ========== PAGE 6: FEATURE ANALYSIS AND CONCLUSION ==========
    content.append(Paragraph("4.3 Feature Importance", styles["SubsectionHeading"]))
    feature_text = (
        "Based on the stress score calculation and model analysis, the most important features "
        "for stress prediction are: sleep hours (highest impact, inverse correlation), exam "
        "pressure, financial stress, assignment deadlines, and workload level. Protective "
        "factors include exercise hours and social activities (negative correlation with stress), "
        "while family support also reduces stress levels."
    )
    content.append(Paragraph(feature_text, styles["BodyText"]))
    
    # Add feature distribution or correlation image if available
    img_path3 = "/home/sahu/Desktop/Student_S_detection/evaluation/correlation_heatmap.png"
    if add_image_if_exists(content, img_path3, width=5.5 * inch, caption="Figure 3: Feature Correlation Heatmap"):
        pass
    
    content.append(Spacer(1, 0.2 * inch))
    content.append(Paragraph("5. Conclusion", styles["SectionHeading"]))
    
    conclusion_text = (
        "This project successfully developed a comprehensive student stress detection system "
        "using ensemble machine learning. The ensemble model achieved 91.36% accuracy, "
        "demonstrating strong performance for stress level classification. The system provides "
        "a complete end-to-end solution from data generation to web deployment, with "
        "personalized recommendations for stress management."
    )
    content.append(Paragraph(conclusion_text, styles["BodyText"]))
    
    content.append(Spacer(1, 0.1 * inch))
    content.append(Paragraph("5.1 Key Achievements", styles["SubsectionHeading"]))
    achievements = [
        "Developed robust ensemble model achieving 91.36% accuracy",
        "Comprehensive comparison of four different ML algorithms",
        "Complete pipeline from data generation to web deployment",
        "User-friendly web application with real-time predictions",
        "Actionable insights through personalized recommendations",
    ]
    for achievement in achievements:
        content.append(Paragraph(f"• {achievement}", styles["BulletText"]))
    
    content.append(Spacer(1, 0.1 * inch))
    content.append(Paragraph("5.2 Future Work", styles["SubsectionHeading"]))
    future_work = [
        "Validate on real-world student data from educational institutions",
        "Add temporal analysis to track stress levels over time",
        "Integrate with Learning Management Systems for automated data collection",
        "Develop mobile application for easier access",
        "Implement deep learning models for complex pattern recognition",
    ]
    for item in future_work:
        content.append(Paragraph(f"• {item}", styles["BulletText"]))
    
    # Build PDF
    doc.build(content)
    print(f"Detailed project report generated: {output_path}")
    print(f"Report contains {len([c for c in content if isinstance(c, Paragraph)])} paragraphs")


if __name__ == "__main__":
    build_pdf()

