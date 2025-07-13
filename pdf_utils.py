"""
PDF generation utilities for the AI Interview Question Generator
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import io

class PDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for the PDF"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
            textColor=colors.darkblue
        ))
        
        # Question style
        self.styles.add(ParagraphStyle(
            name='Question',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            leftIndent=20,
            textColor=colors.black
        ))
        
        # Answer style
        self.styles.add(ParagraphStyle(
            name='Answer',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=15,
            leftIndent=40,
            textColor=colors.grey
        ))
    
    def generate_questions_pdf(self, questions, subject, difficulty):
        """Generate PDF with just questions for downloading before answering"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
        story = []
        
        # Title
        title = Paragraph(f"Interview Questions: {subject}", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Subtitle with difficulty
        subtitle = Paragraph(f"Difficulty Level: {difficulty}", self.styles['CustomSubtitle'])
        story.append(subtitle)
        story.append(Spacer(1, 20))
        
        # Generation info
        generation_info = Paragraph(
            f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
            f"Total Questions: {len(questions)}",
            self.styles['Normal']
        )
        story.append(generation_info)
        story.append(Spacer(1, 30))
        
        # Questions
        for i, question in enumerate(questions, 1):
            # Question number and text
            q_text = f"<b>Question {i}:</b><br/>{question}"
            q_para = Paragraph(q_text, self.styles['Question'])
            story.append(q_para)
            
            # Space for answer
            answer_space = Paragraph(
                "_" * 80 + "<br/>" + "_" * 80 + "<br/>" + "_" * 80 + "<br/>",
                self.styles['Answer']
            )
            story.append(answer_space)
            story.append(Spacer(1, 20))
        
        # Footer
        footer = Paragraph(
            "<i>AI Interview Question Generator - Practice Makes Perfect!</i>",
            self.styles['Normal']
        )
        story.append(Spacer(1, 30))
        story.append(footer)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_evaluation_pdf(self, attempt, questions_answers):
        """Generate PDF report with evaluation results"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
        story = []
        
        # Title
        title = Paragraph("Interview Evaluation Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Attempt details
        details_data = [
            ['Subject:', attempt.subject],
            ['Difficulty:', attempt.difficulty],
            ['Date:', attempt.timestamp.strftime('%B %d, %Y')],
            ['Time:', attempt.timestamp.strftime('%I:%M %p')],
            ['User:', attempt.user.username],
            ['Overall Score:', f"{attempt.average_score:.1f}/5.0"]
        ]
        
        details_table = Table(details_data, colWidths=[2*inch, 3*inch])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(details_table)
        story.append(Spacer(1, 30))
        
        # Score summary
        correct_count = sum(1 for qa in questions_answers if qa.correctness == 'Correct')
        partial_count = sum(1 for qa in questions_answers if qa.correctness == 'Partially Correct')
        incorrect_count = sum(1 for qa in questions_answers if qa.correctness == 'Incorrect')
        
        summary_text = f"""
        <b>Performance Summary:</b><br/>
        Correct Answers: {correct_count}/{len(questions_answers)}<br/>
        Partially Correct: {partial_count}/{len(questions_answers)}<br/>
        Incorrect Answers: {incorrect_count}/{len(questions_answers)}<br/>
        """
        
        summary = Paragraph(summary_text, self.styles['Normal'])
        story.append(summary)
        story.append(Spacer(1, 30))
        
        # Individual questions and answers
        for i, qa in enumerate(questions_answers, 1):
            # Question
            question_text = f"<b>Question {i}:</b><br/>{qa.question_text}"
            question_para = Paragraph(question_text, self.styles['Question'])
            story.append(question_para)
            story.append(Spacer(1, 10))
            
            # User answer
            user_answer_text = f"<b>Your Answer:</b><br/>{qa.user_answer}"
            user_answer_para = Paragraph(user_answer_text, self.styles['Answer'])
            story.append(user_answer_para)
            story.append(Spacer(1, 10))
            
            # Model answer
            model_answer_text = f"<b>Expected Answer:</b><br/>{qa.model_answer}"
            model_answer_para = Paragraph(model_answer_text, self.styles['Answer'])
            story.append(model_answer_para)
            story.append(Spacer(1, 10))
            
            # Evaluation
            score_color = colors.green if qa.score >= 4 else colors.orange if qa.score >= 2 else colors.red
            eval_text = f"""
            <b>Evaluation:</b><br/>
            <font color="{score_color.hexval()}">Score: {qa.score}/5 - {qa.correctness}</font><br/>
            <b>Feedback:</b> {qa.feedback}
            """
            eval_para = Paragraph(eval_text, self.styles['Answer'])
            story.append(eval_para)
            story.append(Spacer(1, 20))
            
            # Page break after every 2 questions (except last)
            if i % 2 == 0 and i < len(questions_answers):
                story.append(PageBreak())
        
        # Footer
        footer = Paragraph(
            f"<i>Generated by AI Interview Question Generator on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>",
            self.styles['Normal']
        )
        story.append(Spacer(1, 30))
        story.append(footer)
        
        doc.build(story)
        buffer.seek(0)
        return buffer

# Global PDF generator instance
pdf_generator = PDFGenerator()
