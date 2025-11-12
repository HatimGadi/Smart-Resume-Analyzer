import os
from PyPDF2 import PdfReader
from jinja2 import Environment, FileSystemLoader
import pdfkit
import google.generativeai as genai
from dotenv import load_dotenv

# ------------------ CONFIGURATION ------------------

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

env = Environment(loader=FileSystemLoader("templates"))

config = pdfkit.configuration(
    wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
)

# ------------------ FUNCTIONS ------------------

def extract_text_from_pdf(file):
    """Extracts readable text from a PDF file."""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def enhance_experience_with_ai(experience_text):
    """Uses Google Gemini AI to rewrite or improve experience points."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        prompt = f"""
        Improve the following resume experience points to look professional, ATS-friendly,
        and impactful. Use bullet points and strong action verbs. Keep it concise.

        Experience:
        {experience_text}
        """

        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else experience_text

    except Exception as e:
        return f"⚠️ AI Suggestion Error: {str(e)}"

def render_resume_pdf(resume_data, template_choice="ats", preview=False):
    """Renders the resume into HTML or PDF using Jinja2 and pdfkit."""
    try:
        template = env.get_template(f"template_{template_choice}.html")
        html_content = template.render(resume_data)

        if preview:
            return html_content  

        options = {
            "page-size": "A4",
            "encoding": "UTF-8",
            "enable-local-file-access": ""
        }

        pdf_bytes = pdfkit.from_string(
            html_content, False, options=options, configuration=config
        )
        return pdf_bytes

    except Exception as e:
        return f"Error generating PDF: {str(e)}"

