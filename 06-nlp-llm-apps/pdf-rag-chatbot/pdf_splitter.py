# pdf_splitter.py
import os

def get_relevant_section(question):
    """
    Returns one or more relevant Academic Calendar section PDF filenames based on keywords.
    If no keyword is matched, defaults to 'General_Information.pdf'.
    """
    question_lower = question.lower()
    keywords_map = {
        "fee": "Fees.pdf",
        "tuition": "Fees.pdf",
        "cost": "Fees.pdf",
        "admission": "Admission.pdf",
        "apply": "Admission.pdf",
        "register": "University_Regulations.pdf",
        "enroll": "University_Regulations.pdf",
        "semester": "Sessional_Dates.pdf",
        "session": "Sessional_Dates.pdf",
        "start": "Sessional_Dates.pdf",
        "deadline": "Sessional_Dates.pdf",
        "scholarship": "Scholarships_and_Awards.pdf",
        "bursary": "Scholarships_and_Awards.pdf",
        "aid": "Scholarships_and_Awards.pdf",
        "program": "Programs_and_Courses.pdf",
        "major": "Programs_and_Courses.pdf",
        "minor": "Programs_and_Courses.pdf",
        "course": "Programs_and_Courses.pdf",
        "class": "Programs_and_Courses.pdf",
        "education": "School_of_Education.pdf",
        "business": "Business.pdf",
        "arts": "Arts_Administration.pdf",
        "computer science": "Computer_Science.pdf"
    }

    matched_sections = set()
    for keyword, section in keywords_map.items():
        if keyword in question_lower:
            matched_sections.add(section)
    return list(matched_sections) if matched_sections else ["General_Information.pdf"]
