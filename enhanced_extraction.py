import nltk
import os
nltk_path = os.path.abspath('./nltk_data')
print(f"[DEBUG] Using NLTK data path: {nltk_path}")
nltk.data.path.append(nltk_path)

import re
import spacy
from pyresparser import ResumeParser
from email_validator import validate_email, EmailNotValidError


class EnhancedResumeExtractor:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    def extract_entities_multi_approach(self, file_path, resume_text):
        # Multi-layered entity extraction combining multiple approaches

        # Approach 1: PyResParser
        name_pyr, email_pyr = self.extract_entities_pyresparser(file_path)

        # Approach 2: Regex-based extraction
        name_regex, email_regex, phone_regex = self.extract_entities_regex(resume_text)

        # Approach 3: spaCy NER
        name_spacy, email_spacy = self.extract_entities_spacy(resume_text)

        # Combine results with confidence scoring
        final_name = self.get_best_name(name_pyr, name_regex, name_spacy)
        final_email = self.get_best_email(email_pyr, email_regex, email_spacy)
        final_phone = phone_regex  # Regex is usually best for phone numbers

        return final_name, final_email, final_phone

    def extract_entities_pyresparser(self, file_path):
        # PyResParser approach
        try:
            data = ResumeParser(file_path).get_extracted_data()
            name = data.get("name", "N/A")
            email = data.get("email", "N/A")
            return name, email
        except Exception as e:
            print(f"[PyResParser Error] {e}")
            return "N/A", "N/A"

    def extract_entities_regex(self, text):
        # Regex-based extraction for fallback

        # Email regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)

        # Phone regex (various formats)
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, text)

        # Name extraction (heuristic approach)
        name = self.extract_name_heuristic(text)

        return (name,
                emails[0] if emails else "N/A",
                '-'.join(phones[0]) if phones else "N/A")

    def extract_name_heuristic(self, text):
        # Heuristic name extraction
        lines = text.split('\n')

        # Look for name patterns in first few lines
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if len(line) > 0 and not any(char.isdigit() for char in line):
                # Check if it looks like a name (2-4 words, proper case)
                words = line.split()
                if 2 <= len(words) <= 4 and all(word.isalpha() for word in words):
                    if all(word[0].isupper() for word in words):
                        return line

        return "N/A"

    def extract_entities_spacy(self, text):
        # spaCy NER extraction
        if not self.nlp:
            return "N/A", "N/A"

        doc = self.nlp(text)

        # Extract person names
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

        # Extract emails (spaCy might not catch all, but worth trying)
        emails = []
        for token in doc:
            if "@" in token.text and "." in token.text:
                emails.append(token.text)

        return (names[0] if names else "N/A",
                emails[0] if emails else "N/A")

    def get_best_name(self, name_pyr, name_regex, name_spacy):
        # Choose the best name from multiple sources
        candidates = [name_pyr, name_regex, name_spacy]
        valid_candidates = [name for name in candidates if name != "N/A"]

        if not valid_candidates:
            return "N/A"

        # Prefer names that appear in multiple sources
        for candidate in valid_candidates:
            if valid_candidates.count(candidate) > 1:
                return candidate

        # Otherwise, prefer PyResParser > spaCy > Regex
        if name_pyr != "N/A":
            return name_pyr
        elif name_spacy != "N/A":
            return name_spacy
        else:
            return name_regex

    def get_best_email(self, email_pyr, email_regex, email_spacy):
        # Choose the best email with validation
        candidates = [email_pyr, email_regex, email_spacy]

        for email in candidates:
            if email != "N/A":
                try:
                    validate_email(email)
                    return email
                except EmailNotValidError:
                    continue

        return "N/A"

    def extract_skills(self, text):
        # Extract skills using keyword matching
        # Common technical skills (expand this list based on your domain)
        skills_database = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn'],
            'tools': ['git', 'jenkins', 'jira', 'slack', 'figma', 'photoshop']
        }

        text_lower = text.lower()
        found_skills = []

        for category, skills in skills_database.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills.append(skill)

        return found_skills

    def extract_experience_years(self, text):
        # Extract years of experience
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience\s*:?\s*(\d+)\+?\s*years?'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return max(int(match) for match in matches)

        return 0