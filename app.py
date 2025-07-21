from flask import Flask, render_template, request, send_file, send_from_directory
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
from enhanced_extraction import EnhancedResumeExtractor
import os
from flask.cli import load_dotenv
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')

# Initialize the enhanced extractor
extractor = EnhancedResumeExtractor()


def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"[PDF Error] {e}")
        return ""


def enhanced_similarity_scoring(job_description, resume_text, skills_weight=0.3):

    # Enhanced similarity scoring considering multiple factors

    # Basic TF-IDF similarity
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    # Fit on both job description and resume
    corpus = [job_description, resume_text]
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Calculate cosine similarity
    basic_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Extract skills from both
    job_skills = extractor.extract_skills(job_description)
    resume_skills = extractor.extract_skills(resume_text)

    # Skills matching score
    if job_skills:
        matching_skills = set(job_skills) & set(resume_skills)
        skills_score = len(matching_skills) / len(set(job_skills))
    else:
        skills_score = 0

    # Combined score
    final_score = (1 - skills_weight) * basic_similarity + skills_weight * skills_score

    return final_score * 100, len(matching_skills), len(set(job_skills))


@app.route('/', methods=['GET', 'POST'])
def index():
    results = []

    if request.method == 'POST':
        job_description = request.form['job_description'].lower()
        resume_files = request.files.getlist('resume_files')

        # Ensure upload folder exists
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        scanned_resumes = []
        processing_errors = []

        for resume_file in resume_files:
            try:
                filename = resume_file.filename
                save_path = os.path.join("uploads", filename)
                resume_file.save(save_path)

                # Extract text
                resume_text = extract_text_from_pdf(save_path)

                if not resume_text.strip():
                    processing_errors.append(f"Could not extract text from {filename}")
                    continue

                # Enhanced entity extraction
                name, email, phone = extractor.extract_entities_multi_approach(save_path, resume_text)

                # Extract additional info
                skills = extractor.extract_skills(resume_text)
                experience_years = extractor.extract_experience_years(resume_text)

                scanned_resumes.append({
                    'filename': filename,
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'skills': skills,
                    'experience_years': experience_years,
                    'resume_text': resume_text.lower()
                })

            except Exception as e:
                processing_errors.append(f"Error processing {filename}: {str(e)}")

        # Enhanced ranking
        ranked_resumes = []
        for resume_data in scanned_resumes:
            similarity_score, matching_skills, total_job_skills = enhanced_similarity_scoring(
                job_description, resume_data['resume_text']
            )

            ranked_resumes.append({
                'name': resume_data['name'],
                'email': resume_data['email'],
                'phone': resume_data['phone'],
                'similarity_score': round(similarity_score, 2),
                'matching_skills': matching_skills,
                'total_job_skills': total_job_skills,
                'skills_match_rate': round((matching_skills / total_job_skills * 100) if total_job_skills > 0 else 0,
                                           1),
                'experience_years': resume_data['experience_years'],
                'top_skills': resume_data['skills'][:10],  # Top 10 skills
                'filename': resume_data['filename']  # Add filename for PDF viewing
            })

        # Sort by similarity score
        ranked_resumes.sort(key=lambda x: x['similarity_score'], reverse=True)
        results = ranked_resumes

        # Save enhanced results to CSV
        with open("ranked_resumes.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Rank", "Name", "Email", "Phone", "Similarity_Score", "Filename"
            ])
            for i, resume in enumerate(results, 1):
                writer.writerow([
                    i, resume['name'], resume['email'], resume['phone'],
                    resume['similarity_score'],
                    resume['filename']
                ])

        # Log processing errors
        if processing_errors:
            print("Processing errors:", processing_errors)

    return render_template("index.html", results=results)


@app.route('/download_csv')
def download_csv():
    csv_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "ranked_resumes.csv")
    return send_file(csv_path, as_attachment=True, download_name="ranked_resumes.csv")


# Route to serve and display PDF resumes directly in the browser
@app.route('/view_resume/<filename>')
def view_resume(filename):

    try:
        # Security check - ensure filename doesn't contain path traversal
        if '..' in filename or filename.startswith('/'):
            return "Invalid filename", 400

        # Check if file exists
        file_path = os.path.join("uploads", filename)
        if not os.path.exists(file_path):
            return "Resume not found", 404

        # Send the PDF file with inline display
        return send_from_directory("uploads", filename, as_attachment=False)

    except Exception as e:
        return f"Error opening resume: {str(e)}", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
