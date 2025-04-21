# -*- coding: utf-8 -*-
"""
Flask Backend for Skill Gap Analyzer.

Receives resume and job description PDFs, runs the skill analysis logic,
and returns the results as JSON.

Requirements:
- Python 3.7+
- Flask (`pip install Flask`)
- google-generativeai library (`pip install google-generativeai`)
- PyPDF2 library (`pip install pypdf2`)
- A Google API Key for the Gemini API (set as an environment variable GOOGLE_API_KEY)

To Run:
1. Save this code as a Python file (e.g., `app.py`).
2. Ensure your original skill analysis functions (extract_text_from_pdf,
   extract_skills_with_gemini, normalize_skills, analyze_skills,
   categorize_and_recommend_skills, print_recommendations) are available
   in the same file or an imported module. For simplicity, I've included
   the necessary functions directly here, adapted from your original script.
3. Install dependencies: `pip install Flask google-generativeai pypdf2`
4. Set your Google API Key environment variable:
   # On Linux/macOS
   export GOOGLE_API_KEY='YOUR_API_KEY'
   # On Windows (Command Prompt)
   set GOOGLE_API_KEY=YOUR_API_KEY
   # On Windows (PowerShell)
   $env:GOOGLE_API_KEY='YOUR_API_KEY'
   (Restart your terminal/IDE after setting the variable if it's not picked up)
5. Run the Flask app: `python app.py`
   The server will typically start on http://127.0.0.1:5000/
"""

import os
import re
import google.generativeai as genai
from PyPDF2 import PdfReader
import json
import logging
import sys
import tempfile # To handle temporary file uploads
from typing import List, Dict, Tuple, Set, Optional, Any
from flask import Flask, request, jsonify # Import Flask components
from werkzeug.utils import secure_filename # For securing filenames
from flask_cors import CORS



# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration & API Setup (Adapted from original script) ---
# Configure logging only once
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Gemini API client
model = None
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        logging.error("FATAL: GOOGLE_API_KEY environment variable not set.")
        logging.error("Please set this variable with your API key before running the script.")
        # In a Flask app, we don't sys.exit immediately, but we should make the API unavailable
        model = None # Ensure model is None if key is missing
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Test a small call or list models to check if the key is valid
        try:
            list(genai.list_models()) # Attempt to list models
            model = genai.GenerativeModel('gemini-1.5-flash')
            logging.info("Gemini API configured successfully using model 'gemini-1.5-flash'.")
        except Exception as api_check_e:
             logging.error(f"Error validating Gemini API key or accessing model: {api_check_e}")
             logging.error("Please ensure your GOOGLE_API_KEY is valid and has access to 'gemini-1.5-flash'.")
             model = None


except Exception as e:
    logging.error(f"Error during initial Gemini API configuration: {e}")
    model = None # Ensure model is None on error

# --- PDF Text Extraction (Adapted from original script) ---
def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extracts text content from a PDF file. Handles file checks.
    Returns text string or None on failure.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"Error: PDF file not found at {pdf_path}")
        return None
    if not os.path.isfile(pdf_path):
        logging.error(f"Error: Path exists but is not a file: {pdf_path}")
        return None
    text = ""
    try:
        reader = PdfReader(pdf_path)
        logging.info(f"Reading PDF: {pdf_path} ({len(reader.pages)} pages)")
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    page_text = page_text.replace('\x00', '')
                    text += page_text + "\n"
                else:
                    logging.warning(f"No text extracted from page {page_num + 1} of {pdf_path}.")
            except Exception as page_error:
                logging.error(f"Error extracting text from page {page_num + 1} of {pdf_path}: {page_error}")
        if not text:
             logging.warning(f"No text could be extracted from the PDF: {pdf_path}.")
             return None
        logging.info(f"Successfully extracted text from {pdf_path}.")
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logging.error(f"Error reading PDF file {pdf_path}: {e}")
        return None

# --- Skill Extraction with Gemini API (Adapted from original script) ---
def extract_skills_with_gemini(document_text: str, document_type: str = "document") -> List[str]:
    """
    Uses the Google Gemini API to extract skills from the provided text.
    Includes enhanced logging for diagnosing JSON parsing issues.
    """
    if not model:
           logging.error("Gemini model was not initialized. Cannot call API for skill extraction.")
           return []
    if not document_text:
        logging.warning(f"Cannot extract skills from empty text for {document_type}.")
        return []
    MAX_CHARS = 20000
    truncated_text = document_text[:MAX_CHARS]
    if len(document_text) > MAX_CHARS:
        logging.warning(f"Document text for {document_type} truncated to {MAX_CHARS} characters for API request.")

    prompt = f"""
    **Objective:** Analyze the provided {document_type} text to identify and extract a comprehensive list of specific skills, technologies, methodologies, tools, and qualifications mentioned.

**Skills to Extract (Examples):**

*   **Technical Skills:**
    *   Programming Languages: Python, Java, C++, JavaScript, SQL, Go
    *   Frameworks/Libraries: React, Angular, Vue.js, Node.js, Django, Flask, Spring Boot, .NET
    *   Databases: PostgreSQL, MySQL, MongoDB, Cassandra, Redis, Oracle
    *   Cloud Platforms: AWS (EC2, S3, Lambda, RDS, EKS), Azure (VMs, Blob Storage, Functions, SQL Database, AKS), GCP (Compute Engine, Cloud Storage, Cloud Functions, Cloud SQL, GKE)
    *   Tools: Git, Docker, Kubernetes, Jenkins, Terraform, Ansible, Jira, Splunk, Grafana
    *   OS/Concepts: Linux, Windows Server, CI/CD, Agile, Scrum, Kanban, REST APIs, Microservices, Data Structures, Algorithms, Machine Learning, IT Security, Network Protocols
*   **Soft Skills:** Communication, Teamwork, Leadership, Problem-Solving, Time Management, Collaboration, Presentation Skills
*   **Certifications:** AWS Certified Solutions Architect, PMP, CISSP, Google Cloud Professional Data Engineer
*   **Languages:** Fluency in specific spoken/written languages (e.g., "Fluent in Spanish")

**Extraction Guidelines:**

1.  Identify *specific* and *distinct* skills, tools, technologies, methodologies, certifications, and relevant qualifications. Prefer specific names (e.g., "AWS S3" if mentioned) over generic terms (e.g., "Cloud Storage").
2.  Focus on actionable skills. Avoid extracting generic sentences, company names, project names, or job titles unless the title itself is a recognized certification (e.g., "PMP certified").
3.  **Normalize Capitalization and Punctuation:** Ensure proper title case or standard technical representation for skills. **For example, convert "it security" to "IT Security", "web design" to "Web Design", "react" to "React.js" (if appropriate context implies the library), and maintain existing correct capitalization like "AWS" or "Python".**
4.  Do *not* infer or invent skills that are not explicitly mentioned or strongly implied by the text context.
5.  Extract the skill term itself, not the sentence containing it.

**Output Format:**

*   **CRITICAL:** Respond *only* with a single, valid JSON list containing strings.
*   Each string in the list must represent *one* distinct skill or qualification identified.
*   **DO NOT** include any introductory text, explanations, summaries, notes, or markdown formatting (like ```json ... ```) surrounding the JSON list. Just the raw JSON list.

**Example Output:**
["Python", "SQL", "Teamwork", "AWS Certified Solutions Architect Associate", "React.js", "Git", "Agile Methodologies", "Docker", "Problem-Solving", "Fluent in Spanish", "IT Security", "Web Design"]

**Input {document_type.capitalize()} Text:**
---
{truncated_text}
---
    ---

    Extracted Skills (JSON list of strings ONLY):
    """
    try:
        logging.info(f"Sending request to Gemini API for {document_type} skills...")
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)

        # --- Robust Response Handling ---
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            logging.error(f"Gemini API request for {document_type} skills blocked. Reason: {response.prompt_feedback.block_reason}")
            if response.prompt_feedback.safety_ratings:
                    for rating in response.prompt_feedback.safety_ratings: logging.error(f"  Safety Rating: Category={rating.category}, Probability={rating.probability}")
            return []
        if not response.candidates:
            logging.warning(f"Gemini API returned no candidates for {document_type} skills.")
            try: finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'; logging.warning(f"  Finish Reason: {finish_reason}")
            except Exception: pass
            return []
        try:
            response_text = response.text.strip()
            logging.info(f"Received response from Gemini API for {document_type} skills.") # Log success *before* parsing
        except ValueError:
             # Handle cases where response.text might raise ValueError
             logging.warning(f"Gemini API response structure issue for {document_type} skills. Could not access response.text safely.")
             if response.candidates: logging.warning(f"Candidate details: {response.candidates[0]}")
             return []
        except Exception as text_extract_err:
            logging.error(f"Error extracting text from Gemini response for {document_type} skills: {text_extract_err}")
            return []
        # --- End Robust Response Handling ---

        # --- JSON Parsing Attempt ---
        json_str = None # Initialize json_str
        try:
            # Try to find the JSON list structure
            start_index = response_text.find('[')
            end_index = response_text.rfind(']')

            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = response_text[start_index : end_index + 1]
                # Attempt to parse the extracted string
                skills = json.loads(json_str)
                if isinstance(skills, list) and all(isinstance(s, str) for s in skills):
                    cleaned_skills = [s.strip() for s in skills if s.strip()]
                    # Successfully parsed! Log success and return.
                    logging.info(f"Successfully parsed {len(cleaned_skills)} skills for {document_type}.")
                    return cleaned_skills
                else:
                    # Parsed something, but not a list of strings
                    logging.warning(f"Parsed JSON for {document_type} skills is not a list of strings. Type: {type(skills)}")
                    # --- MODIFIED LOGGING ---
                    logging.warning(f"Raw Gemini Response Text (Format Error):\n---\n{response_text}\n---")
                    return []
            else:
                # --- MODIFIED LOGGING ---
                logging.warning(f"Could not find JSON list '[]' in Gemini response for {document_type} skills.")
                logging.warning(f"Raw Gemini Response Text (List Not Found):\n---\n{response_text}\n---")
                return []
        except json.JSONDecodeError as json_error:
            # --- MODIFIED LOGGING ---
            logging.error(f"Failed to decode JSON response for {document_type} skills: {json_error}")
            logging.error(f"Substring attempted parsing: {json_str if json_str else 'N/A'}")
            logging.error(f"Raw Gemini Response Text (JSON Error):\n---\n{response_text}\n---")
            return []
        except Exception as parse_error:
             # Catch any other unexpected parsing errors
             logging.error(f"Unexpected error during response parsing for {document_type}: {parse_error}")
             # --- MODIFIED LOGGING ---
             logging.error(f"Raw Gemini Response Text (Parse Error):\n---\n{response_text}\n---")
             return []
        # --- End JSON Parsing Attempt ---

    except genai.types.StopCandidateException as stop_ex:
           logging.error(f"Gemini API call stopped unexpectedly for {document_type} skills: {stop_ex}")
           return []
    except Exception as e:
        logging.error(f"Error calling Gemini API for {document_type} skills: {e}")
        if hasattr(e, 'message'): logging.error(f"  API Error Message: {e.message}")
        return []

# --- Skill Normalization (Adapted from original script) ---
def normalize_skills(skills: List[str]) -> Set[str]:
    """
    Normalizes a list of skills for better comparison.
    """
    normalized = set()
    if not skills: return normalized
    for skill in skills:
        if skill and isinstance(skill, str):
             normalized_skill = skill.lower().strip()
             normalized_skill = re.sub(r'[.,;:]+$', '', normalized_skill).strip() # Remove trailing punctuation
             # Standardize variations
             normalized_skill = re.sub(r'\bjavascript\b', 'js', normalized_skill)
             normalized_skill = re.sub(r'\bjs\b', 'javascript', normalized_skill)
             normalized_skill = re.sub(r'\b\.js\b', ' javascript', normalized_skill)
             normalized_skill = re.sub(r'\breact js\b', 'react', normalized_skill)
             normalized_skill = re.sub(r'\bnode js\b', 'nodejs', normalized_skill)
             normalized_skill = re.sub(r'\bsql server\b', 'mssql', normalized_skill)
             normalized_skill = re.sub(r'\bms sql\b', 'mssql', normalized_skill)
             normalized_skill = re.sub(r'\bpostgres\b', 'postgresql', normalized_skill)
             normalized_skill = re.sub(r'\s+', ' ', normalized_skill).strip() # Consolidate whitespace
             if normalized_skill: normalized.add(normalized_skill)
    count_before = len(skills); count_after = len(normalized)
    if count_before > 0: logging.info(f"Normalized {count_before} raw skills down to {count_after} unique skills.")
    else: logging.info("No raw skills provided for normalization.")
    return normalized

# --- Skill Analysis (Adapted from original script) ---
def analyze_skills(resume_skills: Set[str], jd_skills: Set[str]) -> Dict[str, List[str]]:
    """
    Compares resume skills and job description skills to find matches and gaps.
    """
    resume_skills = resume_skills if resume_skills is not None else set()
    jd_skills = jd_skills if jd_skills is not None else set()
    matched_skills = sorted(list(resume_skills.intersection(jd_skills)))
    missing_skills = sorted(list(jd_skills.difference(resume_skills)))
    additional_skills = sorted(list(resume_skills.difference(jd_skills)))
    logging.info(f"Skill analysis complete: {len(jd_skills)} required, {len(resume_skills)} user has, {len(matched_skills)} matched, {len(missing_skills)} missing, {len(additional_skills)} additional.")
    return { "required": sorted(list(jd_skills)), "user_has": sorted(list(resume_skills)), "matched": matched_skills, "missing": missing_skills, "additional": additional_skills, }

# --- Skill Categorization and Recommendation (Adapted from original script) ---
def categorize_and_recommend_skills(skills_to_learn: List[str]) -> Optional[Dict[str, Any]]:
    """
    Uses Gemini API to categorize skills and suggest learning resources.
    """
    if not skills_to_learn:
        logging.info("No missing skills provided for categorization and recommendation.")
        return None
    if not model:
           logging.error("Gemini model was not initialized. Cannot call API for recommendations.")
           return None

    skills_string = ", ".join(skills_to_learn)
    logging.info(f"Requesting categorization and recommendations for: {skills_string}")

    prompt = f"""
    Given the following list of skills that a person is missing for a job:
    "{skills_string}"

    For each skill in the list, perform the following two tasks:
    1. Categorize the skill eihter as "Technical" or "Soft".
    2. Recommend 1-2 high-quality online learning resources (like specific courses on Coursera, Udemy, edX, LinkedIn Learning, Pluralsight, official documentation, respected tutorials, or books) suitable for a beginner to intermediate level learner trying to acquire this skill for job readiness. Include the name of the resource and the platform/provider.

    Return the result ONLY as a valid JSON dictionary where each key is the skill name (exactly as provided in the input list) and the value is an object containing:
      - "category": A string, either "Technical" or "Soft".
      - "recommendations": A list of objects, where each object has:
          - "name": The name of the course/resource (string).
          - "platform": The platform or provider (e.g., "Coursera", "Udemy", "Official Docs", "Book", "YouTube Channel") (string), also provide the link of the courses you are recommending.

    Example JSON Output Format:
    Also write the response in a pretty format with indentation of 2 spaces with proper puntuation for e.g (it security as IT Security, web design as Web Design).
    {{
      "Python": {{
        "category": "Technical",
        "recommendations": [
          {{"name": "Python for Everybody Specialization", "platform": "Coursera"}},
          {{"name": "Automate the Boring Stuff with Python", "platform": "Book/Udemy"}}
        ]
      }},
      "Teamwork": {{
        "category": "Soft",
        "recommendations": [
          {{"name": "Teamwork Skills: Communicating Effectively in Groups", "platform": "Coursera"}},
          {{"name": "Improving Your Teamwork", "platform": "LinkedIn Learning"}}
        ]
      }},
      "AWS S3": {{
        "category": "Technical",
        "recommendations": [
            {{"name": "AWS S3 Basics", "platform": "AWS Documentation"}},
            {{"name": "Ultimate AWS Certified Solutions Architect Associate", "platform": "Udemy"}}
        ]
      }}
      // ... other skills ...
    }}

    Ensure the output is ONLY the JSON dictionary, with no introductory text, explanations, or markdown formatting.
    """

    try:
        logging.info("Sending request to Gemini API for skill categorization/recommendations...")
        safety_settings = [ # Using the same safety settings
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)

        # --- Robust Response Handling ---
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            logging.error(f"Gemini API request for recommendations was blocked. Reason: {response.prompt_feedback.block_reason}")
            return None
        if not response.candidates:
            logging.warning(f"Gemini API returned no candidates for recommendations. Response: {response}")
            return None
        try:
            response_text = response.text.strip()
            logging.info("Received response from Gemini API for recommendations.")
        except Exception as text_extract_err:
            logging.error(f"Error extracting text from Gemini response for recommendations: {text_extract_err}")
            return None
        # --- End Robust Response Handling ---

        # --- JSON Parsing for Recommendations (WITH ENHANCED LOGGING) ---
        json_str = None # Initialize
        try:
            # Find the JSON object structure more reliably
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')

            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = response_text[start_index : end_index + 1]
                # logging.debug(f"Attempting to parse Recommendation JSON: {json_str[:200]}...") # Optional debug
                recommendation_data = json.loads(json_str)

                # Basic validation of the parsed structure
                if isinstance(recommendation_data, dict):
                    logging.info(f"Successfully parsed recommendations for {len(recommendation_data)} skills.")
                    return recommendation_data
                else:
                    logging.warning(f"Parsed JSON for recommendations is not a dictionary. Type: {type(recommendation_data)}")
                    # --- MODIFIED LOGGING ---
                    logging.warning(f"Raw Gemini Response Text (Rec Format Error):\n---\n{response_text}\n---")
                    return None
            else:
                logging.warning("Could not find valid JSON dictionary '{}' in recommendation response.")
                 # --- MODIFIED LOGGING ---
                logging.warning(f"Raw Gemini Response Text (Rec Dict Not Found):\n---\n{response_text}\n---")
                return None
        except json.JSONDecodeError as json_error:
            logging.error(f"Failed to decode JSON response for recommendations: {json_error}")
            logging.error(f"Substring attempted parsing: {json_str if json_str else 'N/A'}")
             # --- MODIFIED LOGGING ---
            logging.error(f"Raw Gemini Response Text (Rec JSON Error):\n---\n{response_text}\n---")
            return None
        except Exception as parse_error:
             logging.error(f"Unexpected error parsing recommendation response: {parse_error}")
              # --- MODIFIED LOGGING ---
             logging.error(f"Raw Gemini Response Text (Rec Parse Error):\n---\n{response_text}\n---")
             return None
        # --- End JSON Parsing ---

    except genai.types.StopCandidateException as stop_ex:
           logging.error(f"Gemini API call stopped unexpectedly for recommendations: {stop_ex}")
           return None
    except Exception as e:
        logging.error(f"Error calling Gemini API for recommendations: {e}")
        if hasattr(e, 'message'): logging.error(f"  API Error Message: {e.message}")
        return None


# --- Flask Route for Analysis ---
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handles file uploads and triggers the skill analysis.
    Expects 'resume' and 'job_description' files in the form data.
    """
    logging.info("Received request to /analyze endpoint.")

    # Check if Gemini model is initialized
    if model is None:
        error_msg = "Backend not configured: Gemini API key is missing or invalid."
        logging.error(error_msg)
        return jsonify({"error": error_msg}), 500 # Internal Server Error

    # Check if files are present in the request
    if 'resume' not in request.files or 'job_description' not in request.files:
        error_msg = "Missing file(s). Please upload both resume and job description PDFs."
        logging.warning(f"Bad request: {error_msg}")
        return jsonify({"error": error_msg}), 400 # Bad Request

    resume_file = request.files['resume']
    jd_file = request.files['job_description']

    # Check if file names are empty (shouldn't happen with proper frontend)
    if resume_file.filename == '' or jd_file.filename == '':
        error_msg = "One or both selected files have no filename."
        logging.warning(f"Bad request: {error_msg}")
        return jsonify({"error": error_msg}), 400 # Bad Request

    # Secure filenames and save temporarily
    # Use tempfile to handle temporary file creation and cleanup
    resume_path = None
    jd_path = None
    try:
        # Create temporary files to save the uploaded PDFs
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_resume:
            resume_file.save(tmp_resume.name)
            resume_path = tmp_resume.name
            logging.info(f"Saved temporary resume file: {resume_path}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_jd:
            jd_file.save(tmp_jd.name)
            jd_path = tmp_jd.name
            logging.info(f"Saved temporary JD file: {jd_path}")

        # --- Run the Analysis Pipeline ---
        logging.info("Starting analysis pipeline...")
        resume_text = extract_text_from_pdf(resume_path)
        jd_text = extract_text_from_pdf(jd_path)

        if resume_text is None or jd_text is None:
             error_msg = "Failed to extract text from one or both PDF files."
             logging.error(error_msg)
             # Clean up temp files before returning
             if resume_path and os.path.exists(resume_path): os.remove(resume_path)
             if jd_path and os.path.exists(jd_path): os.remove(jd_path)
             return jsonify({"error": error_msg}), 500 # Internal Server Error

        raw_resume_skills = extract_skills_with_gemini(resume_text, "resume")
        raw_jd_skills = extract_skills_with_gemini(jd_text, "job description")

        normalized_resume_skills = normalize_skills(raw_resume_skills)
        normalized_jd_skills = normalize_skills(raw_jd_skills)

        analysis_results = analyze_skills(normalized_resume_skills, normalized_jd_skills)

        missing_skills_list = analysis_results.get("missing", [])
        recommendations_data = categorize_and_recommend_skills(missing_skills_list)

        # Add recommendations to the analysis results for the frontend
        analysis_results["recommendations"] = recommendations_data if recommendations_data is not None else {}

        logging.info("Analysis pipeline completed successfully.")

        # Return the results as JSON
        return jsonify(analysis_results), 200 # OK

    except Exception as e:
        # Catch any unexpected errors during processing
        logging.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during analysis."}), 500 # Internal Server Error
    finally:
        # Clean up temporary files regardless of success or failure
        if resume_path and os.path.exists(resume_path):
            try: os.remove(resume_path)
            except OSError as e: logging.error(f"Error removing temp resume file {resume_path}: {e}")
        if jd_path and os.path.exists(jd_path):
            try: os.remove(jd_path)
            except OSError as e: logging.error(f"Error removing temp JD file {jd_path}: {e}")
        logging.info("Temporary files cleaned up.")


# --- Root Route (Optional, for testing server status) ---
@app.route('/')
def index():
    """Basic route to confirm the server is running."""
    return "Skill Gap Analyzer Backend is running!"

# --- Run the Flask App ---
if __name__ == '__main__':
    # In a production environment, use a production-ready WSGI server
    # like Gunicorn or uWSGI. For local development, app.run() is fine.
    logging.info("Starting Flask development server...")
    app.run() # debug=True provides helpful error messages during development
