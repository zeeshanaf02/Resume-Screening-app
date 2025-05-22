# Install required packages (run these in the terminal if not installed)
# pip install streamlit scikit-learn python-docx PyPDF2 spacy
# python -m spacy download en_core_web_sm

import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import os
import spacy
from collections import defaultdict
import json

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Resume Category Prediction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with dark theme
st.markdown("""
    <style>
    /* Main container and background colors */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .main {
        background-color: #1a1a1a;
        padding: 2rem;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: #2d2d2d;
        padding: 2rem 1rem;
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: none;
        transform: translateY(-2px);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        width: 100%;
        background-color: #2d2d2d;
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        color: #ffffff;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #45a049;
        background-color: #333333;
    }
    
    /* Text areas */
    .stTextArea>div>div>textarea {
        background-color: #2d2d2d;
        border-radius: 8px;
        border: 1px solid #4CAF50;
        padding: 1rem;
        color: #ffffff;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #4CAF50;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    /* Success box */
    .success-box {
        background-color: rgba(76, 175, 80, 0.1);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    /* Prediction result box */
    .prediction-box {
        background-color: rgba(76, 175, 80, 0.1);
        padding: 2rem;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        color: #ffffff;
    }
    .prediction-box h3 {
        color: #4CAF50 !important;
        margin-top: 0;
    }
    .prediction-box p {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    .streamlit-expanderContent {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* Custom text colors */
    .green-text {
        color: #4CAF50 !important;
    }
    .white-text {
        color: #ffffff !important;
    }
    
    /* Work Experience Section */
    .work-experience-card {
        background-color: #2d2d2d;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease;
    }
    .work-experience-card:hover {
        transform: translateX(5px);
    }
    .work-experience-title {
        color: #4CAF50;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .work-experience-meta {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
        color: #a8a8a8;
        font-size: 0.9rem;
    }
    .work-experience-description {
        color: #ffffff;
        line-height: 1.6;
    }
    .team-size-badge {
        background-color: rgba(76, 175, 80, 0.1);
        color: #4CAF50;
        padding: 0.2rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        border: 1px solid #4CAF50;
    }
    .achievements-section {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    .achievement-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.5rem;
    }
    .achievement-item:before {
        content: "•";
        color: #4CAF50;
        margin-right: 0.5rem;
    }
    
    /* Remove default white background from widgets */
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load spacy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    st.error("Please install spacy model by running: python -m spacy download en_core_web_sm")
    st.stop()

# Ensure model files exist before loading them
MODEL_FILES = ['clf.pkl', 'tfidf.pkl', 'encoder.pkl']
for file in MODEL_FILES:
    if not os.path.exists(file):
        st.error(f"Missing file: {file}. Please make sure all required model files are available.")
        st.stop()

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Classifier model
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # TF-IDF vectorizer
le = pickle.load(open('encoder.pkl', 'rb'))  # Label encoder

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)  # Use raw string
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)  # Use raw string
    return cleanText.strip()

# Function to extract text from PDF safely
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''.join([p.extract_text() or '' for p in pdf_reader.pages])  # Ensure no None values
    return text.strip()

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text.strip()

# Function to extract text from TXT with encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text.strip()

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

# Function to extract structured information from resume text
def extract_structured_info(text):
    # Process the text with spacy
    doc = nlp(text)
    
    # Initialize structured data
    structured_data = {
        'contact_info': [],
        'education': [],
        'work_experience': [],
        'skills': set(),
        'languages': set(),
        'sections': defaultdict(list)
    }
    
    # Common section headers in resumes
    section_headers = {
        'education': ['education', 'academic background', 'qualifications'],
        'experience': ['experience', 'work experience', 'employment history', 'work history'],
        'skills': ['skills', 'technical skills', 'competencies'],
        'projects': ['projects', 'project experience'],
        'certifications': ['certifications', 'certificates', 'professional certifications'],
        'languages': ['languages', 'language skills']
    }
    
    # Split text into lines and process
    lines = text.split('\n')
    current_section = 'other'
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        line_lower = line.lower()
        for section, headers in section_headers.items():
            if any(header in line_lower for header in headers):
                current_section = section
                break
                
        # Extract email addresses
        emails = re.findall(r'[\w\.-]+@[\w\.-]+', line)
        if emails:
            structured_data['contact_info'].extend(emails)
            
        # Extract phone numbers
        phones = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', line)
        if phones:
            structured_data['contact_info'].extend(phones)
            
        # Process education-related information
        if current_section == 'education':
            structured_data['education'].append(line)
            
        # Process experience-related information
        elif current_section == 'experience':
            structured_data['work_experience'].append(line)
            
        # Extract skills (including technical terms and tools)
        doc_line = nlp(line)
        for ent in doc_line.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                structured_data['skills'].add(ent.text)
                
        # Add line to appropriate section
        structured_data['sections'][current_section].append(line)
    
    # Convert sets to lists for JSON serialization
    structured_data['skills'] = list(structured_data['skills'])
    structured_data['languages'] = list(structured_data['languages'])
    
    return structured_data

# Function to predict the category of a resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])  # Convert to vector
    predicted_category = svc_model.predict(vectorized_text.toarray())  # Make prediction
    predicted_category_name = le.inverse_transform(predicted_category)  # Decode label
    return predicted_category_name[0]

# Streamlit app layout
def main():
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.markdown("""
        <div class="info-box">
        This app helps you categorize resumes into different job categories using AI.
        Simply upload a resume in PDF, DOCX, or TXT format to get started.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="green-text">### Supported Formats</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <span style="color: #4CAF50">✓</span> PDF files<br>
        <span style="color: #4CAF50">✓</span> DOCX files<br>
        <span style="color: #4CAF50">✓</span> TXT files
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="green-text">### 💡 Tips</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        • Ensure your resume is up-to-date<br>
        • Include relevant skills and experience<br>
        • Use clear formatting<br>
        • Keep file size under 10MB
        </div>
        """, unsafe_allow_html=True)

    # Main content
    st.markdown('<h1 class="white-text">📄 Resume Category Prediction</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Upload your resume below to predict its job category. Our AI model will analyze
    the content and suggest the most suitable category for your profile.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag and drop your resume here",
        type=["pdf", "docx", "txt"],
        help="Maximum file size: 10MB"
    )

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.markdown("""
            <div class="success-box">
            ✅ Successfully extracted the text from the uploaded resume.
            </div>
            """, unsafe_allow_html=True)

            # Extract structured information
            structured_info = extract_structured_info(resume_text)
            
            # Display extracted text in a structured way
            st.markdown('<p class="green-text">### 📑 Resume Content Analysis</p>', unsafe_allow_html=True)
            
            # Create tabs for different views
            overview_tab, sections_tab, raw_tab = st.tabs(["Smart Overview", "Sections", "Raw Text"])
            
            with overview_tab:
                # Contact Information
                if structured_info['contact_info']:
                    st.markdown("""
                    <div class="info-box">
                    <h4 style="color: #4CAF50;">📧 Contact Information</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for contact in structured_info['contact_info']:
                        st.markdown(f"- {contact}")
                # Don't show anything if no contact info exists
                
                # Skills Section
                if structured_info['skills']:
                    st.markdown("""
                    <div class="info-box">
                    <h4 style="color: #4CAF50;">🛠️ Key Skills & Technologies</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    # Display skills in a more visual way
                    skills_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">'
                    for skill in structured_info['skills']:
                        skills_html += f'<span style="background-color: rgba(76, 175, 80, 0.1); padding: 0.3rem 0.8rem; border-radius: 15px; border: 1px solid #4CAF50;">{skill}</span>'
                    skills_html += '</div>'
                    st.markdown(skills_html, unsafe_allow_html=True)
                
                # Education Section
                if structured_info['education']:
                    st.markdown("""
                    <div class="info-box">
                    <h4 style="color: #4CAF50;">🎓 Education</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for edu in structured_info['education']:
                        st.markdown(f"- {edu}")
                
                # Work Experience Section
                if structured_info['work_experience']:
                    st.markdown("""
                    <div class="info-box">
                    <h4 style="color: #4CAF50;">💼 Work Experience</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Initialize collections for different sections
                    languages = []
                    achievements = []
                    extra_curricular = []
                    personal_info = []
                    projects = []
                    current_project = None
                    project_details = []
                    current_section = None
                    
                    for exp in structured_info['work_experience']:
                        exp = exp.strip()
                        if not exp:
                            continue
                            
                        # Skip page numbers and single digits
                        if exp.isdigit():
                            continue
                            
                        # Detect section headers
                        if exp.upper() in ['ACHIEVEMENTS', 'EXTRA CURRICULAR ACTIVITIES', 'PERSONAL INFORMATION', 'LINGUISTIC KNOWLEDGE']:
                            current_section = exp.upper()
                            continue
                            
                        # Handle project information
                        if current_section is None:  # Only process projects outside of other sections
                            if exp.startswith('•'):
                                # If we have a previous project, save it
                                if current_project and project_details:
                                    projects.append({
                                        'title': current_project,
                                        'details': project_details
                                    })
                                current_project = exp.replace('•', '').strip()
                                project_details = []
                            elif exp.startswith('Team Size:'):
                                if current_project:
                                    project_details.append(exp)
                            elif current_project and exp not in ['ACHIEVEMENTS', 'EXTRA CURRICULAR ACTIVITIES', 'PERSONAL INFORMATION', 'LINGUISTIC KNOWLEDGE']:
                                # Combine split lines that end with hyphen
                                if project_details and project_details[-1].endswith('-'):
                                    project_details[-1] = project_details[-1][:-1] + exp
                                else:
                                    project_details.append(exp)
                            continue
                            
                        # Process sections
                        if current_section == 'ACHIEVEMENTS':
                            if exp not in ['ACHIEVEMENTS', 'EXTRA CURRICULAR ACTIVITIES', 'PERSONAL INFORMATION', 'LINGUISTIC KNOWLEDGE']:
                                achievements.append(exp)
                        elif current_section == 'EXTRA CURRICULAR ACTIVITIES':
                            if exp not in ['ACHIEVEMENTS', 'EXTRA CURRICULAR ACTIVITIES', 'PERSONAL INFORMATION', 'LINGUISTIC KNOWLEDGE']:
                                extra_curricular.append(exp)
                        elif current_section == 'LINGUISTIC KNOWLEDGE':
                            # Skip the table header
                            if exp == 'Language Speak Read Write':
                                continue
                            # Extract only the language name from the row
                            if any(word in exp for word in ['Yes', 'No']):
                                lang = exp.split()[0]  # Get just the language name
                                if lang in ['English', 'Kannada', 'Hindi', 'Urdu', 'Telugu', 'Japanese', 'Tamil', 'Malayalam', 'Bengali', 'French', 'German', 'Spanish']:
                                    languages.append(lang)
                    
                    # Add final project if exists
                    if current_project and project_details:
                        projects.append({
                            'title': current_project,
                            'details': project_details
                        })
                    
                    # Display Projects
                    for project in projects:
                        st.markdown(f"""
                        <div class="work-experience-card">
                            <div class="work-experience-title">{project['title']}</div>
                            <div class="work-experience-description">
                                {''.join(f'<div class="achievement-item">{detail}</div>' for detail in project['details'])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display Linguistic Knowledge section if languages exist
                    if languages:
                        st.markdown("""
                        <div class="work-experience-card">
                            <div class="work-experience-title">Linguistic Knowledge</div>
                            <div class="work-experience-description">
                                <div style="display: flex; flex-wrap: wrap; gap: 0.8rem; margin-top: 0.5rem;">
                                    {}
                                </div>
                            </div>
                        </div>
                        """.format(''.join(f'<span class="team-size-badge">{lang}</span>' for lang in languages)), unsafe_allow_html=True)
                    
                    # Display Achievements section if exists
                    if achievements:
                        st.markdown("""
                        <div class="work-experience-card">
                            <div class="work-experience-title">🏆 Achievements</div>
                            <div class="work-experience-description">
                                {}
                            </div>
                        </div>
                        """.format(''.join(f'<div class="achievement-item">{achievement}</div>' for achievement in achievements)), unsafe_allow_html=True)
                    
                    # Display Extra Curricular Activities section if exists
                    if extra_curricular:
                        st.markdown("""
                        <div class="work-experience-card">
                            <div class="work-experience-title">🎨 Extra Curricular Activities</div>
                            <div class="work-experience-description">
                                {}
                            </div>
                        </div>
                        """.format(''.join(f'<div class="achievement-item">{activity}</div>' for activity in extra_curricular)), unsafe_allow_html=True)
            
            with sections_tab:
                for section, content in structured_info['sections'].items():
                    if content:
                        st.markdown(f"""
                        <div class="info-box">
                        <h4 style="color: #4CAF50;">{section.title()}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for item in content:
                            st.markdown(f"- {item}")
            
            with raw_tab:
                st.markdown("""
                <div class="info-box">
                <h4 style="color: #4CAF50;">Complete Resume Text</h4>
                </div>
                """, unsafe_allow_html=True)
                st.text_area("", resume_text, height=300)

            # Display the prediction result
            st.markdown('<p class="green-text">### 🔍 Resume Analysis Result</p>', unsafe_allow_html=True)
            
            # Create a container for the analysis
            st.markdown("""
            <div class="info-box">
            <h4 style="color: #4CAF50;">AI Analysis Summary</h4>
            Our AI model has analyzed your resume content and identified the following:
            </div>
            """, unsafe_allow_html=True)
            
            # Make prediction
            category = pred(resume_text)
            
            # Display prediction in a nice box
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Category</h3>
                <p>{category}</p>
                <div style="margin-top: 1rem; font-size: 0.9rem; color: #a8a8a8;">
                This prediction is based on the analysis of your resume's content, including skills, experience, and keywords.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Add job application options section
            st.markdown('<p class="green-text">### 🚀 Job Application Options</p>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box" style="margin-top: 1rem;">
            <h4 style="color: #4CAF50;">Apply for {category} Jobs</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div class="job-platform-card" style="background-color: #2d2d2d; padding: 1.5rem; border-radius: 8px; border: 1px solid #4CAF50;">
                    <h5 style="color: #4CAF50; margin-bottom: 1rem;">LinkedIn</h5>
                    <p style="color: #ffffff; margin-bottom: 1rem;">Find professional opportunities and network with industry leaders</p>
                    <a href="https://www.linkedin.com/jobs/search/?keywords={category}" target="_blank" style="background-color: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; display: inline-block;">Search {category} Jobs</a>
                </div>
                <div class="job-platform-card" style="background-color: #2d2d2d; padding: 1.5rem; border-radius: 8px; border: 1px solid #4CAF50;">
                    <h5 style="color: #4CAF50; margin-bottom: 1rem;">Indeed</h5>
                    <p style="color: #ffffff; margin-bottom: 1rem;">Search millions of jobs from thousands of company websites</p>
                    <a href="https://www.indeed.com/jobs?q={category}" target="_blank" style="background-color: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; display: inline-block;">Search {category} Jobs</a>
                </div>
                <div class="job-platform-card" style="background-color: #2d2d2d; padding: 1.5rem; border-radius: 8px; border: 1px solid #4CAF50;">
                    <h5 style="color: #4CAF50; margin-bottom: 1rem;">Naukri</h5>
                    <p style="color: #ffffff; margin-bottom: 1rem;">India's leading job portal with millions of job opportunities</p>
                    <a href="https://www.naukri.com/{category}-jobs" target="_blank" style="background-color: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; display: inline-block;">Search {category} Jobs</a>
                </div>
            </div>
            </div>
            """.format(category=category), unsafe_allow_html=True)

            # Add a note about the prediction
            st.markdown("""
            <div class="info-box" style="margin-top: 1rem;">
            <h4 style="color: #4CAF50;">What does this mean?</h4>
            This category represents the primary job sector that best matches your resume's content. 
            The prediction is based on various factors including:
            <ul>
                <li>Key skills and technologies mentioned</li>
                <li>Work experience and responsibilities</li>
                <li>Educational background</li>
                <li>Industry-specific keywords</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
