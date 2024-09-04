import streamlit as st
import openai
import PyPDF2
import io
import json

# Set your OpenAI API key securely
openai.api_key = 'sk-proj-NduY6NQbG0f2uR6n22F9T3BlbkFJqNztRCFEE5TPFh4vB6Ym'  # Replace with your actual OpenAI API key

# Function to extract text from PDF uploaded via Streamlit
def extract_text_from_uploaded_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Function to use GenAI to extract criteria from job description
def use_genai_to_extract_criteria(jd_text):
    prompt = (
        "Extract and structure the following details from the job description: "
        "1. Education requirements "
        "2. Required experience "
        "3. Mandatory skills "
        "4. Certifications "
        "5. Desired skills (for brownie points). "
        "The job description is as follows:\n\n"
        f"{jd_text}\n\n"
        "Please provide the response as a JSON object. For example:\n"
        "{\"education\": \"Bachelor's Degree, Master's Degree\", "
        "\"experience\": \"5 years experience in data science\", "
        "\"skills\": \"Python, SQL, Machine Learning\", "
        "\"certifications\": \"AWS Certified, PMP\", "
        "\"desired_skills\": \"Deep Learning, NLP\"}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.5
    )

    content = response.choices[0].message['content'].strip()

    try:
        json_object = json.loads(content)
        return content
    except json.JSONDecodeError:
        st.error("Failed to parse JSON from AI response. Here's the raw response:")
        st.write(content)
        return ""

# Function to match CV with job description criteria using GenAI
def match_cv_with_criteria(cv_text, criteria_json):
    if not criteria_json:
        st.error("Criteria JSON is empty or invalid.")
        return {}

    try:
        criteria = json.loads(criteria_json)
        prompt = (
            "Given the job description criteria and the candidate's CV text, "
            "please match the following: "
            "1. Which education qualifications from the job description are present in the CV? "
            "2. Which experiences from the job description are present in the CV? "
            "3. Which mandatory skills and certifications from the job description are present in the CV? "
            "4. Which desired skills from the job description are present in the CV? "
            "5. Identify missing qualifications, experiences, skills, or certifications from the CV. "
            "Also, assign a score (0 to 10) for each desired skill based on the extent of match. "
            "The job description criteria are as follows:\n\n"
            f"{criteria_json}\n\n"
            "The CV text is as follows:\n\n"
            f"{cv_text}\n\n"
            "Please provide the response in the following format: "
            "{\"matching_education\": [...], \n"
            "\"matching_experience\": [...], \n"
            "\"matching_skills\": [...], \n"
            "\"matching_certifications\": [...], \n"
            "\"missing_points\": [...], \n"
            "\"brownie_points\": [{\"skill\": \"Deep Learning\", \"score\": 8}, ...]}."
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5
        )

        return response.choices[0].message['content'].strip()

    except json.JSONDecodeError as e:
        st.error(f"Error parsing criteria JSON: {e}")
        return {}

# Streamlit app
st.set_page_config(page_title="Resume Scanner", page_icon=":page_facing_up:", layout="wide")
st.title("Resume Scanner :mag_right:")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    st.markdown("[Extract Criteria](#extract-criteria)", unsafe_allow_html=True)
    st.markdown("[Upload CVs](#upload-cvs)", unsafe_allow_html=True)

# Upload job description PDF
st.header("Extract Criteria")
st.write("Upload the Job Description PDF and extract the necessary criteria.")

jd_file = st.file_uploader("Upload Job Description PDF", type="pdf")

if jd_file:
    jd_text = extract_text_from_uploaded_pdf(jd_file)
    st.markdown("### Extracted Text from Job Description:")
    st.write(jd_text)

    if st.button("Extract Criteria"):
        if jd_text:
            genai_output = use_genai_to_extract_criteria(jd_text)
            st.markdown("### Extracted Criteria:")
            st.json(genai_output)

            st.session_state.criteria_json = genai_output
        else:
            st.error("No text extracted from the job description PDF.")

# Upload CVs for validation
st.header("Upload CVs")
st.write("Upload one or multiple CVs to match against the job description criteria.")

cv_files = st.file_uploader("Upload CVs (multiple allowed)", type="pdf", accept_multiple_files=True)

if 'criteria_json' in st.session_state and cv_files:
    for cv_file in cv_files:
        cv_text = extract_text_from_uploaded_pdf(cv_file)
        st.markdown(f"### Extracted Text from CV: `{cv_file.name}`")
        st.write(cv_text)

        if cv_text:
            matching_results = match_cv_with_criteria(cv_text, st.session_state.criteria_json)
            st.markdown(f"### Matching Results for `{cv_file.name}`")

            # Beautify the matching results
            try:
                results = json.loads(matching_results)

                st.markdown("#### Education Matches:")
                st.write(results.get("matching_education", "None"))

                st.markdown("#### Experience Matches:")
                st.write(results.get("matching_experience", "None"))

                st.markdown("#### Skill Matches:")
                st.write(results.get("matching_skills", "None"))

                st.markdown("#### Certification Matches:")
                st.write(results.get("matching_certifications", "None"))

                st.markdown("#### Missing Points:")
                st.write(results.get("missing_points", "None"))

                st.markdown("#### Brownie Points:")
                brownie_points = results.get("brownie_points", [])
                for bp in brownie_points:
                    skill = bp.get("skill", "")
                    score = bp.get("score", 0)
                    st.markdown(f"**{skill}**: {score}/10")  # Display the skill name and score
                    st.progress(score / 10)  # Display the progress bar

            except json.JSONDecodeError:
                st.error("Error parsing the matching results.")
        else:
            st.error(f"No text extracted from {cv_file.name}.")
else:
    if 'criteria_json' not in st.session_state:
        st.info("Please upload a job description PDF and extract criteria first.")