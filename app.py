import openai,PyPDF2,io,json,hashlib,re,docx2txt,logging,os,tempfile,pdfplumber,textract
from datetime import datetime
from PyPDF2 import PdfReader
import pandas as pd
import streamlit as st
from PIL import Image
from spire.doc import Document
# from docx import Document
#from dotenv import load_dotenv
#load_dotenv()
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Set Streamlit to use wide mode for the full width of the page
st.set_page_config(page_title="Resume Scanner", page_icon=":page_facing_up:",layout="wide")

# Load the logo image
logo_image = Image.open('assests/HD_Human_Resources_Banner.jpg')

# Optionally, you can resize only if the original size is too large
# For high-definition display, consider not resizing if the image is already suitable
resized_logo = logo_image.resize((1500, 300), Image.LANCZOS)  # Maintain quality during resizing

# Display the logo
st.image(resized_logo, use_container_width=True)  # Dynamically adjust width

# Add styled container for the title and description with a maximum width
st.markdown("""
    <div style="background-color: lightblue; padding: 20px; border-radius: 10px; text-align: center; 
                 max-width: 450px; margin: auto;">
        <p style="color: black; margin-left: 20px; margin-top: 20px; font-weight: bold;font-size: 40px;">CV Screening Portal</p>
        <p style="color: black; margin-left: 15px; margin-top: 20px; font-size: 25px;">AI based CV screening</p>
    </div>
""", unsafe_allow_html=True)

# Change background color using custom CSS
st.markdown(
    """
    <style>
    body {
               background-color: green;

    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hide Streamlit's default footer and customize H1 style
hide_streamlit_style = """
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with open('style.css') as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 
# Example: Add your image handling or other logic here
images = ['6MarkQ']

openai.api_key = st.secrets["secret_section"]["OPENAI_API_KEY"]
#openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF uploaded via Streamlit
def extract_text_from_doc(file_path):
    """
    Extract text from a .doc file using Spire.Doc
    
    Args:
        file_path: Path to the .doc file
    
    Returns:
        str: Extracted text from the file
    """
    # Create a Document object
    document = Document()
    
    # Load the Word document
    document.LoadFromFile(file_path)

    # Extract the text of the document
    document_text = document.GetText()
    document.Close()  # Close the document
    return document_text

def extract_text_from_uploaded_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF, DOCX, or DOC file.

    Args:
        uploaded_file: A file-like object containing the document

    Returns:
        str: Extracted text from the file or an empty string if an error occurs
    """
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        file_bytes = uploaded_file.read()
        text = ""
        if file_type == "pdf":
            # Extract tables using pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                row_text = "\t".join([cell or "" for cell in row])
                                text += row_text + "\n"

            # Extract plain text using PyPDF2
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            return text.strip()
        elif file_type == "docx":
            # Extract text from DOCX
            doc = docx2txt.process(uploaded_file)
            #print(doc)
            return doc.strip()
            
        elif file_type == "doc":
            # For .doc files, we need to save it temporarily and use Spire.Doc
            with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name

            try:
                # Extract text using Spire.Doc
                text = extract_text_from_doc(tmp_file_path)
                return text
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        else:
            st.error("Unsupported file type. Please upload only PDF, DOCX, or DOC files.", icon="🚫")
            return ""

    except Exception as e:
        st.error(f"Error reading file: {e}", icon="🚫")
        logging.error(f"Error reading file: {e}")
        return ""
    
def use_genai_to_extract_criteria(jd_text):
    #print("job description:",jd_text)
    prompt = (
        f"""As an expert HR consultant and also SME(subject matter expert), your task is to analyze the given Job Description (JD) and extract structured information in JSON format. The JD may list job role, education, experience, skills, and certifications (under both Essential Criteria and Desirable Criteria), so you must carefully read the JD text and classify each requirement accordingly.
            Extraction Guidelines:
            "Essential Criteria" includes:

            "education": Qualifications explicitly listed as mandatory or required.
            "experience": Minimum experience required for eligibility.
            "skills": Include only the core skills that are explicitly required for the role. Ensure they align with the required years of experience — for example, if 5+ years of experience is required, include only those skills the candidate is expected to have mastered at that level.
            "certifications": Certifications explicitly required in the JD.
            "Desirable Criteria" includes:

            "education": Qualifications listed as preferred, but not mandatory.
            "experience": Additional experience that is preferred, but not required.
            "skills": Additional or nice-to-have skills.
            "certifications": Certifications mentioned as beneficial, but not required.

            💡 Relevant Education Rule:
            If no specific qualifications are mentioned in the JD, or if only partial qualifications are listed (e.g., just a domain like "CS(computer science)","IT(information technology)" without degree names), then infer and include **relevant standard degrees** (such as "B.E./B.Tech/Engineering in Computer Science", "MCA", or "MSc IT") based on the job domain. Inferred qualifications should be placed in the **Desirable Criteria > education** section unless the JD explicitly states they are required.
           
            **Special Rule for B.E./B.Tech/Engineering:**
            - If the JD specifically mentions **"B.E./B.Tech/Engineering in CS/IT"**, then only candidates with those specializations should be considered under **Essential Criteria**.
            - If no such specialization is mentioned and it only says "B.E./B.Tech/Engineering", then all branches of B.E./B.Tech/Engineering graduation are considered eligible.

            Key Classification Rules:
            Do not assume all certifications belong to "desired_skills" strictly follow these rules.
            **If a certification is mandatory, place it in "Essential Criteria".
            **If a certification is preferred, place it in "Desirable Criteria".
            Similarly, classify education, experience, and skills based on their mention in the JD.
            Expected JSON Output Format:

            {{
            "Essential Criteria": {{
                "education": "<Extracted mandatory education>",
                "experience": "<Extracted mandatory experience>",
                "skills": "<Extracted mandatory skills>",
                "certifications": "<Extracted mandatory certifications>"
            }},
            "Desirable Criteria": {{
                "education": "<Extracted preferred education>",
                "experience": "<Extracted preferred experience>",
                "skills": "<Extracted preferred skills>",
                "certifications": "<Extracted preferred certifications>"
            }}
            }}
            Example JD Classification Logic:
            Example 1:
            JD states: "B.E. / B.Tech /Engineering/ MCA(Master of Computer Applications), MSC IT, MTech is required, while an MBA in IT is preferred."

            Essential Criteria → education: "B.E.(Bachelor of Engineering) / B.Tech(Bachelor of Technology)/MCA(Master of Computer Applications)/ MSC(Master of Science) IT/ MTech(Master of Technology)/MBA(Master of Business Administration) in IT"
            → Desirable Criteria → education: "MBA in IT"
            
            Example 2:
            JD states: "MCSE certification is required. PMP certification is preferred."

            Essential Criteria → certifications: "MCSE"
            Desirable Criteria → certifications: "PMP"
            Example 3:
            JD states: "Minimum 5 years of experience required. Experience in government projects is a plus."

            Essential Criteria → experience: "Minimum 5 years"
            Desirable Criteria → experience: "Experience in government projects"
            This dynamic classification ensures that each requirement is correctly placed in Essential or Desirable Criteria based on the actual JD text, not assumptions.
            following is the jd text
            {jd_text}"""
                )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            return content
        except json.JSONDecodeError:
            st.error("Failed to parse JSON from AI response. Here's the raw response:")
            st.write(content)
            return ""
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return ""

# Function to extract total years of experience using OpenAI's GPT model
@st.cache_data
def extract_experience_from_cv(cv_text):
    # Get the current year dynamically
    x = datetime.now()
    current_year = x.strftime("%m-%Y")
    
    print("CV_Text : ",cv_text)
    # Enhanced prompt template specifically for handling overlapping dates
    prompt_template = f"""
    Please analyze this {cv_text} carefully to calculate the total years of professional experience. Follow these steps:
    
    1. First, list out all date ranges found in chronological order:
       - Replace 'Current' 'Present' or 'till date' with {current_year}
       - Include all years mentioned with positions Formats from following 
       - Format as YYYY-YYYY for each position
       - Format as DD-MM-YYYY - DD-MM-YYYY(For example:10-Jul-12 to 31-Jan-21)
       - Format as YYYY-MM-DD - YYYY-MM-DD(For example:12-Jul-10 to 21-Jan-31)
       - Format as YYYY-DD-MM - YYYY-DD-MM(For example:12-10-Jul to 21-31-Jan)
       - Format as MM-YYYY-MM-YYYY
       - Format as YYYY-MM-YYYY-MM
       - Format as YYYY-YY.
       - if in the cv_text 'start year' and 'present year' or 'end year' are not mentioned,then extract experience from 'cvtext' , if also not mentioned experience in cvtext then return 0.
    
    2. Then, merge overlapping periods :
       - Identify any overlapping years
       - Only count overlapping periods once
       - Create a timeline of non-overlapping periods

    3. Calculate total experience:
       - Sum up all non-overlapping periods.
       - Round to one decimal place Return in Sngle Value.

    "Only replace vague terms like "Current", "Present", or "till date" with the current year ({current_year})."

    "Do NOT alter explicitly mentioned future dates like 05-2025. Accept and process them as-is."
    """

    try:
        # Query GPT-4 API with enhanced system message
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert in analyzing cv_text and calculating professional experience 
                     Pay special attention to overlapping date ranges(If two or more projects within the same company 
                     overlap, merge them into a single continuous range.) and ensure no double-counting of experience.
                     Always show your work step by step."""
                },
                {"role": "user", "content": prompt_template}
            ],
            max_tokens=1000,
            temperature=0
        )
        
        # Extract GPT response
        gpt_response = response.choices[0].message.content.strip()
        #print("gpt_response:",gpt_response)
        # Handle "present" or "current year" in the response
        gpt_response = gpt_response.replace("Present", str(current_year)).replace("current", str(current_year))
        print("gpt_response:",gpt_response)
        # Extract experience and start year using improved regex
        experience_match = re.findall(r'(\d+(?:\.\d+)?)\s*years?', gpt_response, re.IGNORECASE)
        print("experience_match:",experience_match)
        start_year_match = re.search(r'Start Year:\s*(\d{4})', gpt_response, re.IGNORECASE)
        
        # Extract and convert values
        if experience_match:
    # Choose the most relevant value by looking at the context or largest value
            total_experience = max(map(float, experience_match))
            print("total_experience:", total_experience)
            total_experience = str(round(total_experience, 1))  # Round to one decimal place
            print("total_experience2:", total_experience)
        else:
            total_experience = "Not found"

            
        start_year = start_year_match.group(1) if start_year_match else "Not found"
        
        # Debugging output
        # print("\nFull GPT Response:", gpt_response)
        # print("\nExtracted Total Experience:", total_experience)
        # print("Extracted Start Year:", start_year)
        
        return {
            "total_years": total_experience,
            "start_year": start_year,
            "gpt_response": gpt_response
        }
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            "total_years": "Not found",
            "start_year": "Not found",
            "error": str(e)
        }

cv_cache = {}

def normalize_cv_text(cv_text):
    """ Normalize CV text for consistent hashing. """
    return ' '.join(cv_text.strip().split()).lower()

def generate_cv_hash(cv_text):
    """ Generate a consistent hash for the normalized CV text. """
    normalized_text = normalize_cv_text(cv_text)
    return hashlib.sha256(normalized_text.encode()).hexdigest()

def extract_candidate_name_from_cv(cv_text):
    """
    Extracts the candidate's name from the {cv_text} content.

    Args:
        cv_text (str): The {cv_text} content.

    Returns:
        str: Extracted candidate name or 'Unknown Candidate' if not found.
    """
    try:
        # Extract top few lines of the CV for name extraction
        # cv_header = "\n".join(cv_text.splitlines()[:50])  # Top 5 lines for context
        # print("cvheader:",cv_header)
        # Prompt for GPT
        prompt = (
            "Extract the candidate's full name from this cv_text. The name is likely at the top "
            "'Name:', 'Resume of:', 'Name of Staff','name of candidate','first name last name', or similar. Ignore job titles, contact details, and other information.\n\n"
            f"Name:\n{cv_text}"
        )

        # GPT call
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a name extraction specialist. Extract candidate name only from a {cv_text}."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
            temperature=0
        )

        # Extracted name
        candidate_name = response.choices[0].message.content.strip()
        #print('candidate_name:',candidate_name)
        # Validate and return
        if candidate_name and len(candidate_name.split()) <= 4:
            return candidate_name
        return "Unknown Candidate"

    except Exception as e:
        logging.error(f"Error extracting candidate name: {e}")
        return "Unknown Candidate"
cv_cache = {}

def match_cv_with_criteria(cv_text, criteria_json):
    # Generate hash for the CV
    print("job description:",criteria_json)
    cv_hash = generate_cv_hash(cv_text)

    # Check if the CV has already been processed
    if cv_hash in cv_cache:
        return cv_cache[cv_hash]

    # Validate input
    if not criteria_json:
        st.error("Criteria JSON is empty or invalid.")
        return {'cv_text': cv_text}

    try:
        # Extract candidate name from CV text
        candidate_name = extract_candidate_name_from_cv(cv_text)

        # Load criteria JSON
        # criteria = json.loads(criteria_json)

        # Extract total years of experience
        experience_info = extract_experience_from_cv(cv_text)
        #print("total:",experience_info)
        total_years = (experience_info.get("total_years", 0))
        print("total:",total_years)
        try:
            total_years = float(total_years)  # Convert to float if possible
        except ValueError:
            total_years = 0
        
        # Prepare detailed GPT prompt for matching
        prompt = (
            "CRITICAL INSTRUCTIONS: STRICTLY EVALUATE CANDIDATE'S CV AGAINST ESSENTIAL CRITERIA ONLY,DO NOT use Desirable Criteria for evaluation or rejection. \n\n"
            "Evaluation Rules:\n"
            "1. ONLY match against ESSENTIAL CRITERIA from the job description.\n"
            "2. COMPLETELY IGNORE Desirable Criteria.\n"
            "3. STRICTLY Candidate MUST PASS if ALL Essential Criteria are met.\n"
            "4. DO NOT include Desirable Criteria in 'Rejection reasons'.\n"
            "5. STRICTLY EXCLUDE certifications unless they are explicitly listed under Essential Criteria.\n\n"
            "6. Skill Stratification must be consistent across all candidates and strictly based on essential criteria from the job description.\n\n"
            "7. STRICTLY, candidates who fail must have a score of zero and should not receive any skill scores.\n\n"
                "Education Evaluation Logic:\n"
                    "- Normalize education fields from both JD and CV (e.g., 'CSE', 'CS', 'Computer Science', 'Information Technology' should be treated equivalently).\n"
                    "- Match abbreviations (e.g., 'B.E.' = 'Bachelor of Engineering' or 'Engineering' , 'B.Tech' = 'Bachelor of Technology').\n"
                    "- If JD does NOT specify education explicitly or if field/specialization is missing, then DO NOT reject the candidate for education. Instead, if CV contains valid degree (e.g., B.E., B.Tech, MCA, M.Sc IT, MBA in IT), accept it and do NOT return 'None'.\n"
            "Extract and evaluate ONLY the ESSENTIAL CRITERIA from the job description below:\n"
            "Job Description Essential Criteria:\n"
            f"{criteria_json}\n\n"
            "Candidate CV:\n"
            f"{cv_text}\n\n"
            f"Total Years of Experience: {total_years} Years\n\n"
            "IMPORTANT: STRICTLY follow these instructions:\n"
            "- Provide the evaluation in JSON format as follows:\n\n"
            "{\n"
            "  \"Matching Education\": [List of education qualifications from the cv_text that match the ESSENTIAL education criteria from the criteria_json. If the criteria_json specifies a stream (e.g., IT, CSE, Computer Science), only consider CV qualifications in relevant or related fields.],for BE/Btech/Engineering no specific stream mentioned consider this candidate\n"
            "  \"Matching Experience\": [List of matched ESSENTIAL cv_text work experiences that align with the required experience stated in the criteria_json. Only include experiences in the relevant domain or field (e.g., IT, finance, healthcare, etc.) and exclude any experience where the candidate has worked with MPSeDC (M.P. State Electronics Development Corporation Ltd) in the last 6 months.],\n"
            "  \"Matching Skills\": [List of experience from the cv_text that matches the ESSENTIAL experience required by the criteria_json, ensuring it is in the relevant domain or field specified (e.g., IT, finance, healthcare, etc.) and that the candidate has not worked with MPSeDC (M.P. State Electronics Development Corporation Ltd) in the last 6 months. Ensure the skills align with the domain specified in the criteria_json.],\n"
            "  \"Matching Certifications\": [List of matching cv_text certifications from essential criteria only],\n"
            "  \"Rejection reasons\": [List of specific reasons why the candidate was rejected based on essential criteria, such as: 'Missing', 'Irrelevant domain experience (e.g., HR, Accounting)', 'Education not matching', 'Mixed or unclear job roles', or 'Insufficient detail to verify essential criteria'],\n"
            "  \"Skill Stratification\": {\n"
            "    \"<SkillName1>\": score (1–10)"
            "    \"<SkillName2>\": score (1–10)"
            "    \"<SkillName3>\": score (1–10)"
            "    \"<SkillName4>\": score (1–10)"
            "    \"<SkillName5>\": score (1–10)"
            "  } (NOTE: These 5 skills must come ONLY from the Essential Skills listed in the job description. All candidates must be evaluated against the same 5 skills. If the candidate FAILS any essential requirement, all skill scores MUST be 0.),\n"
            "  \"Pass\": true/false (STRICTLY based on essential criteria only)\n"
            "}"
        )

        # Fetch GPT-3.5 analysis
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise job criteria matcher or SME. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0
        )

        # Extract and clean the response
        raw_response = response.choices[0].message.content.strip()
        
        # Additional JSON parsing safeguards
        try:
            matching_results = json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON between first { and last }
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                try:
                    matching_results = json.loads(json_match.group(0))
                except:
                    # Fallback to default results
                    matching_results = {
                        "Matching Education": [],
                        "Matching Experience": [],
                        "Matching Skills": [],
                        "Matching Certifications": [],
                        "Rejection reasons": [],
                        "Skill Stratification": {},
                        "Pass": False
                    }
            else:
                raise ValueError("Unable to parse JSON response")
            
        # Prepare final results
        results = {
            "Candidate Name": candidate_name,
            "Status": "Pass" if matching_results.get('Pass', False) else "Fail",
            "Total Years of Experience": total_years,
            "Matching Education": matching_results.get('Matching Education', []),
            "Matching Experience": matching_results.get('Matching Experience', []),
            "Matching Skills": matching_results.get('Matching Skills', []),
            "Matching Certifications": matching_results.get('Matching Certifications', []),
            "Rejection reasons": matching_results.get('Rejection reasons', []),
            "Stratification": matching_results.get('Skill Stratification', {}),
            "Skill Score": sum(matching_results.get('Skill Stratification', {}).values())
        }
        print("candidate results",results)
        # Cache results
        cv_cache[cv_hash] = results

        return results

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Return a default result structure
        return {
            "Candidate Name": candidate_name or "Unknown Candidate",
            "Status": "Fail",
            "Total Years of Experience": 0,
            "Matching Education": [],
            "Matching Experience": [],
            "Matching Skills": [],
            "Matching Certifications": [],
            "Rejection reasons": ["Failed to process CV"],
            "Stratification": {},
            "Skill Score": 0
        }
    
def get_skill_score_justification(criteria_json, skill, score, cv_text):
    """
    Generate a unique justification for the skill score based on the candidate's resume text
    and evaluation criteria.
    
    :param criteria_json: JSON object with evaluation criteria
    :param skill: Skill being evaluated
    :param score: Score assigned to the skill
    :param cv_text: Candidate's resume text
    :return: Unique justification for the skill score
    """
    # Prepare the prompt with a focus on unique explanations
    if score == 0:
        prompt = (
            f"Evaluate the candidate's skill '{skill}' based on the given criteria {criteria_json}. "
            f"The resume text is: '{cv_text}'. "
            "Explain in 2 distinct bullet points (20 words max each) why the resume does not meet the skill requirements."
            "Reference missing keywords, project details, or experience gaps based on industry standards."
            "Avoid repeating justifications across different resumes."
            "\n*Each bullet point must start on a new line.*"
        )
    else:
        prompt = (
            f"Evaluate the candidate's skill '{skill}' based on the given criteria {criteria_json}. "
            f"The resume text is: '{cv_text}'. "
            "Explain in 2 distinct bullet points (20 words max each) why the resume justifies a score of {score}/10."
            "Highlight specific achievements, project contributions, or relevant skills from the resume."
            "Ensure uniqueness for each candidate—avoid generic reasoning."
            "\n*Each bullet point must start on a new line.*"
        )

    # OpenAI API call
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert recruiter analyzing resumes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.3  # Slightly increase temperature for variability
    )

    # Extract response content
    explanation = response.choices[0].message.content.strip()

    return explanation

def display_pass_fail_verdict(results, cv_text):
    candidate_name = results['Candidate Name']
    skill_scores = results.get("Stratification", {})
    
    # Check pass/fail status
    
    
    pass_fail = results.get("Status", "Fail")
    if pass_fail != 'Pass':
        return  # Skip displaying failed candidates

    with st.container():
        # Pass verdict
        st.markdown(f"### Results for {candidate_name}:")
        st.markdown(f"<h2 style='color: green; font-size: 1em;'>🟢 Profile Eligibility: PASS ✔️</h2>", unsafe_allow_html=True)
        
        # Check for skill scores
        if skill_scores:
            expander = st.expander(f"**Click to view {candidate_name}'s detailed skill assessment**")
            with expander:
                table_data = []
                for skill, score in skill_scores.items():
                    explanation = get_skill_score_justification(criteria_json, skill, score, cv_text)
                    table_data.append({
                        "Candidate Name": candidate_name,
                        "Skill": skill,
                        "Score": f"{score}/10",
                        "Justification": explanation
                    })
                
                df = pd.DataFrame(table_data)
                blankIndex=[''] * len(df)
                df.index=blankIndex
                st.table(df)
                
                # Display Additional Skill Score
                Additional_Skill_Score = round(results.get("Skill Score", 0), 1)
                st.markdown(f"""
                    <h3 style='font-size:20px;'>Additional Skill Score: <strong>{Additional_Skill_Score:.1f}</strong> out of 50</h3>
                """, unsafe_allow_html=True)
        
        # Success message for passing candidates
        st.success("The candidate has passed based on the job description criteria.")
        st.markdown("</div>", unsafe_allow_html=True)


# Function to display and rank candidates in a table
def display_candidates_table(candidates):
    if not candidates:
        st.info("No candidates to display.")
        return

    # Standardize the results dictionary
    processed_candidates = []
    for candidate in candidates:
        processed_candidate = {
            'Candidate Name': candidate.get('Candidate Name', 'Unknown Candidate'),
            'Status': candidate.get('Status', 'Fail'),
            'Total Years of Experience': candidate.get('Total Years of Experience', 0),
            'Skill Score': candidate.get('Skill Score', 0),
            'Matching Education': ', '.join([str(item) for item in candidate.get('Matching Education', [])]) or 'None',
            'Matching Experience': ', '.join([str(item) for item in candidate.get('Matching Experience', [])]) or 'None',
            'Matching Skills': ', '.join([str(item) for item in candidate.get('Matching Skills', [])]) or 'None',
            'Matching Certifications': ', '.join([str(item) for item in candidate.get('Matching Certifications', [])]) or 'None',
            'Rejection reasons': ', '.join([str(item) for item in candidate.get('Rejection reasons', [])]) or 'None',

            'Stratification': str(candidate.get('Stratification', {}))
        }
        processed_candidates.append(processed_candidate)

    # Create DataFrame
    df = pd.DataFrame(processed_candidates)

    # Ensure columns are properly formatted
    # Ensure columns are properly formatted
    df['Total Years of Experience'] = df['Total Years of Experience'].apply(lambda x: float(x)).round(1)
    df['Skill Score'] = df['Skill Score'].apply(lambda x: float(x)).round(0)
    print("skilll score",df['Skill Score'])
    # Format 'Total Years of Experience' for display
    df['Total Years of Experience'] = df['Total Years of Experience'].map('{:.1f}'.format)
    

    # Sorting logic
    df['rank'] = df.apply(lambda row: (
        0 if row['Status'] == 'Pass' else 1, 
        -row['Skill Score'], 
        -float(row['Total Years of Experience'])
    ), axis=1)
    df = df.sort_values(by='rank').drop(columns=['rank'])


    # Reorder columns for better readability
    column_order = [
        'Candidate Name', 'Status', 'Rejection reasons','Total Years of Experience', 
        'Matching Education', 'Matching Experience',  
        'Matching Certifications',  
        'Skill Score', 'Stratification'
    ]
    df = df[column_order]
    
    # Display the table
    st.markdown("## :trophy: Candidate Rankings")
    
    # Styling
    def color_pass_fail(val):
        color = 'lightgreen' if val == 'Pass' else 'lightcoral'
        return f'background-color: {color}'

    # Apply conditional formatting only to the 'Status' column
    styled_df = (
        df.style
        .applymap(color_pass_fail, subset=['Status'])  # Apply styling to 'Status' column
        .format(precision=1)  # Set formatting for numeric columns
    )

# Display styled DataFrame
    st.dataframe(styled_df, hide_index=True)

    return df

# Add a styled box for the file uploaders
st.markdown("""
    <div style="background-color: lightblue; padding: 4px; border-radius: 5px; text-align: left; 
                 max-width: 380px;">
        <p style="color: black; margin-left: 60px; margin-top: 20px; font-weight: bold;font-size: 20px;">Upload Job Description (PDF)</p>
    </div>
""", unsafe_allow_html=True)

# File uploader for Job Description
jd_file = st.file_uploader(" ", type=["pdf", "docx","doc"])

# Header for Candidate Resumes Upload
st.markdown("""
    <div style="background-color: lightblue; padding: 2px; border-radius: 5px; text-align: left; 
                 max-width: 400px;">
        <p style="color: black; margin-left: 55px; margin-top: 20px; font-weight: bold;font-size: 20px;">Upload Candidate Resumes (PDF)</p>
    </div>
""", unsafe_allow_html=True)

# File uploader for Candidate Resumes
cv_files = st.file_uploader("", type=["pdf", "docx","doc"], accept_multiple_files=True)
st.write(f"Total CV'S Uploaded: {len(cv_files)} ")
# Ensure criteria_json is initialized in session state
if 'criteria_json' not in st.session_state:
    st.session_state['criteria_json'] = None

# Button to extract criteria and match candidates
if st.button("Extract Criteria and Match Candidates"):
    if jd_file:
        jd_text = extract_text_from_uploaded_pdf(jd_file)
        print("jdtext:", jd_text)
        if jd_text:
            criteria_json = use_genai_to_extract_criteria(jd_text)
            print("criteria jd text", criteria_json)
            if criteria_json:
                st.session_state.criteria_json = criteria_json
                st.success("Job description criteria extracted successfully.")

                if cv_files:
                    candidates_results = []
                    for cv_file in cv_files:
                        # st.write(f"📄 Processing CV: {cv_file.name}")
                        try:
                            cv_text = extract_text_from_uploaded_pdf(cv_file)
                            if not cv_text.strip():
                                st.error(f"❌ No text extracted from {cv_file.name}")
                                continue

                            # Optional: Try to extract candidate name
                            candidate_name = "Unknown"
                            try:
                                result = match_cv_with_criteria(cv_text, st.session_state.criteria_json)
                                candidate_name = result.get("Candidate Name", "Unknown")
                            except Exception as e:
                                st.error(f"❌ Failed to match {cv_file.name}: {e}")
                                continue

                            if result:
                                result["File Name"] = cv_file.name  # Optional tracking
                                candidates_results.append(result)
                                # st.success(f"✅ Matched: {candidate_name} from {cv_file.name}")
                            else:
                                st.warning(f"⚠️ No match result for {cv_file.name}")
                        except Exception as e:
                            st.error(f"❌ Error processing {cv_file.name}: {e}")
                    
                    # Display candidates table first
                    if candidates_results:
                        display_candidates_table(candidates_results)
                        
                    # Then display individual matching details
                    for result in candidates_results:
                        #st.markdown(f"### Results for {result['Candidate Name']}:")
                        display_pass_fail_verdict(result, cv_text)
                else:
                    st.error("Please upload at least one CV PDF.")
            else:
                st.error("Failed to extract job description criteria.")
        else:
            st.error("The uploaded JD file appears to be empty.")
    else:
        st.error("Please upload a Job Description PDF.")


disclaimer_text = """
<div style="color: grey; margin-top: 50px;">
    <strong>Disclaimer:</strong> Certification validity must be verified manually. 
    These results are AI-generated, and candidates should be thoroughly evaluated 
    through technical interviews. 
    
</div>
"""

st.markdown(disclaimer_text, unsafe_allow_html=True)
       
footer = """
    <style>
        .footer {
            position: fixed;    /* Fixed at the bottom */
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #002F74;
            color: white;
            text-align: center;
            padding: 5px;
            font-weight: bold;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .footer p {
            font-style: italic;
            font-size: 14px;
            margin: 0;
            flex: 1 1 50%;
        }    
    </style>
    <div class="footer">
        <p style="text-align: left;">Copyright v0.3 © 2025 MPSeDC. All rights reserved.</p>
        <p style="text-align: right;">The responses provided on this website are AI-generated. User discretion is advised.</p>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
