import streamlit as st
import google.generativeai as genai
import requests # For GitHub API calls
import os
import json # For pretty printing errors from GitHub API
import datetime
import base64

# --- Configuration ---
MODEL_NAME = "gemini-1.5-flash-latest"
GITHUB_API_BASE_URL = "https://api.github.com/search/repositories"

# --- Helper Function to get API Keys ---
def get_api_key(key_name: str, secrets_key: str, env_var_name: str) -> str | None:
    """
    Retrieves an API key from Streamlit secrets or environment variables.
    """
    try:
        api_key = st.secrets[secrets_key]
        if api_key:
            return api_key
    except (FileNotFoundError, KeyError):
        api_key = os.getenv(env_var_name)
        if api_key:
            return api_key
    return None

# --- Function to Call Gemini API using SDK ---
def generate_github_query_from_text(natural_language_input: str) -> tuple[str | None, str | None]:
    """
    Uses Gemini to generate GitHub API query parameters from natural language.
    """
    today = datetime.date.today()
    today_str = today.strftime('%Y-%m-%d')
    
    prompt = f"""
You are an expert AI assistant that translates natural language descriptions into precise GitHub Search API query parameters.
Your goal is to create a query string that can be used as the value for the 'q' parameter in the GitHub API's /search/repositories endpoint.

--- Date/Time Context ---
For your reference, today's date is {today_str}.
--- End Date/Time Context ---

Follow these GitHub Search API query syntax guidelines:
- Basic keywords: 'search terms' (implicitly ANDed)
- Language: 'language:python', 'language:javascript'
- Stars: 'stars:>1000', 'stars:50..100'
- Forks: 'forks:>=500'
- Topics: 'topic:machine-learning', 'topic:data-visualization'
- License: 'license:mit', 'license:apache-2.0'
- User/Organization: 'user:someuser', 'org:someorg'
- Date ranges: 'created:>2022-01-01', 'pushed:>=2023-01-01'
- Boolean operators: Use spaces for AND. For OR, use 'OR'. For NOT, use 'NOT' or '-'.
- Qualifiers for specific fields: 'in:name', 'in:description', 'in:readme'
- Sorting: 'sort:stars', 'sort:forks', 'sort:updated' (append -asc or -desc)

Important Instructions:
1. Your output should be ONLY the search terms and qualifiers, suitable to be appended after `?q=` in a GitHub API URL.
2. Do NOT include `q=` in your output.
3. Pay close attention to the Date/Time Context provided above when handling time-based requests.
4. If the request is too vague or cannot be translated, output 'ERROR: Could not generate a valid query from the input.'

User's request: "{natural_language_input}"

Generated GitHub Query Parameters:
"""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=150,
                top_p=1.0,
                top_k=1
            )
        )
        if not response.parts:
            safety_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else "No parts returned."
            return None, f"ERROR: LLM response blocked or empty. Safety Feedback: {safety_feedback}"
        
        generated_text = response.text.strip()
        if "ERROR:" in generated_text:
            return None, generated_text
        if not generated_text:
            return None, "ERROR: LLM returned an empty query."
        if generated_text.lower().startswith("q="): # Defensive check
            generated_text = generated_text[2:].strip()
            
        return generated_text, None
    except Exception as e:
        return None, f"ERROR: An error occurred during the Gemini API call: {str(e)}"

# --- Function to Fetch Repos from GitHub API ---
def fetch_github_repos(query_params: str, github_pat: str | None) -> tuple[list | None, str | None]:
    """
    Fetches repositories from GitHub API based on the query parameters.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28" # Recommended by GitHub
    }
    if github_pat:
        headers["Authorization"] = f"token {github_pat}"

    # Replace spaces with '+' for URL encoding, though requests usually handles this.
    # query_params_encoded = query_params.replace(" ", "+")
    # Using params dictionary is safer as requests handles encoding.
    params = {'q': query_params, 'per_page': 100} # Fetch up to 30 results

    try:
        response = requests.get(GITHUB_API_BASE_URL, headers=headers, params=params, timeout=20)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        data = response.json()
        return data.get("items", []), None # 'items' contains the list of repos
        
    except requests.exceptions.HTTPError as http_err:
        error_content = "No additional error content."
        try:
            error_content = json.dumps(response.json(), indent=2)
        except json.JSONDecodeError:
            error_content = response.text
        return None, f"ERROR: GitHub API HTTP error: {http_err}\nResponse: {error_content}"
    except requests.exceptions.RequestException as req_err:
        return None, f"ERROR: GitHub API request failed: {req_err}"
    except Exception as e:
        return None, f"ERROR: An unexpected error occurred fetching from GitHub: {str(e)}"

# ... (Keep existing imports, MODEL_NAME, GITHUB_API_BASE_URL, get_api_key, generate_github_query_from_text, fetch_github_repos) ...

def get_repo_details(owner: str, repo_name: str, github_pat: str | None) -> tuple[dict | None, str | None]:
    """Fetches basic details for a repository, including the default branch."""
    url = f"https://api.github.com/repos/{owner}/{repo_name}"
    headers = {"Accept": "application/vnd.github.v3+json", "X-GitHub-Api-Version": "2022-11-28"}
    if github_pat:
        headers["Authorization"] = f"token {github_pat}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as http_err:
        return None, f"ERROR: GitHub API HTTP error fetching repo details: {http_err}"
    except requests.exceptions.RequestException as req_err:
        return None, f"ERROR: GitHub API request failed fetching repo details: {req_err}"

def get_readme_content(owner: str, repo_name: str, github_pat: str | None) -> tuple[str | None, str | None]:
    """Fetches and decodes the README content for a repository."""
    url = f"https://api.github.com/repos/{owner}/{repo_name}/readme"
    headers = {"Accept": "application/vnd.github.v3.raw", "X-GitHub-Api-Version": "2022-11-28"} # Get raw content
    # Alternative: application/vnd.github.v3+json to get JSON with 'content' field (base64)
    # For simplicity with raw, we'll use the .raw media type if available,
    # otherwise, we'd need to parse JSON and decode base64.
    # Let's stick to JSON to handle encoding properly for now.
    headers["Accept"] = "application/vnd.github.v3+json" 

    if github_pat:
        headers["Authorization"] = f"token {github_pat}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        readme_data = response.json()
        if readme_data.get("encoding") == "base64" and readme_data.get("content"):
            decoded_content = base64.b64decode(readme_data["content"]).decode('utf-8')
            return decoded_content, None
        return None, "ERROR: README content not found or not base64 encoded."
    except requests.exceptions.HTTPError as http_err:
        return None, f"ERROR: GitHub API HTTP error fetching README: {http_err}"
    except requests.exceptions.RequestException as req_err:
        return None, f"ERROR: GitHub API request failed fetching README: {req_err}"
    except Exception as e:
        return None, f"ERROR: Unexpected error decoding README: {str(e)}"


def get_file_tree(owner: str, repo_name: str, branch: str, github_pat: str | None) -> tuple[list | None, str | None]:
    """Fetches the file tree for a repository branch."""
    url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{branch}?recursive=1"
    headers = {"Accept": "application/vnd.github.v3+json", "X-GitHub-Api-Version": "2022-11-28"}
    if github_pat:
        headers["Authorization"] = f"token {github_pat}"
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tree_data = response.json()
        if tree_data.get("truncated"):
            st.warning("File tree is too large and has been truncated by GitHub API.")
        
        file_paths = [item["path"] for item in tree_data.get("tree", []) if item.get("type") == "blob"]
        return file_paths, None
    except requests.exceptions.HTTPError as http_err:
        return None, f"ERROR: GitHub API HTTP error fetching file tree: {http_err}"
    except requests.exceptions.RequestException as req_err:
        return None, f"ERROR: GitHub API request failed fetching file tree: {req_err}"

# --- New Function for Q&A with Gemini ---
def answer_readme_question(readme_content: str, question: str) -> tuple[str | None, str | None]:
    """Answers a question based on README content using Gemini."""
    prompt = f"""
You are a helpful AI assistant. Based ONLY on the following README content from a GitHub repository, please answer the user's question.
If the information is not present in the README, state that clearly.

<README_CONTENT>
{readme_content}
</README_CONTENT>

<USER_QUESTION>
{question}
</USER_QUESTION>

Answer:
"""
    try:
        model = genai.GenerativeModel(MODEL_NAME) # Assumes MODEL_NAME is defined globally
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, # Slightly higher for Q&A
                max_output_tokens=500 
            )
        )
        if not response.parts:
            safety_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else "No parts returned."
            return None, f"ERROR: LLM response blocked or empty. Safety Feedback: {safety_feedback}"
        return response.text.strip(), None
    except Exception as e:
        return None, f"ERROR: An error occurred during the Gemini API call for Q&A: {str(e)}"





# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="GitHub Query Agent")
st.title("🤖 GitHub Repository Query Agent")
st.caption("Describe repositories, AI generates the query, then fetch and view them from GitHub.")

# --- API Key Setup ---
gemini_api_key = get_api_key("Gemini API Key", "GEMINI_API_KEY", "GEMINI_API_KEY")
github_pat = get_api_key("GitHub PAT", "GITHUB_PAT", "GITHUB_PAT")

# Gemini API Key
if not gemini_api_key:
    st.sidebar.warning("Gemini API Key not found. Please add it to `.streamlit/secrets.toml` or as an environment variable.")
    gemini_api_key_input = st.sidebar.text_input("Enter Gemini API Key (Required):", type="password")
    if gemini_api_key_input:
        gemini_api_key = gemini_api_key_input
        st.sidebar.success("Using Gemini API Key from input.")
    else:
        st.error("Gemini API Key is required.")
        st.stop()
else:
    st.sidebar.success("Gemini API Key loaded.")

# GitHub PAT
if not github_pat:
    st.sidebar.warning("GitHub PAT not found. Using unauthenticated requests (rate limits apply). Add to `.streamlit/secrets.toml` or env var for best results.")
    github_pat_input = st.sidebar.text_input("Enter GitHub PAT (Recommended):", type="password")
    if github_pat_input:
        github_pat = github_pat_input
        st.sidebar.success("Using GitHub PAT from input.")
else:
    st.sidebar.success("GitHub PAT loaded.")


# Configure Gemini client
try:
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()

# --- Session State Initialization ---
if 'gemini_query' not in st.session_state:
    st.session_state.gemini_query = ""
if 'gemini_error' not in st.session_state:
    st.session_state.gemini_error = ""
if 'github_repos' not in st.session_state:
    st.session_state.github_repos = []
if 'github_error' not in st.session_state:
    st.session_state.github_error = ""
if 'selected_repo_full_name' not in st.session_state:
    st.session_state.selected_repo_full_name = None
if 'readme_content' not in st.session_state:
    st.session_state.readme_content = None
if 'file_list' not in st.session_state:
    st.session_state.file_list = None
if 'repo_overview_error' not in st.session_state:
    st.session_state.repo_overview_error = None
if 'qa_answer' not in st.session_state:
    st.session_state.qa_answer = None
if 'qa_error' not in st.session_state:
    st.session_state.qa_error = None
if 'repo_info_loaded' not in st.session_state: # Flag to know if overview is loaded
    st.session_state.repo_info_loaded = False

# --- Main App Logic ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Describe Desired Repositories")
    natural_input = st.text_area(
        "Enter your description:",
        height=100,
        placeholder="e.g., 'Python libraries for machine learning with >1000 stars, updated last year'"
    )

    if st.button("✨ Generate GitHub Query Parameters", use_container_width=True, type="primary"):
        st.session_state.github_repos = [] # Clear previous GitHub results
        st.session_state.github_error = ""
        if natural_input:
            with st.spinner("🤖 AI is generating query..."):
                query_params, error = generate_github_query_from_text(natural_input)
                st.session_state.gemini_query = query_params
                st.session_state.gemini_error = error
        else:
            st.session_state.gemini_query = ""
            st.session_state.gemini_error = "Please enter a description first."

    if st.session_state.gemini_error:
        st.error(f"Gemini Error: {st.session_state.gemini_error}")
    
    if st.session_state.gemini_query:
        st.subheader("Generated Query Parameters")
        st.code(st.session_state.gemini_query, language="text")
        
        st.markdown("---")
        st.subheader("2. Fetch Repositories from GitHub")
        if st.button("🔍 Fetch Repositories", use_container_width=True):
            if st.session_state.gemini_query:
                with st.spinner("📡 Contacting GitHub API..."):
                    repos, error = fetch_github_repos(st.session_state.gemini_query, github_pat)
                    st.session_state.github_repos = repos if repos else []
                    st.session_state.github_error = error
            else:
                st.session_state.github_error = "No query parameters generated to fetch."

        if st.session_state.github_error:
            st.error(f"GitHub Fetch Error: {st.session_state.github_error}")

with col2:
    st.subheader("3. GitHub Repository Results")
    if not st.session_state.github_repos and not st.session_state.github_error and not st.session_state.gemini_query:
        st.info("Results will appear here after fetching.")
    elif not st.session_state.github_repos and st.session_state.gemini_query and not st.session_state.github_error:
         st.info("Query parameters generated. Click 'Fetch Repositories' to see results.")


    if st.session_state.github_repos:
        st.success(f"Found {len(st.session_state.github_repos)} repositories.")
        for repo in st.session_state.github_repos:
            with st.expander(f"{repo.get('full_name', 'N/A')} (⭐ {repo.get('stargazers_count', 0)})"):
                st.markdown(f"**Description:** {repo.get('description', 'No description provided.')}")
                st.markdown(f"**Language:** {repo.get('language', 'N/A')}")
                st.markdown(f"**Stars:** {repo.get('stargazers_count', 0)}")
                st.markdown(f"**Forks:** {repo.get('forks_count', 0)}")
                st.markdown(f"**Open Issues:** {repo.get('open_issues_count', 0)}")
                st.markdown(f"**URL:** [{repo.get('html_url')}]({repo.get('html_url')})")
                st.markdown(f"**Last Updated:** {repo.get('updated_at', 'N/A')}")
                st.markdown("---") # Optional: Adds a small visual separator
                if st.button("Select for Q&A", key=f"select_{repo.get('id')}", use_container_width=True):
                    # Set the selected repo name
                    st.session_state.selected_repo_full_name = repo.get('full_name')
                    
                    # Clear out any old Q&A data
                    st.session_state.readme_content = None
                    st.session_state.file_list = None
                    st.session_state.repo_overview_error = None
                    st.session_state.qa_answer = None
                    st.session_state.qa_error = None
                    st.session_state.repo_info_loaded = False
                    
                    # Rerun the script immediately to update the UI
                    st.rerun()
                
    elif st.session_state.github_error: # Show error if repos list is empty due to an error
        pass # Error is already displayed in col1
    elif st.session_state.gemini_query and not st.session_state.github_error : # If query exists but no repos and no error yet
        pass # Message handled above

# --- Repository Q&A Section ---
st.markdown("---") # Main separator
st.header("💬 Repository Q&A")

if not st.session_state.selected_repo_full_name:
    st.info("Select a repository from the search results above to enable Q&A.")
else:
    st.subheader(f"Q&A for: `{st.session_state.selected_repo_full_name}`")

    owner, repo_name = st.session_state.selected_repo_full_name.split('/')

    if not st.session_state.repo_info_loaded:
        if st.button("Load Repo Overview (README & File List)", key="load_repo_overview"):
            with st.spinner("Fetching repository overview..."):
                # 1. Get default branch
                repo_data, err = get_repo_details(owner, repo_name, github_pat)
                if err or not repo_data:
                    st.session_state.repo_overview_error = err or "Could not fetch repo details."
                else:
                    default_branch = repo_data.get("default_branch")
                    if not default_branch:
                        st.session_state.repo_overview_error = "Could not determine default branch."
                    else:
                        # 2. Get README
                        readme, readme_err = get_readme_content(owner, repo_name, github_pat)
                        st.session_state.readme_content = readme
                        if readme_err: # Store error but continue to fetch file list
                            st.session_state.repo_overview_error = (st.session_state.repo_overview_error or "") + "\nREADME: " + readme_err

                        # 3. Get File List
                        files, files_err = get_file_tree(owner, repo_name, default_branch, github_pat)
                        st.session_state.file_list = files
                        if files_err:
                            st.session_state.repo_overview_error = (st.session_state.repo_overview_error or "") + "\nFile List: " + files_err

                        st.session_state.repo_info_loaded = True # Mark as loaded
                        if not st.session_state.repo_overview_error: # Clear error if all good
                            st.session_state.repo_overview_error = None

    if st.session_state.repo_overview_error:
        st.error(st.session_state.repo_overview_error)

    if st.session_state.repo_info_loaded and not st.session_state.repo_overview_error :
        if st.session_state.readme_content:
            with st.expander("View README.md", expanded=False):
                st.markdown(st.session_state.readme_content)
        else:
            st.warning("README content could not be loaded or is not available.")

        if st.session_state.file_list:
            with st.expander("View File List", expanded=False):
                # Displaying a long list can be overwhelming. Show a sample or make it searchable.
                # For now, just show the count and a sample.
                st.write(f"Total files found: {len(st.session_state.file_list)}")
                st.text_area("Files:", "\n".join(st.session_state.file_list[:50]), height=200, disabled=True,
                             help="Showing up to 50 files. Full list might be truncated if very large.")
        else:
            st.warning("File list could not be loaded.")

        st.markdown("---")
        st.subheader("Ask about the README")
        qa_question = st.text_input("Your question about the README:", key="qa_question_input")
        if st.button("💬 Ask Gemini", key="ask_readme_button"):
            if qa_question and st.session_state.readme_content:
                with st.spinner("🤖 Gemini is thinking about the README..."):
                    answer, error = answer_readme_question(st.session_state.readme_content, qa_question)
                    st.session_state.qa_answer = answer
                    st.session_state.qa_error = error
            elif not st.session_state.readme_content:
                st.session_state.qa_error = "README content is not loaded to ask questions about."
            else:
                st.session_state.qa_error = "Please enter a question."

        if st.session_state.qa_error:
            st.error(st.session_state.qa_error)
        if st.session_state.qa_answer:
            st.markdown("**Gemini's Answer (based on README):**")
            st.markdown(st.session_state.qa_answer)


