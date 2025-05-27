import streamlit as st
import google.generativeai as genai
import requests # For GitHub API calls
import os
import json # For pretty printing errors from GitHub API

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
    prompt = f"""
You are an expert AI assistant that translates natural language descriptions into precise GitHub Search API query parameters.
Your goal is to create a query string that can be used as the value for the 'q' parameter in the GitHub API's /search/repositories endpoint.

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
  Example: 'framework language:python OR language:javascript stars:>1000'
- Qualifiers for specific fields: 'in:name', 'in:description', 'in:readme'
  Example: 'security in:readme language:go'
- Sorting: 'sort:stars', 'sort:forks', 'sort:updated' (append -asc or -desc, e.g., 'sort:updated-desc')
  If sorting is requested, include it as part of the query string.

Important Instructions:
1. Your output should be ONLY the search terms and qualifiers, suitable to be appended after `?q=` in a GitHub API URL.
2. Do NOT include `q=` in your output. For example, if the user asks for 'python machine learning libraries', you should output 'python machine learning language:python'.
3. Ensure the query string is well-formed for the GitHub API.
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
    params = {'q': query_params, 'per_page': 30} # Fetch up to 30 results

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
    elif st.session_state.github_error: # Show error if repos list is empty due to an error
        pass # Error is already displayed in col1
    elif st.session_state.gemini_query and not st.session_state.github_error : # If query exists but no repos and no error yet
        pass # Message handled above


st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io), [Google Gemini](https://ai.google.dev/), and [GitHub API](https://docs.github.com/en/rest).")

