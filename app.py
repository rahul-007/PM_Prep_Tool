import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks import LangChainTracer

# ==============================
# API Key Config
# ==============================

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
LANGSMITH_API_KEY = st.secrets.get("LANGSMITH_API_KEY")

# Initialize LLM with LangSmith monitoring
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# LangSmith Monitoring Setup
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="PM Job Prep", layout="wide")
st.title("🚀 Product Manager Job Prep Assistant")

# Sidebar Navigation
module = st.sidebar.radio(
    "Choose a module:",
    ["Resume Module", "Interview Prep Module", "Mock Interview"]
)

# ==============================
# 1. Resume Module
# ==============================
if module == "Resume Module":
    st.header("📄 Resume Optimizer")

    jd = st.text_area("Paste Job Description")
    resume = st.text_area("Paste Your Resume")

    if st.button("Analyze Resume"):
        if jd and resume:
            prompt = f"""
            You are an expert resume reviewer and ATS analyzer.
            Compare this resume with the job description.
            Provide:
            1. ATS Score (0-100).
            2. Missing keywords/skills.
            3. Suggestions to improve phrasing, formatting, or structure.
            4. Benchmarking insights vs successful PM resumes.
            
            Job Description:
            {jd}

            Resume:
            {resume}
            """
            response = llm.invoke(prompt)
            st.write(response.content)
        else:
            st.warning("Please provide both Job Description and Resume.")

# ==============================
# 2. Interview Prep Module
# ==============================
elif module == "Interview Prep Module":
    st.header("🎯 Interview Practice")

    qn = st.text_input("Enter your interview question")
    action = st.radio(
        "Choose what you want to do:",
        ["I know the answer → I’ll submit", "I need to see ideal answers first"]
    )

    if action == "I know the answer → I’ll submit":
        ans = st.text_area("Write or paste your answer here")
        if st.button("Get Feedback"):
            if ans and qn:
                prompt = f"""
                You are an expert Product Management interviewer.
                Question: {qn}
                Candidate's Answer: {ans}

                Tasks:
                1. Evaluate the answer for each role-level:
                   - Fresher
                   - PM
                   - Sr. PM
                   - Director
                2. Score (0-10) at each level.
                3. Provide detailed feedback.
                4. Share ideal answer using frameworks (STAR, CIRCLES, AARM, etc.) for each role-level.
                """
                response = llm.invoke(prompt)
                st.write(response.content)
            else:
                st.warning("Please provide both question and answer.")

    else:
        if st.button("Show Ideal Answers"):
            if qn:
                prompt = f"""
                You are an expert Product Management interviewer.
                Provide the IDEAL answers for this question:
                {qn}

                Show the answers for different role levels:
                - Fresher
                - PM
                - Sr. PM
                - Director

                Use structured frameworks (STAR, CIRCLES, AARM, etc.) 
                and explain why the focus areas differ at each level.
                """
                response = llm.invoke(prompt)
                st.write(response.content)
            else:
                st.warning("Please provide the question.")

# ==============================
# 3. Mock Interview Module
# ==============================
elif module == "Mock Interview":
    st.header("🗣️ Mock Interview (Lite)")

    # Session State for conversation
    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Your Answer (or type 'done' to finish interview):")

    mock_interview_prompt = """
    You are acting as an interviewer for a Product Manager role. 
    The conversation is a mock interview with the user (candidate). 
    Your job is to:

    1. Ask relevant interview questions and follow-ups based on user input and chat history.
    2. Keep the interview conversational and realistic.
    3. Track which areas are covered:
       - Execution / Problem Solving
       - Metrics & Data
       - Product Design & UX
       - Strategy & Roadmap
       - Behavioral / Leadership
    4. If the user provides a weak or incomplete answer:
       - Give brief constructive feedback.
       - Suggest the next logical area or question to move the interview forward.
    5. If the user has covered all the major areas:
       - Stop the interview naturally.
       - Provide a FINAL REPORT that includes:
           a) Positive areas (what the candidate did well).
           b) Areas for improvement.
           c) Overall role-fit evaluation (Fresher, PM, Sr. PM, Director).

    Keep your tone professional, constructive, and realistic. 
    Do NOT overwhelm the user with too many questions at once. 
    One question or follow-up at a time.
    """

    if st.button("Submit Answer"):
        if user_input:
            # Append user message to history
            st.session_state.history.append(HumanMessage(content=user_input))

            # Build prompt with history
            conversation = "\n".join(
                [f"User: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
                 for m in st.session_state.history]
            )

            final_prompt = f"{mock_interview_prompt}\n\nConversation so far:\n{conversation}"

            response = llm.invoke(final_prompt)

            # Append AI response
            st.session_state.history.append(AIMessage(content=response.content))

    # Display chat history
    for msg in st.session_state.history:
        if isinstance(msg, HumanMessage):
            st.write(f"🧑‍💼 You: {msg.content}")
        else:
            st.write(f"🤖 Interviewer: {msg.content}")

    # Manual Report Generation
    if st.button("📊 Generate Final Report Now"):
        conversation = "\n".join(
            [f"User: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
             for m in st.session_state.history]
        )

        report_prompt = f"""
        Based on the following mock interview conversation, generate a FINAL REPORT.
        Include:
        1. Positive areas (what the candidate did well).
        2. Areas of improvement.
        3. Overall role-fit evaluation (Fresher, PM, Sr. PM, Director).

        Conversation:
        {conversation}
        """

        report = llm.invoke(report_prompt)
        st.subheader("📊 Final Evaluation Report")
        st.write(report.content)
