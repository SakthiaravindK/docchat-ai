import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DocChat AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 DocChat AI")
st.write("Document Q&A and Toxic Comment Detection")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2 = st.tabs(["📄 Document Q&A", "🛡️ Toxic Comment Detection"])

# ----------------------------
# Document Q&A
# ----------------------------
with tab1:
    st.subheader("Ask Questions from Document")

    document = st.text_area(
        "Paste your document here",
        height=220,
        placeholder="Paste document content..."
    )

    question = st.text_input(
        "Ask your question",
        placeholder="Example: What is Python?"
    )

    if st.button("Ask", key="ask_btn"):
        if not document.strip():
            st.error("Document cannot be empty")
        elif not question.strip():
            st.error("Question cannot be empty")
        else:
            try:
                with st.spinner("Generating answer..."):
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "document": document,
                            "question": question
                        },
                        timeout=30
                    )

                result = response.json()
                answer = result.get("answer", "No answer returned")

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })

                st.success("Answer generated")
                st.write(answer)

            except requests.exceptions.RequestException:
                st.error("FastAPI server is offline. Please start the backend.")

    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Chat History")

        for item in st.session_state.chat_history:
            st.markdown(f"**You:** {item['question']}")
            st.markdown(f"**DocChat AI:** {item['answer']}")
            st.markdown("---")


# ----------------------------
# Toxic Comment Detection
# ----------------------------
with tab2:
    st.subheader("Check Comment Toxicity")

    comment = st.text_area(
        "Enter comment",
        height=150,
        placeholder="Example: You are stupid"
    )

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

    if st.button("Check Toxicity", key="tox_btn"):
        try:
            with st.spinner("Checking toxicity..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"comment": comment},
                    timeout=30
                )

            result = response.json()

            label = result.get("label", "unknown")
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "No explanation")

            if confidence >= threshold:
                st.success(f"Label: {label}")
                st.write(f"Confidence: {confidence}")
                st.write(f"Explanation: {explanation}")
            else:
                st.warning("Low confidence result")
                st.write(result)

        except requests.exceptions.RequestException:
            st.error("FastAPI server is offline. Please start the backend.")