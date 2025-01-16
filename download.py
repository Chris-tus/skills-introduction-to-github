import streamlit as st

st.title("Download Your File")

# Extract session_id from query parameters
query_params = st.experimental_get_query_params()
session_id = query_params.get("session_id", [None])[0]

if session_id is None:
    st.error("Missing session ID. Please complete payment.")
else:
    # Retrieve the ZIP file from session state
    zip_key = f"zip_{session_id}"
    if zip_key not in st.session_state:
        st.error("File not found. Please return to the main page to create a new template.")
    else:
        zip_buffer = st.session_state[zip_key]

        # Serve the ZIP file for download
        st.success("Payment confirmed! Your file is ready for download.")
        st.download_button(
            label="Download Your File",
            data=zip_buffer,
            file_name="diamond_dot_template.zip",
            mime="application/zip",
        )

# Button to return to the main page
st.markdown(
    """
    <a href="/" target="_self">
        <button style="padding: 10px 20px; background-color: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer;">
            Generate New Template
        </button>
    </a>
    """,
    unsafe_allow_html=True,
)
