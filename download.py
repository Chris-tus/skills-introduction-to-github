import streamlit as st
import io

# Title of the page
st.title("Download Your File")

# Extract session_id from query parameters
query_params = st.experimental_get_query_params()
session_id = query_params.get("session_id", [None])[0]

if session_id is None:
    st.error("Missing session ID. Please complete payment.")
else:
    # Retrieve ZIP file from session state
    zip_key = f"zip_{session_id}"
    if zip_key not in st.session_state:
        st.error("File not found. Please generate a new template.")
    else:
        # Retrieve the ZIP file
        zip_buffer = st.session_state[zip_key]

        # Display success message
        st.success("Payment confirmed! Your file is ready for download.")

        # Download button
        st.download_button(
            label="Download Your File",
            data=zip_buffer,
            file_name="diamond_dot_template.zip",
            mime="application/zip",
        )

# Add a "Generate New Template" button to return to the main page
if st.button("Generate New Template"):
    st.session_state.pop("download_session_id", None)  # Clear the session ID
    st.experimental_set_query_params()  # Clear query params
    st.experimental_rerun()  # Redirect to the main page
