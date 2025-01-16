# Download Page
st.title("Download Your File")

# Extract session_id from query parameters
query_params = st.experimental_get_query_params()
session_id = query_params.get("session_id", [None])[0]

if session_id is None:
    st.error("Invalid or missing session ID. Please complete payment.")
else:
    # Validate the session ID
    if session_id != st.session_state.get("download_session_id"):
        st.error("Payment validation failed. Please complete payment again.")
    else:
        # Generate the ZIP file for download
        with io.BytesIO() as zip_buffer:
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                zf.writestr("color_dot_map.png", "Dummy content for color dot map.")
                zf.writestr("number_dot_map.png", "Dummy content for number dot map.")
                zf.writestr("project_specification.pdf", "Dummy content for specification PDF.")

            zip_buffer.seek(0)

            # Provide the download button
            st.success("Payment confirmed! Your file is ready for download.")
            st.download_button(
                label="Download",
                data=zip_buffer,
                file_name="diamond_dot_template.zip",
                mime="application/zip",
            )
