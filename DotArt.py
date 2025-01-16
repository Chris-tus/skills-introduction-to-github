import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import numpy as np
import io
import zipfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from urllib.parse import urlencode

# Function to reduce colors using K-Means
def reduce_colors(image, num_colors):
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)  # Flatten to 2D array
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
    new_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_.reshape(img_array.shape[:2])
    return new_colors, labels

# Streamlit UI
st.set_page_config(layout="wide")
col_title = st.columns([4])[0]

with col_title:
    st.title("Diamond Dot Template Creator")

# File uploader (disappears after a file is uploaded)
if "uploaded_file" not in st.session_state:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
else:
    uploaded_file = st.session_state.uploaded_file

# Sidebar Settings
st.sidebar.header("Settings")
# Title Input Option
st.sidebar.subheader("Project Title")
project_title = st.sidebar.text_input("Enter your project title:", "My Diamond Dot Template")

# Top Section: Number of Colors and Rhinestone Size
top_col1, top_col2 = st.sidebar.columns(2)
with top_col1:
    num_colors = st.number_input("Number of Colors", min_value=2, max_value=50, value=6, step=1, label_visibility="visible")
with top_col2:
    rhinestone_size = st.number_input("Rhinestone Size (mm)", min_value=1.0, max_value=10.0, value=3.0, step=0.1, label_visibility="visible")

# Middle Section: Dot Shape and Alignment Style
middle_col1, middle_col2 = st.sidebar.columns(2)

# Dot Shape Selection
with middle_col1:
    st.subheader("Dot Shape")
    dot_shape = st.radio(
        "Shape:",
        options=["Circle", "Square", "Diamond"],
        format_func=lambda shape: {
            "Circle": "● Circle",
            "Square": "■ Square",
            "Diamond": "◆ Diamond",
        }[shape],
        label_visibility="collapsed",
    )

# Alignment Selection
with middle_col2:
    st.subheader("Alignment")
    dot_alignment = st.radio(
        "Style:",
        options=["Honeycomb", "Straight"],
        format_func=lambda style: {
            "Honeycomb": "Honeycomb",
            "Straight": "Straight",
        }[style],
        label_visibility="collapsed",
    )

# Display the selected alignment for confirmation
st.sidebar.write(f"Selected alignment: {dot_alignment}")

# Bottom Section: Image Dimensions
bottom_col1, bottom_col2 = st.sidebar.columns(2)
if "image_width" not in st.session_state:
    st.session_state.image_width = 30.0
if "image_height" not in st.session_state:
    st.session_state.image_height = 30.0
aspect_ratio = st.session_state.image_height / st.session_state.image_width

# Width and Height Inputs with Synchronization
aspect_ratio = st.session_state.image_height / st.session_state.image_width if "image_height" in st.session_state and st.session_state.image_width else 1.0

# Width input
with bottom_col1:
    new_width = st.number_input(
        "Width (cm)", 
        min_value=5.0, 
        max_value=100.0, 
        value=st.session_state.get("image_width", 30.0), 
        step=1.0, 
        label_visibility="visible"
    )
    if "image_width" not in st.session_state or new_width != st.session_state.image_width:
        st.session_state.image_width = new_width
        st.session_state.image_height = new_width * aspect_ratio

# Height input
with bottom_col2:
    new_height = st.number_input(
        "Height (cm)", 
        min_value=5.0, 
        max_value=100.0, 
        value=st.session_state.get("image_height", 30.0), 
        step=1.0, 
        label_visibility="visible"
    )
    if "image_height" not in st.session_state or new_height != st.session_state.image_height:
        st.session_state.image_height = new_height
        st.session_state.image_width = new_height / aspect_ratio

# Logic to process uploaded file
if uploaded_file:
    col1, col2 = st.columns([3, 1])
    img = Image.open(uploaded_file).convert("RGB")

    with col2:
        st.subheader("Uploaded Image")
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col1:
        # Aspect ratio logic
        original_width, original_height = img.size
        aspect_ratio = original_height / original_width
        st.session_state.image_height = st.session_state.image_width * aspect_ratio

        # Resize image
        resolution = 50
        resized_width = resolution
        resized_height = int(resolution * (st.session_state.image_height / st.session_state.image_width))
        img = img.resize((resized_width, resized_height))

        # Extract colors
        reduced_colors, labels = reduce_colors(img, num_colors)

        # Rhinestone settings
        rhinestones = []
        ignore_colors = []
        for i, color in enumerate(reduced_colors):
            color_hex = "#{:02x}{:02x}{:02x}".format(*color)
            st.sidebar.subheader(f"Rhinestone {i + 1}")
            color_picker = st.sidebar.color_picker(f"Rhinestone Color {i + 1}", value=color_hex)
            ignore = st.sidebar.checkbox(f"Ignore Rhinestone {i + 1}", value=False)
            ignore_colors.append(ignore)
            rhinestones.append({"color": color_picker})

        # Dimensions in pixels
        cm_to_pixels = 37.795
        final_width_px = int(st.session_state.image_width * cm_to_pixels)
        final_height_px = int(st.session_state.image_height * cm_to_pixels)
        size_px = int(rhinestone_size * cm_to_pixels / 10)
        step_x = size_px + 2
        step_y = size_px + 2

        def add_scale_bar(image, width_cm, height_cm, pixels_width, pixels_height):
            """Adds a scale bar and dimensions to the image."""
            draw = ImageDraw.Draw(image)
            scale_bar_y = pixels_height + 20
            scale_bar_start_x = 10
            scale_bar_end_x = pixels_width - 10
            bar_height = 10

            # Draw the scale bar
            draw.rectangle(
                [scale_bar_start_x, scale_bar_y, scale_bar_end_x, scale_bar_y + bar_height],
                fill="black",
            )

            # Add scale markers
            num_markers = 5
            marker_interval = (scale_bar_end_x - scale_bar_start_x) / num_markers
            marker_label_interval = width_cm / num_markers

            for i in range(num_markers + 1):
                marker_x = scale_bar_start_x + i * marker_interval
                marker_label = f"{i * marker_label_interval:.1f} cm"
                draw.line(
                    [(marker_x, scale_bar_y), (marker_x, scale_bar_y + bar_height)],
                    fill="white",
                    width=2,
                )
                draw.text(
                    (marker_x - 10, scale_bar_y + bar_height + 5),
                    marker_label,
                    fill="black",
                    font=ImageFont.load_default(),
                )

            # Add finished dimensions
            dimensions_text = f"Finished Dimensions: {width_cm:.1f} cm x {height_cm:.1f} cm"
            draw.text(
                (pixels_width / 2 - len(dimensions_text) * 3, scale_bar_y + bar_height + 30),
                dimensions_text,
                fill="black",
                font=ImageFont.load_default(),
            )

        # Generate Color Dot Map with Scale Bar
        st.subheader("Colour Dot Map")
        color_dot_img = Image.new("RGB", (final_width_px, final_height_px + 50), "white")
        draw_color = ImageDraw.Draw(color_dot_img)

        for i, rhinestone in enumerate(rhinestones):
            if ignore_colors[i]:
                continue
            for row in range(0, final_height_px, step_y):
                for col in range(0, final_width_px, step_x):
                    col_offset = (
                        size_px // 2 if (row // step_y) % 2 == 1 and dot_alignment == "Honeycomb" else 0
                    )
                    x = col + col_offset
                    y = row
                    if x + size_px > final_width_px or y + size_px > final_height_px:
                        continue
                    label_y = min(y * labels.shape[0] // final_height_px, labels.shape[0] - 1)
                    label_x = min(x * labels.shape[1] // final_width_px, labels.shape[1] - 1)
                    if labels[label_y][label_x] == i:
                        if dot_shape == "Circle":
                            draw_color.ellipse([x, y, x + size_px, y + size_px], fill=rhinestone["color"])
                        elif dot_shape == "Square":
                            draw_color.rectangle([x, y, x + size_px, y + size_px], fill=rhinestone["color"])
                        elif dot_shape == "Diamond":
                            diamond_coords = [
                                (x + size_px / 2, y),  # Top
                                (x, y + size_px / 2),  # Left
                                (x + size_px / 2, y + size_px),  # Bottom
                                (x + size_px, y + size_px / 2),  # Right
                            ]
                            draw_color.polygon(diamond_coords, fill=rhinestone["color"])

        # Add scale bar
        add_scale_bar(color_dot_img, st.session_state.image_width, st.session_state.image_height, final_width_px, final_height_px)
        st.image(color_dot_img, caption="Generated Rhinestone Map (Color Dot Map with Scale)", use_container_width=True)

        # Generate Number Dot Map with Scale Bar
        col2.subheader("Number Dot Map")
        numbers_img = Image.new("RGB", (final_width_px, final_height_px + 50), "white")
        draw_numbers = ImageDraw.Draw(numbers_img)

        try:
            font = ImageFont.truetype("arial.ttf", size=int(size_px * 0.5))
        except IOError:
            font = ImageFont.load_default()

        for i, rhinestone in enumerate(rhinestones):
            if ignore_colors[i]:
                continue
            for row in range(0, final_height_px, step_y):
                for col in range(0, final_width_px, step_x):
                    col_offset = (
                        size_px // 2 if (row // step_y) % 2 == 1 and dot_alignment == "Honeycomb" else 0
                    )
                    x = col + col_offset
                    y = row
                    if x + size_px > final_width_px or y + size_px > final_height_px:
                        continue
                    label_y = min(y * labels.shape[0] // final_height_px, labels.shape[0] - 1)
                    label_x = min(x * labels.shape[1] // final_width_px, labels.shape[1] - 1)
                    if labels[label_y][label_x] == i:
                        text = str(i + 1)
                        text_bbox = font.getbbox(text)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        text_x = x + (size_px - text_width) / 2
                        text_y = y + (size_px - text_height) / 2

                        # Draw the appropriate shape
                        if dot_shape == "Circle":
                            draw_numbers.ellipse([x, y, x + size_px, y + size_px], outline="gainsboro", width=1)
                        elif dot_shape == "Square":
                            draw_numbers.rectangle([x, y, x + size_px, y + size_px], outline="gainsboro", width=1)
                        elif dot_shape == "Diamond":
                            diamond_coords = [
                                (x + size_px / 2, y),  # Top
                                (x, y + size_px / 2),  # Left
                                (x + size_px / 2, y + size_px),  # Bottom
                                (x + size_px, y + size_px / 2),  # Right
                            ]
                            draw_numbers.polygon(diamond_coords, outline="gainsboro", width=1)

                        # Draw the text
                        draw_numbers.text((text_x, text_y), text, fill="black", font=font)

        # Add scale bar to Number Dot Map
        add_scale_bar(numbers_img, st.session_state.image_width, st.session_state.image_height, final_width_px, final_height_px)
        col2.image(numbers_img, caption="Generated Rhinestone Map (Number Dot Map with Scale)", use_container_width=True)

    # Rhinestone Legend
    st.subheader("Rhinestone Legend and Dot Counts")
    legend_cols = st.columns(2)
    for i, rhinestone in enumerate(rhinestones):
        if ignore_colors[i]:
            continue
        dot_count = np.sum(labels == i)
        shape_css = ""
        if dot_shape == "Circle":
            shape_css = "border-radius: 50%;"  # Makes the shape a circle
        elif dot_shape == "Square":
            shape_css = ""  # Default square
        elif dot_shape == "Diamond":
            shape_css = "transform: rotate(45deg);"  # Rotates the square to form a diamond

        with legend_cols[i % 2]:
            st.markdown(f"**Rhinestone {i + 1}**")
            st.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <div style="
                        width: 20px; 
                        height: 20px; 
                        background-color: {rhinestone['color']}; 
                        margin-right: 10px; 
                        {shape_css}">
                    </div>
                    {rhinestone['color']} - {dot_count} dots
                </div>
                """,
                unsafe_allow_html=True,
            )

# Function to create a Project Specification PDF with improved layout
def create_project_specification_pdf(uploaded_image_file, color_dot_img, numbers_img, rhinestones, ignore_colors, labels):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    # Set up page dimensions
    page_width, page_height = A4
    margin = 40
    available_width = page_width - 2 * margin

    # Title
    file_name=f"{project_title.replace(' ', '_')}.zip",

    # Helper function to draw images while maintaining aspect ratio
    def draw_image_with_aspect_ratio(pdf, image, x, y, max_width, max_height):
        img_width, img_height = image.size
        aspect_ratio = img_width / img_height

        if max_width / max_height > aspect_ratio:
            # Constrain by height
            draw_height = max_height
            draw_width = draw_height * aspect_ratio
        else:
            # Constrain by width
            draw_width = max_width
            draw_height = draw_width / aspect_ratio

        pdf.drawImage(ImageReader(image), x + (max_width - draw_width) / 2, y - draw_height, width=draw_width, height=draw_height)

    # Add Colour Dot Map (Main focus)
    color_dot_map_height = 250
    color_dot_map_y = page_height - margin - 60
    pdf.drawCentredString(page_width / 2, color_dot_map_y, "Colour Dot Map:")
    draw_image_with_aspect_ratio(
        pdf, 
        color_dot_img, 
        margin, 
        color_dot_map_y - 20, 
        available_width, 
        color_dot_map_height
    )

    # Add Uploaded Image and Number Dot Map side by side
    image_section_y = color_dot_map_y - color_dot_map_height - 50  # Add buffer between sections
    uploaded_image_width = (available_width / 2) - 10
    pdf.drawString(margin, image_section_y, "Uploaded Image:")
    draw_image_with_aspect_ratio(
        pdf, 
        Image.open(uploaded_image_file), 
        margin, 
        image_section_y - 20, 
        uploaded_image_width, 
        200
    )

    pdf.drawString(margin + uploaded_image_width + 20, image_section_y, "Number Dot Map:")
    draw_image_with_aspect_ratio(
        pdf, 
        numbers_img, 
        margin + uploaded_image_width + 20, 
        image_section_y - 20, 
        uploaded_image_width, 
        200
    )

    # Add Rhinestone Legend below
    legend_y = image_section_y - 250  # Add space below the images
    pdf.drawString(margin, legend_y, "Rhinestone Legend:")

    y_position = legend_y - 20
    box_size = 10  # Size of color box

    for i, rhinestone in enumerate(rhinestones):
        if ignore_colors[i]:
            continue
        dot_count = int(np.sum(labels == i))
        pdf.setFillColorRGB(*tuple(int(rhinestone["color"][i:i+2], 16) / 255.0 for i in (1, 3, 5)))
        pdf.rect(margin, y_position, box_size, box_size, fill=True, stroke=False)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(margin + box_size + 5, y_position + 2, f"Rhinestone {i + 1}: {rhinestone['color']} - {dot_count} dots")
        y_position -= 15

    pdf.save()
    buffer.seek(0)
    return buffer

# Example payment URL (Replace with your actual payment gateway checkout link)
PAYMENT_BASE_URL = "https://buy.stripe.com/bIY3cs4Kj52g8dqcMM"

# Check for payment confirmation via query parameters
query_params = st.experimental_get_query_params()
payment_confirmed = query_params.get("paid", ["false"])[0] == "true"

# Create a unique identifier for each file generation session
if "download_session_id" not in st.session_state:
    st.session_state.download_session_id = str(st.session_state.run_id)  # Unique session ID per run

# Payment URL with session validation
payment_url = f"{PAYMENT_BASE_URL}?{urlencode({'session_id': st.session_state.download_session_id})}"

if uploaded_file:
    if not payment_confirmed:
        # Force payment before download
        st.markdown(
            f"""
            <a href="{payment_url}" target="_blank">
                <button style="padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">
                    Proceed to Payment
                </button>
            </a>
            """,
            unsafe_allow_html=True,
        )
        st.warning("Complete payment to enable the download. This payment applies to this download session only.")
    else:
        # Check that the session ID matches
        session_id = query_params.get("session_id", [""])[0]
        if session_id != st.session_state.download_session_id:
            st.error("Payment validation failed. Please complete payment again.")
        else:
            # Payment validated for this session; allow download
            with io.BytesIO() as zip_buffer:
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    # Save Color Dot Map
                    color_dot_buffer = io.BytesIO()
                    color_dot_img.save(color_dot_buffer, format="PNG", dpi=(cm_to_pixels, cm_to_pixels))
                    zf.writestr(f"{project_title.replace(' ', '_')}_color_dot_map.png", color_dot_buffer.getvalue())

                    # Save Number Dot Map
                    number_dot_buffer = io.BytesIO()
                    numbers_img.save(number_dot_buffer, format="PNG", dpi=(cm_to_pixels, cm_to_pixels))
                    zf.writestr(f"{project_title.replace(' ', '_')}_number_dot_map.png", number_dot_buffer.getvalue())

                    # Save Project Specification PDF
                    pdf_buffer = create_project_specification_pdf(uploaded_file, color_dot_img, numbers_img, rhinestones, ignore_colors, labels)
                    zf.writestr(f"{project_title.replace(' ', '_')}_specification.pdf", pdf_buffer.getvalue())

                zip_buffer.seek(0)

                # Add download button
                st.download_button(
                    label=f"Download {project_title} ZIP",
                    data=zip_buffer,
                    file_name=f"{project_title.replace(' ', '_')}.zip",
                    mime="application/zip"
                )
