import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import numpy as np
import io
import zipfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

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
num_colors = st.sidebar.slider("Number of Rhinestone Colors", min_value=2, max_value=20, value=6)
rhinestone_size = st.sidebar.number_input("Rhinestone Size (mm)", min_value=1.0, max_value=10.0, value=3.0)
st.sidebar.header("Final Image Dimensions (in cm)")
image_width = st.sidebar.number_input("Width (cm)", min_value=5.0, max_value=100.0, value=30.0)
aspect_ratio_locked = st.sidebar.checkbox("Lock Aspect Ratio", value=True)
image_height = st.sidebar.number_input("Height (cm)", min_value=5.0, max_value=100.0, value=30.0 if not aspect_ratio_locked else None)

# Add toggle for dot alignment style
dot_alignment = st.sidebar.radio("Dot Alignment Style", ["Brick Style (Offset)", "Straight Alignment"])

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
        if aspect_ratio_locked:
            aspect_ratio = original_height / original_width
            image_height = image_width * aspect_ratio

        # Resize image
        resolution = 50
        resized_width = resolution
        resized_height = int(resolution * (image_height / image_width))
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
        final_width_px = int(image_width * cm_to_pixels)
        final_height_px = int(image_height * cm_to_pixels)
        size_px = int(rhinestone_size * cm_to_pixels / 10)
        step_x = size_px + 2
        step_y = size_px + 2

        # Generate Color Dot Map
        st.subheader("Colour Dot Map")
        color_dot_img = Image.new("RGB", (final_width_px, final_height_px), "white")
        draw_color = ImageDraw.Draw(color_dot_img)
        for i, rhinestone in enumerate(rhinestones):
            if ignore_colors[i]:
                continue
            for row in range(0, final_height_px, step_y):
                for col in range(0, final_width_px, step_x):
                    col_offset = (
                        size_px // 2 if (row // step_y) % 2 == 1 and dot_alignment == "Brick Style (Offset)" else 0
                    )
                    x = col + col_offset
                    y = row
                    if x + size_px > final_width_px or y + size_px > final_height_px:
                        continue
                    label_y = min(y * labels.shape[0] // final_height_px, labels.shape[0] - 1)
                    label_x = min(x * labels.shape[1] // final_width_px, labels.shape[1] - 1)
                    if labels[label_y][label_x] == i:
                        draw_color.ellipse([x, y, x + size_px, y + size_px], fill=rhinestone["color"])
        st.image(color_dot_img, caption="Generated Rhinestone Map (Color Dot Map)", use_container_width=True)

        # Paint-by-Numbers Map with circles and numbers
        col2.subheader("Number Dot Map")
        numbers_img = Image.new("RGB", (final_width_px, final_height_px), "white")
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
                        size_px // 2 if (row // step_y) % 2 == 1 and dot_alignment == "Brick Style (Offset)" else 0
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
                        draw_numbers.ellipse([x, y, x + size_px, y + size_px], outline="gainsboro", width=1)
                        draw_numbers.text((text_x, text_y), text, fill="black", font=font)
        col2.image(numbers_img, caption="Generated Rhinestone Map (Paint-by-Numbers with Circles)", use_container_width=True)

    # Rhinestone Legend
    st.subheader("Rhinestone Legend and Dot Counts")
    legend_cols = st.columns(2)
    for i, rhinestone in enumerate(rhinestones):
        if ignore_colors[i]:
            continue
        dot_count = np.sum(labels == i)
        with legend_cols[i % 2]:
            st.markdown(f"**Rhinestone {i + 1}**")
            st.markdown(
                f"<div style='display: flex; align-items: center;'>"
                f"<div style='width: 20px; height: 20px; background-color: {rhinestone['color']}; margin-right: 10px;'></div>"
                f"{rhinestone['color']} - {dot_count} dots</div>",
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
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(page_width / 2, page_height - margin, "Diamond Dot Template Creator")

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

# Add a download button for ZIP
if uploaded_file:
    with io.BytesIO() as zip_buffer:
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            # Save Color Dot Map
            color_dot_buffer = io.BytesIO()
            color_dot_img.save(color_dot_buffer, format="PNG", dpi=(cm_to_pixels, cm_to_pixels))
            zf.writestr("color_dot_map.png", color_dot_buffer.getvalue())

            # Save Number Dot Map
            number_dot_buffer = io.BytesIO()
            numbers_img.save(number_dot_buffer, format="PNG", dpi=(cm_to_pixels, cm_to_pixels))
            zf.writestr("number_dot_map.png", number_dot_buffer.getvalue())

            # Save Project Specification PDF
            pdf_buffer = create_project_specification_pdf(uploaded_file, color_dot_img, numbers_img, rhinestones, ignore_colors, labels)
            zf.writestr("project_specification.pdf", pdf_buffer.getvalue())

        zip_buffer.seek(0)

        # Add download button
        st.download_button(
            label="Download Project ZIP",
            data=zip_buffer,
            file_name="diamond_dot_project.zip",
            mime="application/zip"
        )
