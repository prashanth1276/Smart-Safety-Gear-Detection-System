# Import necessary libraries for Streamlit and email handling
import streamlit as st
import pandas as pd
import base64
import cv2
import numpy as np
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tempfile
from mongoDb import save_results, get_history, clear_all_history  # MongoDB functions

# Load your trained YOLO model
from ultralytics import YOLO

# Load the model
model = YOLO(r'C:\Users\Prashanth S\Downloads\Dataset-And-Output\results_yolov8n_10e\kaggle\working\runs\detect\train\weights\best.pt')

# Function to send an Email
def send_email_alert(subject, missing_items_str, user_email, app_password):
    sender_email = user_email
    receiver_email = "prashanths272005@gmail.com"  # Change this to the supervisor's email

    # Create the email content with the professional message
    message = f"""
    Dear Supervisor,

    I hope this message finds you well.

    It has come to our attention that certain safety gear items are missing in the workplace. Please find the details of the missing items below:

    Missing Items:
    {missing_items_str}

    We kindly request that all employees ensure they are wearing the required safety gear at all times. This is crucial to maintaining a safe and compliant work environment for everyone.

    Thank you for your prompt attention to this matter.
    """

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(msg)
            st.info("Email sent to the supervisor.")
    except smtplib.SMTPAuthenticationError:
        st.error("Error: Wrong email or password. Please check your credentials.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Streamlit UI

# Applying Background Image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://static.vecteezy.com/system/resources/previews/007/278/150/non_2x/dark-background-abstract-with-light-effect-vector.jpg");
background-size: cover;
}

[data-testid="stHeader"]{
background-image: url("https://static.vecteezy.com/system/resources/previews/007/278/150/non_2x/dark-background-abstract-with-light-effect-vector.jpg");
background-size: cover;
}

[data-testid="stSidebarContent"]{
background-image: url("https://static.vecteezy.com/system/resources/previews/007/278/150/non_2x/dark-background-abstract-with-light-effect-vector.jpg");
background-size: cover;
}
</style>
"""

st.markdown(page_bg_img,unsafe_allow_html=True)
st.title("🛠️Safety Gear Detection System")
st.markdown(
    """
    <div style="text-align: center; font-size: 18px;">
        Identify individuals missing safety gear in uploaded images or videos.
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.header("Upload")
upload_choice = st.sidebar.radio("Choose input type:", ("Image", "Video"))

# Confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Define colors for each class
colors = {
    'Hardhat': (0, 255, 0),       # Green
    'Mask': (255, 0, 0),          # Blue
    'NO-Hardhat': (0, 0, 255),    # Red
    'NO-Mask': (255, 165, 0),     # Orange
    'NO-Safety Vest': (128, 0, 128),  # Purple
    'Person': (0, 255, 255),      # Cyan
    'Safety Cone': (255, 0, 255),  # Magenta
    'Safety Vest': (0, 128, 0),    # Dark Green
    'machinery': (128, 128, 0),   # Olive
    'vehicle': (128, 0, 0)        # Maroon
}

# Ask user if they want to send an email alert
send_email_alert_flag = st.sidebar.radio(
    "Would you like to send an email alert about missing safety gear?",
    ("No", "Yes")
)

user_email = None
app_password = None

# If they want to receive an email, ask for email and app password
if send_email_alert_flag == "Yes":
    user_email = st.text_input("Enter your email address:")
    app_password = st.text_input("Enter your app password:", type="password")

# Detection process based on image or video input
if upload_choice == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.header("Input Image")
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to OpenCV format
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Run detection
        results = model.predict(source=img, conf=confidence_threshold, save=True)

        # Track detected items and missing items
        detected_items = set()
        missing_items = set()

        # Display results
        for result in results:
            boxes = result.boxes  # Access the boxes attribute
            if len(boxes.xyxy) > 0:  # Check if any boxes were detected
                labels = boxes.cls.tolist()  # Get detected class indices
                detected_labels = [result.names[int(label)] for label in labels]  # Map indices to names

                # Annotate the image with bounding boxes and labels
                for box, label in zip(boxes.xyxy, detected_labels):
                    x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
                    color = colors.get(label, (255, 255, 255))  # Default to white if label not found
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Draw rectangle with the specified color
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Track detected and missing items
                    if label.startswith("NO-"):
                        missing_items.add(label[3:])
                    else:
                        detected_items.add(label)

        # Convert annotated image back to RGB format for Streamlit
        annotated_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        st.header("Output")
        st.image(annotated_image, caption="Detection Result", use_column_width=True)

        # Check if there are missing items
        if missing_items:
            missing_items_str = ', '.join(missing_items)
            st.error(f"Missing safety gear detected: {missing_items_str}")
            # Save to MongoDB
            save_results(uploaded_image.name, list(detected_items), list(missing_items))
            if user_email and app_password:
                send_email_alert("Safety Gear Alert", f"{missing_items_str}", user_email, app_password)
        else:
            st.success("All essential safety gear is detected.")
            # Save to MongoDB
            save_results(uploaded_image.name, list(detected_items), [])

elif upload_choice == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        st.header("Input Video")
        st.video(uploaded_video)

        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
            temp_input_file.write(uploaded_video.read())
            temp_input_path = temp_input_file.name

        # Temporary save location for the processed video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_file:
            temp_output_path = temp_output_file.name

        # Open the saved video file
        cap = cv2.VideoCapture(temp_input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define VideoWriter with matching width, height, and fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        missing_items = set()

        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection on the frame
            results = model.predict(source=frame, conf=confidence_threshold)

            for result in results:
                boxes = result.boxes  # Access the boxes attribute
                if len(boxes.xyxy) > 0:  # Check if any boxes were detected
                    labels = boxes.cls.tolist()  # Get detected class indices
                    detected_labels = [result.names[int(label)] for label in labels]  # Map indices to names

                    # Annotate the frame with bounding boxes and labels
                    for box, label in zip(boxes.xyxy, detected_labels):
                        x1, y1, x2, y2 = map(int, box)
                        color = colors.get(label, (255, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Track missing items
                        if label.startswith("NO-"):
                            missing_items.add(label[3:])

            out.write(frame)

        # Release video resources
        cap.release()
        out.release()

        # Provide the processed video as a download link
        st.header("Output")
        with open(temp_output_path, 'rb') as file:
            st.download_button("Download Processed Video", file, file_name="processed_video.mp4")

        # Check if there are missing items
        if missing_items:
            missing_items_str = ', '.join(missing_items)
            st.error(f"Missing safety gear detected in video: {missing_items_str}")
            # Save to MongoDB
            save_results(uploaded_video.name, list(detected_labels), list(missing_items))
            if user_email and app_password:
                send_email_alert("Safety Gear Alert", f"{missing_items_str}", user_email, app_password)
        else:
            st.success("All essential safety gear detected in the video.")

# Sidebar: Header and buttons for showing history and deleting
st.sidebar.header("Detection History")

# Show detection history
if st.sidebar.button("Show Detection History"):
    history = get_history()  # Fetch history from MongoDB
    if history:
    # Display history in a more organized format
        for idx, record in enumerate(history):
            file_name = record["file_name"]
            detected_items = record["detected_items"]
            missing_items = record["missing_items"]
            detection_time = record["detection_time"]

            st.sidebar.write(f"#### Detection {idx + 1}")
            st.sidebar.markdown(f"**File Name**: {file_name}")
            
            # List detected items
            st.sidebar.markdown("**Detected Items:**")
            for item in detected_items:
                st.sidebar.write(f"  - {item.strip()}")  # Display each detected item on a new line

            # List missing items
            st.sidebar.markdown("**Missing Items:**")
            for item in missing_items:
                st.sidebar.write(f"  - {item.strip()}")  # Display each missing item on a new line

            # Display detection time
            st.sidebar.markdown(f"**Detection Time**: {detection_time}")
            st.sidebar.markdown("---")  # Add a separator between records
    else:
        st.sidebar.write("No history available.")


# Button to delete all history
if st.sidebar.button("Delete All History"):
    clear_all_history()  # Clear all records from MongoDB
    st.sidebar.write("All history has been deleted.")  # Confirmation message

# Sidebar section for logs
st.sidebar.header("Detection Logs")  # Add a heading for the logs section

# Check if the button is clicked
if st.sidebar.button("Download Detection Logs"):
    logs = get_history()  # Fetch detection logs from the database
    if logs:
        logs_df = pd.DataFrame(logs)  # Convert logs to a DataFrame
        csv = logs_df.to_csv(index=False)  # Convert DataFrame to CSV

        # Encode the CSV data to Base64
        b64_csv = base64.b64encode(csv.encode()).decode()

        # Generate a download link
        st.sidebar.markdown(
            f'<a href="data:file/csv;base64,{b64_csv}" download="detection_logs.csv">Click here to download logs</a>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.error("No logs available.")  # Error message if no logs