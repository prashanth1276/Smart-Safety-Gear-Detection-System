# Smart Safety Gear Detection System

## Project Description

The **Smart Safety Gear Detection System** is an automated solution aimed at ensuring workplace safety by detecting missing personal protective equipment (PPE), such as helmets, masks, and safety vests, in real-time. The system leverages **YOLOv8**, a state-of-the-art object detection model, to identify individuals who are not wearing essential safety gear. The project is equipped with a user-friendly interface built using **Streamlit**, allowing users to upload images or videos for analysis. The system sends automated email alerts for non-compliance and stores detection results in **MongoDB** for historical tracking.

---

## Features

- **Safety Gear Detection**: Identifies the presence or absence of safety gear, such as helmets, masks, and safety vests.
- **Streamlit Interface**: Users can upload images or videos for analysis, view annotated results, and adjust detection settings.
- **Email Notification**: Sends automated email alerts when safety gear is missing.
- **MongoDB Integration**: Stores historical detection results for easy access and management.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries/Frameworks**: Streamlit, OpenCV, YOLOv8, Matplotlib, Seaborn, Pandas, PyMongo
- **Database**: MongoDB
- **Tools/Platforms**: Google Colab, VS Code, Streamlit Web App

---

## Installation Instructions

Follow these steps to set up and run the project:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/prashanth1276/Smart-Safety-Gear-Detection-System.git
    ```
    ```
    cd Smart-Safety-Gear-Detection-System
    ```

2. **Set up a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - For Windows:
      ```bash
      venv\Scripts\activate
      ```
    - For macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

4. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Model Usage Instructions

1. **Trained Model**: The project includes a pre-trained YOLOv8 model, which can be found as `best.pt`. To use the trained model for detection, ensure you place this file in the project directory.

2. **Integration with Other Platforms**: If you need to integrate the trained model into another platform, you can use the model in `best.onnx` format. This format is compatible with various deployment environments and platforms supporting ONNX models.

3. **Alternative Model**: You can also find a lightweight YOLOv8 model version, `yolov8n.pt`, which is designed for faster processing at the cost of some accuracy. Use it if you need a quicker but less resource-intensive model.

---

## Dataset

The dataset used for training the model is available on Kaggle. You can download it from the following link:

[Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow/data)

---

## Usage Instructions

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Upload images or videos through the interface to check for missing safety gear. The app will display annotated images showing the detected items, and missing safety gear will be highlighted.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- YOLOv8 for object detection
- Streamlit for creating the interactive UI
- MongoDB for storing detection results

---

## For More Information

For further details and in-depth instructions, please refer to the full documentation in the repository.
