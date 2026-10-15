import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ==========================================
# 1. App Configuration & Eco-Theme Styling
# ==========================================

st.set_page_config(
    page_title="EcoSort | Smart Recycling AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Eco-System CSS Injection for Modern UI/UX
st.markdown("""
    <style>
    /* Main Background & Font Styling */
    .stApp {
        background: linear-gradient(135deg, #f4f9f4 0%, #e3eee3 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1b3b2b;
        color: #ffffff;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #e2ece9 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2e694e;
    }

    /* Primary Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1b5e20 0%, #0d3b12 100%);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.5);
        transform: translateY(-2px);
    }

    /* Headers */
    h1, h2, h3 {
        color: #1b3b2b !important;
        font-weight: 700;
    }

    /* Metrics */
    .stMetric {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    
    /* Progress Bar */
    [data-testid="stProgressBar"] > div > div > div {
        background: linear-gradient(90deg, #4caf50 0%, #81c784 100%);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "ecosort_artifacts"

MODEL_PATH = ARTIFACTS_DIR / "EcoSort_EfficientNetV2S.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
KNOWLEDGE_PATH = ARTIFACTS_DIR / "recycling_knowledge.json"

IMAGE_SIZE = (400, 400)


# ==========================================
# 3. Load Model and JSON Files
# ==========================================

@st.cache_resource
def load_model():
    """Load the trained EfficientNetV2S model."""
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    """Load class names in the original training order."""
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_recycling_knowledge():
    """Load recycling guidance for each waste category."""
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================
# 4. Image Preprocessing
# ==========================================

def preprocess_image(image):
    """
    Prepare an image for EfficientNetV2S inference.
    """
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# ==========================================
# 5. Make Predictions
# ==========================================

def predict_waste(model, image, class_names):
    """Predict the waste category and return top 3 results."""
    image_array = preprocess_image(image)

    predictions = model.predict(
        image_array,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    top_indices = np.argsort(predictions)[::-1][:3]

    top_predictions = [
        {
            "class_name": class_names[int(index)],
            "confidence": float(predictions[index]),
        }
        for index in top_indices
    ]

    return predicted_class, confidence, top_predictions


# ==========================================
# 6. Display Helpers
# ==========================================

def format_category(category):
    """Convert category names into readable titles."""
    return category.replace("_", " ").replace("-", " ").title()


def get_category_guide(knowledge, category):
    """Find guidance without case sensitivity."""
    category_key = str(category).strip().lower()
    guide = knowledge.get(category_key)

    if guide is not None:
        return guide

    for key, value in knowledge.items():
        if str(key).strip().lower() == category_key:
            return value

    return None


def display_recycling_guide(guide, predicted_class):
    """Display the title, recommended actions, and warning."""
    st.divider()
    st.header("🌿 Recycling & Disposal Guide")

    st.caption(
        f"Guidance for category: {format_category(predicted_class)}"
    )

    if guide is None:
        st.warning(
            f"No recycling guide found for category: "
            f"{predicted_class}"
        )
        st.caption(
            "Please check the category keys in recycling_knowledge.json."
        )
        return

    guide_title = guide.get(
        "title",
        format_category(predicted_class),
    )

    st.subheader(f"🌱 {guide_title}")

    actions = guide.get("actions", [])

    if actions:
        st.markdown("###  Recommended Actions")
        for action in actions:
            st.markdown(f"- {action}")
    else:
        st.info("No recommended actions available at the moment.")

    warning = guide.get("warning")

    if warning:
        st.warning(f"⚠️ Eco & Safety Warning: {warning}")


# ==========================================
# 7. Sidebar
# ==========================================

with st.sidebar:
    st.markdown("# 🍃 EcoSort AI")
    st.caption("AI-Powered Waste Classification & Sustainability")

    st.divider()

    st.markdown("### 🌍 About the Project")

    st.write(
        """
        EcoSort uses advanced deep learning to accurately identify waste categories 
        and provide practical guidance for sustainable recycling and disposal.
        """
    )

    st.markdown("**Model:** EfficientNetV2S")
    st.markdown("**Input Size:** 400 × 400 RGB")
    st.markdown("**Categories:** 10 Main Classes")

    st.divider()

    st.caption(
        "AI predictions are advisory. Always follow local waste-disposal "
        "and recycling regulations."
    )


# ==========================================
# 8. Main Page
# ==========================================

st.title("♻️ EcoSort - Towards a Greener Future")

st.markdown(
    """
    ### Give your waste a second life.

    Upload an image of a waste item to identify its category
    and discover how to dispose of or recycle it responsibly.
    """
)

st.divider()


# ==========================================
# 9. Check and Load Artifacts
# ==========================================

required_files = [
    MODEL_PATH,
    CLASS_NAMES_PATH,
    KNOWLEDGE_PATH,
]

missing_files = [
    path.name
    for path in required_files
    if not path.is_file()
]

if missing_files:
    st.error(
        "Some required project files are missing from "
        "the ecosort_artifacts folder:"
    )

    for filename in missing_files:
        st.markdown(f"- `{filename}`")

    st.info(
        "Ensure that the model file and both JSON files are "
        "inside the ecosort_artifacts folder."
    )

    st.stop()


try:
    model = load_model()
    class_names = load_class_names()
    recycling_knowledge = load_recycling_knowledge()

except Exception as error:
    st.error("The model or supporting files could not be loaded successfully.")
    st.exception(error)
    st.stop()


if not isinstance(class_names, list):
    st.error(
        "class_names.json must contain a JSON list "
        "of class names in the model's output order."
    )
    st.stop()

if not isinstance(recycling_knowledge, dict):
    st.error(
        "recycling_knowledge.json must contain a JSON object "
        "with category keys and their guidance details."
    )
    st.stop()


# ==========================================
# 10. Upload Image
# ==========================================

left_col, right_col = st.columns(
    [1, 1],
    gap="large",
)

with left_col:
    st.subheader(" Upload Waste Image")

    uploaded_file = st.file_uploader(
        "Choose a clear image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear image of a single waste item.",
    )

    image = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                width=400,
            )

        except Exception:
            st.error(
                "The uploaded file could not be read as a valid image."
            )
            st.stop()

    classify_button = st.button(
        "🔍 Classify Waste Item",
        type="primary",
        use_container_width=True,
        disabled=image is None,
    )


# ==========================================
# 11. Classification Results
# ==========================================

if classify_button and image is not None:
    st.session_state.pop("prediction_result", None)

    try:
        with st.spinner("Analyzing image and environmental footprint..."):
            predicted_class, confidence, top_predictions = predict_waste(
                model,
                image,
                class_names,
            )

        st.session_state["prediction_result"] = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions,
        }

    except Exception as error:
        st.error(
            "An error occurred while classifying the image."
        )
        st.exception(error)


# ==========================================
# 12. Display Classification Result
# ==========================================

result = st.session_state.get("prediction_result")

with right_col:
    st.subheader(" Analysis Results")

    if result is None:
        st.info(
            "Upload an image and click **Classify Waste Item** "
            "to view predictions."
        )

    else:
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        top_predictions = result["top_predictions"]

        st.success("Image classification completed successfully!")

        st.markdown("**Predicted Category:**")

        st.markdown(
            f"## 🌍 {format_category(predicted_class)}"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2%}",
        )

        st.caption(
            "Confidence reflects the model's softmax score, "
            "not an absolute guarantee."
        )

        st.markdown("####  Top 3 Predictions")

        for item in top_predictions:
            category_name = format_category(
                item["class_name"]
            )
            score = item["confidence"]

            st.write(f"**{category_name}** — {score:.2%}")

            st.progress(
                min(max(score, 0.0), 1.0)
            )


# ==========================================
# 13. Recycling and Disposal Guidance
# ==========================================

if result is not None:
    predicted_class = result["predicted_class"]

    guide = get_category_guide(
        recycling_knowledge,
        predicted_class,
    )

    display_recycling_guide(
        guide,
        predicted_class,
    )


# ==========================================
# 14. Footer
# ==========================================

st.divider()

st.caption(
    "EcoSort AI Platform | Deep Learning • Green Tech • Sustainable Future 🌿"
)