import os
import base64

import pandas as pd
import requests
import streamlit as st

from medical_imaging_ai.constants import CLASSES


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE_URL}/analyze"

st.set_page_config(
    page_title="Chest X-ray AI Assistant",
    page_icon="🩻",
    layout="wide"
)


# --------------------------------------------------
# Page header
# --------------------------------------------------

st.title("🩻 AI-Assisted Chest X-ray Analysis")

st.caption(
    "Educational multi-label chest X-ray classification system"
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Analysis settings")

disease = st.sidebar.selectbox(
    "Disease for Grad-CAM explanation",
    CLASSES
)


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-ray")

        st.image(
            image_bytes,
            use_container_width=True
        )

    with col2:
        st.subheader("Analysis")

        if st.button("Analyze X-ray", type="primary"):

            files = {
                "file": (
                    uploaded_file.name,
                    image_bytes,
                    uploaded_file.type
                )
            }

            data = {
                "disease": disease
            }

            try:
                with st.spinner("Analyzing image..."):

                    response = requests.post(
                        API_URL,
                        files=files,
                        data=data,
                        timeout=(10, 180)
                    )

                    response.raise_for_status()

                    result = response.json()

                    st.session_state["analysis_result"] = result

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to FastAPI. "
                    "Make sure the backend is running."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The analysis request timed out. "
                    "Please try again."
                )

            except requests.exceptions.HTTPError:
                st.error(
                    f"API error ({response.status_code}): "
                    f"{response.text}"
                )

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

            except ValueError:
                st.error("FastAPI returned an invalid JSON response.")


st.divider()
st.header("DICOM Analysis")

uploaded_dicom = st.file_uploader(
    "Upload DICOM file",
    type=["dcm"],
    key="dicom_upload"
)

if st.button(
    "Upload and Analyze",
    disabled=uploaded_dicom is None
):

    try:
        with st.spinner("Uploading DICOM to Orthanc..."):

            upload_response = requests.post(
                f"{API_BASE_URL}/dicom/upload",
                files={
                    "file": (
                        uploaded_dicom.name,
                        uploaded_dicom.getvalue(),
                        "application/dicom"
                    )
                },
                timeout=60
            )

            upload_response.raise_for_status()
            upload_result = upload_response.json()

        instance_id = upload_result["instance_id"]

        with st.spinner("Analyzing DICOM image..."):

            analysis_response = requests.post(
                f"{API_BASE_URL}/dicom/analyze/{instance_id}",
                timeout=120
            )

            analysis_response.raise_for_status()
            result = analysis_response.json()

        st.success("DICOM analysis completed!")

        st.subheader("Study Information")
        st.write("Patient ID:", result["patient_id"])
        st.write("Study UID:", result["study_uid"])

        st.subheader("Predictions")

        prediction_df = pd.DataFrame(result["predictions"])
        prediction_df.columns = ["Finding", "Score"]
        prediction_df["Score"] = (
            prediction_df["Score"] * 100
        ).round(2)

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(result["disclaimer"])

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")


# --------------------------------------------------
# Display analysis results
# --------------------------------------------------

if "analysis_result" in st.session_state:

    result = st.session_state["analysis_result"]

    st.divider()

    st.header("Analysis Results")

    
    # ----------------------------------------------
    # Predictions
    # ----------------------------------------------

    predictions = result["predictions"]

    prediction_df = pd.DataFrame(predictions)

    # Validate the API response
    required_columns = {"disease", "probability"}

    if not required_columns.issubset(prediction_df.columns):
        st.error("Unexpected prediction response format.")
        st.stop()

    # Rename columns for display
    prediction_df = prediction_df.rename(
        columns={
            "disease": "Finding",
            "probability": "Score"
        }
    )

    # Sort predictions from highest to lowest
    prediction_df = prediction_df.sort_values(
        by="Score",
        ascending=False
    ).reset_index(drop=True)

    # ----------------------------------------------
    # Prediction table and chart
    # ----------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted findings")

        st.dataframe(
            prediction_df,
            hide_index=True,
            use_container_width=True
        )

    with col2:
        st.subheader("Prediction scores")

        st.bar_chart(
            prediction_df.set_index("Finding")
        )

    
    # ----------------------------------------------
    # Grad-CAM explanation
    # ----------------------------------------------

    st.divider()

    st.subheader("Visual explanation")

    explained_disease = result["explained_disease"]
    explained_score = result["explained_probability"]

    st.markdown(
        f"**Finding being explained:** {explained_disease}"
    )

    st.metric(
        label="Model score",
        value=f"{explained_score:.4f}"
    )

    # Decode Grad-CAM image returned by FastAPI
    overlay_bytes = base64.b64decode(
        result["gradcam_overlay"]
    )

    # Display original and Grad-CAM side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original X-ray**")

        st.image(
            image_bytes,
            use_container_width=True
        )

    with col2:
        st.markdown("**Grad-CAM overlay**")

        st.image(
            overlay_bytes,
            use_container_width=True
        )

    st.caption(
        "Grad-CAM highlights image regions that contributed "
        "to the model's score for the selected finding. "
        "It is an explanatory visualization, not proof of disease."
    )

    # ----------------------------------------------
    # Disclaimer
    # ----------------------------------------------

    st.divider()

    st.warning(
        "For educational and research purposes only. "
        "Not for clinical diagnosis or treatment decisions."
    )


# --------------------------------------------------
# Study History
# --------------------------------------------------

st.divider()

st.header("Study History")

st.write(
    "Browse previous X-ray analyses stored in the database."
)

if st.button("Refresh Study History"):

    try:
        response = requests.get(
            f"{API_BASE_URL}/studies",
            timeout=10
        )

        response.raise_for_status()

        history_data = response.json()

        st.session_state["study_history"] = (
            history_data["studies"]
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Unable to retrieve study history: {e}")


# Load history when the page is first opened
if "study_history" not in st.session_state:

    try:
        response = requests.get(
            f"{API_BASE_URL}/studies",
            timeout=10
        )

        response.raise_for_status()

        history_data = response.json()

        st.session_state["study_history"] = (
            history_data["studies"]
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to FastAPI: {e}")


# Display study history
studies = st.session_state.get("study_history", [])

if studies:

    history_df = pd.DataFrame(studies)

    st.subheader("Previous analyses")

    st.dataframe(
        history_df,
        hide_index=True,
        use_container_width=True
    )

    # Select a study
    study_ids = [study["id"] for study in studies]

    selected_study_id = st.selectbox(
        "Select a study to inspect",
        study_ids,
        format_func=lambda study_id: (
            f"Study #{study_id} — "
            + next(
                study["image_filename"]
                for study in studies
                if study["id"] == study_id
            )
        )
    )

    if st.button("View Study Details"):

        try:
            response = requests.get(
                f"{API_BASE_URL}/studies/{selected_study_id}",
                timeout=10
            )

            response.raise_for_status()

            study = response.json()

            st.session_state["selected_study"] = study

        except requests.exceptions.RequestException as e:
            st.error(f"Unable to retrieve study: {e}")


    # Display selected study
    if "selected_study" in st.session_state:

        study = st.session_state["selected_study"]

        # Avoid displaying a previously selected study
        # when a different ID has just been selected.
        if study["id"] == selected_study_id:

            st.divider()

            st.subheader(
                f"Study #{study['id']} Details"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"**Filename:** {study['image_filename']}"
                )

                st.write(
                    f"**Analyzed at:** {study['created_at']}"
                )

            with col2:
                st.write(
                    f"**Explained finding:** "
                    f"{study['explained_disease']}"
                )

                st.write(
                    f"**Model score:** "
                    f"{study['explained_probability']:.4f}"
                )

            # Previous prediction results
            st.subheader("Prediction results")

            study_predictions = pd.DataFrame(
                study["predictions"]
            )

            study_predictions = study_predictions.rename(
                columns={
                    "disease": "Finding",
                    "probability": "Score"
                }
            )

            st.dataframe(
                study_predictions,
                hide_index=True,
                use_container_width=True
            )

            st.bar_chart(
                study_predictions.set_index("Finding")
            )

        # ----------------------------------------------
        # Historical images
        # ----------------------------------------------

        st.divider()

        st.subheader("Study images")

        try:
            original_response = requests.get(
                f"{API_BASE_URL}/studies/{study['id']}/image",
                timeout=15
            )

            gradcam_response = requests.get(
                f"{API_BASE_URL}/studies/{study['id']}/gradcam",
                timeout=15
            )

            original_response.raise_for_status()
            gradcam_response.raise_for_status()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original X-ray**")
                st.image(
                    original_response.content,
                    use_container_width=True
                )

            with col2:
                st.markdown("**Grad-CAM overlay**")
                st.image(
                    gradcam_response.content,
                    use_container_width=True
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Unable to retrieve study images: {e}")

else:
    st.info(
        "No previous studies found. "
        "Analyze an X-ray to create your first study."
    )



