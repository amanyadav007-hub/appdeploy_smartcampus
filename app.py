
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Smart Campus Analytics",
    page_icon="🎓",
    layout="wide"
)

# Load files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "Deployment_Student_Dataset.csv"))
model = joblib.load(os.path.join(BASE_DIR, "random_forest_model.pkl"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))

# Features
features = [
    "Gender", "Age", "College", "Department", "Semester",
    "Attendance_%", "CGPA", "Backlogs", "Internship", "Skills"
]

# Sidebar
st.sidebar.title("🎓 Smart Campus")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Placement Prediction", "About"]
)

# Dashboard
if page == "Dashboard":

    st.title("🎓 Smart Campus Analytics")
    st.subheader("Student Performance & Placement Insights")

    total = len(df)
    placed = (df["Placement_Status"] == "Placed").sum()
    not_placed = (df["Placement_Status"] == "Not Placed").sum()
    rate = placed / total * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total)
    col2.metric("Placed Students", placed)
    col3.metric("Not Placed", not_placed)
    col4.metric("Placement Rate", f"{rate:.2f}%")

    st.divider()

    col5, col6, col7 = st.columns(3)

    col5.metric("Average CGPA", f"{df['CGPA'].mean():.2f}")
    col6.metric("Average Attendance", f"{df['Attendance_%'].mean():.2f}%")
    col7.metric("Average Backlogs", f"{df['Backlogs'].mean():.2f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Overall Placement")
        placement_counts = df["Placement_Status"].value_counts()
        fig = px.pie(
            values=placement_counts.values,
            names=placement_counts.index,
            title="Placed vs Not Placed",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Department-wise Placement")
        dept_data = df.groupby(
            ["Department", "Placement_Status"]
        ).size().reset_index(name="Students")

        fig = px.bar(
            dept_data,
            x="Department",
            y="Students",
            color="Placement_Status",
            barmode="group",
            title="Department Placement"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("CGPA vs Placement")
    fig = px.box(
        df,
        x="Placement_Status",
        y="CGPA",
        color="Placement_Status"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Attendance vs Placement")
    fig = px.box(
        df,
        x="Placement_Status",
        y="Attendance_%",
        color="Placement_Status"
    )
    st.plotly_chart(fig, use_container_width=True)


# Placement Prediction
elif page == "Placement Prediction":

    st.title("🔮 Student Placement Prediction")
    st.write("Enter student details to predict placement status.")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                sorted(df["Gender"].dropna().unique())
            )

            age = st.number_input(
                "Age", min_value=18, max_value=30, value=21
            )

            college = st.selectbox(
                "College",
                sorted(df["College"].dropna().unique())
            )

            department = st.selectbox(
                "Department",
                sorted(df["Department"].dropna().unique())
            )

            semester = st.number_input(
                "Semester", min_value=1, max_value=12, value=7
            )

        with col2:
            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0, max_value=100.0, value=85.0
            )

            cgpa = st.number_input(
                "CGPA",
                min_value=0.0, max_value=10.0, value=8.0
            )

            backlogs = st.number_input(
                "Backlogs", min_value=0, max_value=20, value=0
            )

            internship = st.selectbox(
                "Internship",
                sorted(df["Internship"].dropna().unique())
            )

            skills = st.selectbox(
                "Skills",
                sorted(df["Skills"].dropna().unique())
            )

        submit = st.form_submit_button("Predict Placement")

    if submit:

        input_data = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "College": college,
            "Department": department,
            "Semester": semester,
            "Attendance_%": attendance,
            "CGPA": cgpa,
            "Backlogs": backlogs,
            "Internship": internship,
            "Skills": skills
        }], columns=features)

        transformed_data = preprocessor.transform(input_data)
        prediction = model.predict(transformed_data)[0]

        st.divider()

        if prediction == "Placed":
            st.success("🎉 Prediction: Student is likely to be PLACED!")
        else:
            st.warning("Prediction: Student is likely to be NOT PLACED.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed_data)[0]
            classes = list(model.classes_)
            placed_probability = probabilities[classes.index("Placed")]

            st.metric(
                "Estimated Placement Probability",
                f"{placed_probability * 100:.2f}%"
            )

            st.caption(
                "This is a model estimate, not a guarantee of actual placement."
            )


# About
elif page == "About":

    st.title("About Smart Campus Analytics")

    st.write("""
    Smart Campus Analytics is a student placement prediction project.

    **Main Objectives:**
    - Analyze student performance.
    - Understand placement trends.
    - Compare department-wise placement.
    - Predict student placement using Machine Learning.

    **Technologies Used:**
    - Python
    - Pandas
    - Scikit-learn
    - Random Forest
    - Plotly
    - Streamlit
    """)
