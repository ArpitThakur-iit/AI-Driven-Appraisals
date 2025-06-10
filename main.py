# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Streamlit config
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("employee_data_with_ai.csv").drop(columns=['Unnamed: 0'])
    return df

df = load_data()
option = st.sidebar.selectbox("Choose a view", ["Employee View", "Admin View", "Fairness Check"])

# ------------------------- Employee View -------------------------
if option == "Employee View":
    st.title("Employee Performance Dashboard")
    employee_name = st.selectbox("Select Employee", df["Name"].unique())
    employee_data = df[df["Name"] == employee_name].iloc[0]
    metrics = ['MeetingsAttended', 'EngagementScore', 'TasksCompleted', 'TaskScore', 'ProjectScore']
    team_min = df.min(numeric_only=True)
    team_max = df.max(numeric_only=True)

    def normalize_to_percent(score, min_val, max_val):
        return ((score - min_val) / (max_val - min_val)) * 100

    employee_normalized = [normalize_to_percent(employee_data[m], team_min[m], team_max[m]) for m in metrics]
    team_avg_normalized = [normalize_to_percent(df[m].mean(), team_min[m], team_max[m]) for m in metrics]

    # Summary Card
    st.subheader("Employee Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Score", f"{employee_data['Overall_Score']:.1f}")
        st.metric("Rank", employee_data['Performance_Label'])
        st.metric("Top Strength", metrics[np.argmax(employee_normalized)])
        st.metric("To Improve", metrics[np.argmin(employee_normalized)])
    with col2:
        fig, ax = plt.subplots()
        sns.barplot(x=['You', 'Team Avg'], y=[employee_data['Overall_Score'], df['Overall_Score'].mean()],
                    palette=['#4CAF50', '#2196F3'], ax=ax)
        ax.set_title("Overall Score Comparison")
        st.pyplot(fig)

    # Bar Chart
    fig, ax = plt.subplots()
    x = range(len(metrics))
    ax.bar(x, employee_normalized, width=0.4, label='You', color='#4CAF50')
    ax.bar([i + 0.4 for i in x], team_avg_normalized, width=0.4, label='Team Avg', color='#2196F3')
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(metrics, rotation=45)
    ax.set_ylabel("Normalized Score (%)")
    ax.set_title(f"{employee_name}'s Performance")
    ax.legend()
    st.pyplot(fig)

    # Radar Chart
    def plot_radar_chart():
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        emp_scores = np.append(employee_normalized, employee_normalized[0])
        team_scores = np.append(team_avg_normalized, team_avg_normalized[0])
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(angles, emp_scores, label='You', color='#4CAF50')
        ax.fill(angles, emp_scores, alpha=0.3, color='#4CAF50')
        ax.plot(angles, team_scores, label='Team Avg', color='#2196F3')
        ax.fill(angles, team_scores, alpha=0.3, color='#2196F3')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title("Radar Chart - Normalized Metrics")
        ax.legend()
        return fig

    st.pyplot(plot_radar_chart())

    # ------------------------- Review Section -------------------------
    st.subheader("In case of any error in the analytics fill out the form here!")
    with st.form("error_form"):
        review_text = st.text_area("Elaborate the error here", height=150)
        submitted = st.form_submit_button("Submit")
        if submitted and review_text.strip():
            try:
                # Email config (update these!)
                sender = "your_email@example.com"
                receiver = "admin_email@example.com"
                password = "your_password_or_app_password"

                msg = MIMEText(f"Review from {employee_name}:\n\n{review_text}")
                msg['Subject'] = f"Employee Dashboard Error from {employee_name}"
                msg['From'] = sender
                msg['To'] = receiver

                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login(sender, password)
                    server.send_message(msg)

                st.success("✅ Your error has been reported to the admin!")
            except Exception as e:
                st.error(f"❌ Failed to send: {e}")

    st.subheader("📬 You are welcome to share your reviews here!")
    with st.form("review_form"):
        review_text = st.text_area("Your Feedback on the Dashboard Insights", height=150)
        submitted = st.form_submit_button("Submit Review")
        if submitted and review_text.strip():
            try:
                # Email config (update these!)
                sender = "your_email@example.com"
                receiver = "admin_email@example.com"
                password = "your_password_or_app_password"

                msg = MIMEText(f"Review from {employee_name}:\n\n{review_text}")
                msg['Subject'] = f"Employee Dashboard Review from {employee_name}"
                msg['From'] = sender
                msg['To'] = receiver

                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login(sender, password)
                    server.send_message(msg)

                st.success("✅ Your review has been sent to the admin!")
            except Exception as e:
                st.error(f"❌ Failed to send review: {e}")

# ------------------------- Admin View -------------------------
elif option == "Admin View":
    st.title("📊 Admin Dashboard - Team Performance Overview")
    st.subheader("1. Performance Score Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(data=df, x='Overall_Score', hue='Performance_Label', kde=True, palette='viridis', ax=ax1)
    ax1.set_title("Distribution of Performance Scores")
    st.pyplot(fig1)

    st.subheader("2. Cluster Comparison (Boxplot)")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='Performance_Label', y='Overall_Score', palette='Set2', order=['High', 'Mid', 'Low'], ax=ax2)
    ax2.set_title("Performance Clusters Comparison")
    st.pyplot(fig2)

    st.subheader("3. Metrics Radar Charts - Top & Bottom 5")
    def plot_radar_chart(subset_df, title):
        categories = ['MeetingsAttended', 'EngagementScore', 'TasksCompleted', 'TaskScore', 'ProjectScore']
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        for _, row in subset_df.iterrows():
            values = row[categories].tolist() + [row[categories[0]]]
            angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))] + [0]
            ax.plot(angles, values, label=row['Name'])
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(title, size=12)
        ax.set_yticklabels([])
        return fig

    st.markdown("**Top 5 Performers**")
    st.pyplot(plot_radar_chart(df.nlargest(5, 'Overall_Score'), "Top 5 Performers - Radar Chart"))

    st.markdown("**Bottom 5 Performers**")
    st.pyplot(plot_radar_chart(df.nsmallest(5, 'Overall_Score'), "Bottom 5 Performers - Radar Chart"))

    st.subheader("4. Feature Correlation Heatmap")
    corr = df[['MeetingsAttended', 'EngagementScore', 'TasksCompleted', 'TaskScore', 'ProjectScore']].corr()
    fig3, ax3 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax3)
    st.pyplot(fig3)

    st.subheader("5. AI Predicted vs Actual Scores")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df, x='Overall_Score', y='Predicted_Score', hue='Performance_Label', palette='Set2', ax=ax4)
    ax4.plot([df['Overall_Score'].min(), df['Overall_Score'].max()],
             [df['Overall_Score'].min(), df['Overall_Score'].max()], 'r--')
    ax4.set_title("AI Predicted vs Actual Scores")
    st.pyplot(fig4)

# ------------------------- Fairness Check -------------------------
if option == "Fairness Check":
    st.subheader("Gender-Based Score Comparison")
    score_columns = ['MeetingsAttended', 'EngagementScore', 'TasksCompleted', 'TaskScore', 'ProjectScore']
    gender_grouped = df.groupby("Gender")[score_columns].mean().T

    fig, ax = plt.subplots()
    gender_grouped.plot(kind='bar', ax=ax, rot=45)
    ax.set_title("Average Scores by Gender")
    st.pyplot(fig)

    gender_diff = gender_grouped["Male"] - gender_grouped["Female"]
    threshold = st.slider("Highlight difference above threshold", 0.0, 10.0, 2.0)
    significant_diff = gender_diff[abs(gender_diff) > threshold]
    significant_features = significant_diff.index.tolist()

    if significant_features:
        st.warning("⚠️ Significant differences found in:")
        st.write(significant_features)
    else:
        st.success("✅ No significant differences found.")
