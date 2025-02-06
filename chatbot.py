import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.ensemble import RandomForestRegressor
import sys
import io
import seaborn as sns

sys.path.append('c:/users/pratik bhosale\\appdata\\local\\programs\\python\\python313\\lib\\site-packages (2024.10.0)')

st.set_page_config(layout="wide")

# Set up Streamlit page
st.title("Zomato Data Analysis and Restaurant Recommendation App")

# File path for the default dataset
DEFAULT_FILE_PATH = "zomato.csv"

# Load default dataset
def load_default_data():
    try:
        data = pd.read_csv(DEFAULT_FILE_PATH, encoding='ISO-8859-1')
        return data
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
        return None

# Load the data
data = load_default_data()

# Page Navigation (Centered)
page = st.radio("Select a page:", ("Home", "Data Analysis", "Chatbot", "Predictive Modeling"), index=0, horizontal=True)

# Home Page
if page == "Home":
    st.subheader("Welcome to the Zomato Data Analysis and Restaurant Recommendation App")
    st.write("""
    **How to use:**
    - Use the top navigation to select between different sections of the app.
    - In the 'Data Analysis' section, explore the data.
    - In the 'Chatbot', interact with a chatbot to get restaurant recommendations.
    - Use the 'Predictive Modeling' section to predict aggregate ratings.
    """)

# Data Analysis Section
elif page == "Data Analysis":
    if data is not None:
        show_dataset = st.checkbox("Show Dataset")
        show_info = st.checkbox("Show Dataset Info")
        check_missing_values = st.checkbox("Check Missing Values")
        show_rating_distribution = st.checkbox("Distribution of Aggregate Ratings")
        show_top_cuisines = st.checkbox("Top 10 Cuisines")
        interactive_filtering = st.checkbox("Interactive Data Filtering")
        show_city_distribution = st.checkbox("Show City-Wise Distribution")

        # Display dataset
        if show_dataset:
            st.write(data.head())

        # General Information
        if show_info:
            buffer = io.StringIO()  # Create an in-memory text stream
            data.info(buf=buffer)  # Capture dataset info in buffer
            content = buffer.getvalue()  # Get buffer content
            st.text(content)  # Display the content in Streamlit

        if check_missing_values:
            st.write(data.isnull().sum())

        # Drop rows with missing Cuisines
        data = data.dropna(subset=['Cuisines'])

        # Data Insights
        st.subheader("Data Insights")
        if show_rating_distribution:
            fig, ax = plt.subplots()
            sns.histplot(data['Aggregate rating'], kde=True, bins=20, ax=ax)
            ax.set_title('Distribution of Aggregate Ratings')
            st.pyplot(fig)

        if show_top_cuisines:
            cuisine_counts = data['Cuisines'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index, ax=ax)
            ax.set_title('Top 10 Cuisines')
            st.pyplot(fig)

        # Interactive Data Filtering
        if interactive_filtering:
            st.subheader("Interactive Data Filtering")
            cities = st.multiselect("Select Cities", options=data['City'].unique(), default=None)
            price_range = st.multiselect("Select Price Range", options=data['Price range'].unique(), default=None)

            filtered_data = data
            if cities:
                filtered_data = filtered_data[filtered_data['City'].isin(cities)]
            if price_range:
                filtered_data = filtered_data[filtered_data['Price range'].isin(price_range)]

            st.write(f"Filtered Data: {filtered_data.shape[0]} rows")
            st.dataframe(filtered_data)

            # Option to download filtered data
            st.subheader("Download Processed Data")
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(filtered_data)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name='filtered_zomato_data.csv',
                mime='text/csv',
            )

        # City-wise distribution of restaurants
        if show_city_distribution:
            st.subheader("City-Wise Distribution of Restaurants")
            city_counts = data['City'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=city_counts.values, y=city_counts.index, ax=ax)
            ax.set_title('Top Cities with Most Restaurants')
            st.pyplot(fig)
    else:
        st.warning("Data not found.")

# Chatbot Section (Replacing Recommendation System)
elif page == "Chatbot":
    if data is not None:
        st.subheader("Restaurant Chatbot 🤖 - Welcome! How can I help you today? 🌟")

        # List of top 10 restaurants for chatbot selection
        top_restaurants = data['Restaurant Name'].value_counts().head(10).index.tolist()

        # Chatbot interaction with emojis and enhanced user experience
        st.write("Please select a restaurant from the list below 🍕🍽️:")
        restaurant_choice = st.selectbox("Select a restaurant 🏙️:", top_restaurants)

        if restaurant_choice:
            st.write(f"🎉 Great choice! You've selected **{restaurant_choice}**. Here's more information about it 🥳:")

            # Display images related to cuisines (Pizza example)
            if 'Pizza' in data[data['Restaurant Name'] == restaurant_choice]['Cuisines'].values[0]:
                st.image("https://upload.wikimedia.org/wikipedia/commons/5/5f/Pizza_logo.svg", width=200, caption="🍕 Pizza")
                st.image("https://upload.wikimedia.org/wikipedia/commons/9/99/Chhota_Bheem_logo.png", width=200, caption="Chhota Bheem 🦸‍♂️")
                st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/Doremon_logo.svg", width=200, caption="Doremon 🐱")

            # Restaurant Data
            restaurant_data = data[data['Restaurant Name'] == restaurant_choice].iloc[0]
            
            # Display additional details with emojis and animations
            st.write(f"**🍴 Cuisines:** {restaurant_data['Cuisines']}")
            st.write(f"**⭐ Aggregate Rating:** {restaurant_data['Aggregate rating']}")
            st.write(f"**📍 Location:** {restaurant_data['City']}")
            st.write(f"**💸 Average Cost for Two:** ₹{restaurant_data['Average Cost for two']}")
            st.write(f"**💬 Number of Votes:** {restaurant_data['Votes']} votes")
            st.write(f"**🏷️ Price Range:** {restaurant_data['Price range']}")

            # Add a friendly quote with fun emojis
            st.write("Let me know if you'd like recommendations or more information 🌈💬!")

         
        st.write("How else can I assist you? 🤔💡")
    else:
        st.warning("Data not found.")

# Predictive Modeling Section
elif page == "Predictive Modeling":
    if data is not None:
        st.subheader("Predictive Modeling: Aggregate Rating Prediction")
        features = data[['Average Cost for two', 'Votes', 'Price range']]
        target = data['Aggregate rating']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")

        importances = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=features.columns, ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)

        # Additional evaluation metrics for predictive modeling
        st.subheader("Model Evaluation Metrics")
        r2_score = model.score(X_test, y_test)
        st.write(f"R-squared: {r2_score:.2f}")
    else:
        st.warning("Data not found.")
