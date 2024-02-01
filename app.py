import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

anime_p = pickle.load(open('anime.pkl', 'rb'))
anime_sim_p = pickle.load(open('anime_sim.pkl', 'rb'))

anime_df = pd.DataFrame(anime_p)
ani_sim_df = pd.DataFrame(anime_sim_p)

anime_title = anime_df['name'].values

# Page configurations
st.set_page_config(page_title="Anime Recommender", page_icon=":tv:")


@st.cache_data
# Collaborative Func.
def anime_recommendation(ani_name):
    try:
        st.subheader(f"Top 10 Similar Anime to {ani_name}")

        recommended_anime = []
        for anime in ani_sim_df.sort_values(by=ani_name, ascending=False).index[1:11]:
            match_percentage = round(ani_sim_df[anime][ani_name] * 100, 2)
            recommended_anime.append({'Anime Name': anime, 'Match Percentage': match_percentage})

        # Create a DataFrame for better visualization
        recommended_df = pd.DataFrame(recommended_anime)

        # Display the DataFrame in Streamlit
        st.table(recommended_df)

    except IndexError:
        st.warning("Anime not found. Please enter a valid anime name.")
    except Exception as e:
        st.error(f"An error occurred. Please try again.{e}")


# Function to render collaborative recommendation page
def render_collaborative_page():
    st.image('data//img1.png')
    st.subheader("on User-Rating")

    anime_name_input = st.selectbox(
        'Enter Anime', anime_title
    )

    # Recommendation button ;v
    if st.button("Get Recommendations"):
        anime_recommendation(anime_name_input)


@st.cache_data
# Content Based Func.
def content_based_recommendation(anime_name, anime_df=anime_df):

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(anime_df['genre'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Cosine_sim on saving thru pickle taking disk space >1GB.
    anime_index = anime_df[anime_df['name'] == anime_name].index[0]
    sim_scores = list(enumerate(cosine_sim[anime_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_anime_indices = [i[0] for i in sim_scores[1:11]]
    top_anime_names = anime_df['name'].iloc[top_anime_indices]
    top_10_anime = top_anime_names.reset_index(drop=True)

    return top_10_anime


# Function to render content-based recommendation page
def render_content_based_page():
    # st.title("Genre-Based Recommendation")
    st.image('data//img1.png')
    st.subheader('Genre-Based')
    anime_name = st.selectbox('Enter Anime', anime_title)

    try:
        # Recommend button
        if st.button("Recommend"):
            recommendations = content_based_recommendation(anime_name)

            st.write("Recommendations:")
            for i, anime in enumerate(recommendations, start=1):
                st.write(f"{i}. {anime}")

    except IndexError:
        st.error("Anime not found. Please enter a valid anime name.")
    except Exception as e:
        st.error(f"An error occurred. Please try again.{e}")


# Sidebar navigation
page = st.sidebar.radio("Select a page", ["Collaborative Recommendation", "Content-Based Recommendation"])

# Display the selected page
if page == "Collaborative Recommendation":
    render_collaborative_page()
else:
    render_content_based_page()
