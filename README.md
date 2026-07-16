# 🎌 Anime Recommender System

An intelligent Anime Recommendation System built using **Python**, **Streamlit**, and **Machine Learning** that suggests anime similar to a selected title.

The application uses a **content-based recommendation engine** powered by similarity scores to recommend anime based on their characteristics.

---

## 🚀 Demo

https://your-streamlit-link.streamlit.app

---

## 📸 Preview



| Home Page | Recommendations |
|-----------|-----------------|
| ![](images/home.png) | ![](images/result.png) |

---

# ✨ Features

- Search from thousands of anime titles
- Get top similar anime recommendations instantly
- Clean and responsive Streamlit interface
- Fast recommendation using precomputed similarity matrix
- Easy to run locally

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

---

# 📂 Project Structure

```
Anime_recommender/
│
├── app.py
├── anime.pkl
├── similarity.pkl
├── requirements.txt
├── README.md
│
├── data/
├── notebooks/
└── images/
```

---

# ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/mayankbungla/Anime_recommender.git
```

Move into the project folder

```bash
cd Anime_recommender
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🧠 How It Works

1. Load the anime dataset.
2. Perform preprocessing and feature engineering.
3. Generate feature vectors.
4. Compute cosine similarity between anime.
5. When a user selects an anime, retrieve the most similar titles based on similarity scores.
6. Display recommended anime along with their posters.

---

# 📊 Dataset

This project uses an anime dataset containing information such as:

- Anime Name
- Genre
- Type
- Rating
- Synopsis
- Episodes
- Popularity

---

# 📦 Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- requests
- pickle

Install them using:

```bash
pip install -r requirements.txt
```

---

# 💡 Future Improvements

- Hybrid Recommendation System
- Collaborative Filtering
- Personalized user recommendations
- Genre-based filtering
- Search autocomplete
- Better UI/UX
- Deploy using Docker
- User authentication

---

# 👨‍💻 Author

**Mayank Bungla**

GitHub: https://github.com/mayankbungla

---

# ⭐ If you found this project useful, consider giving it a star!
