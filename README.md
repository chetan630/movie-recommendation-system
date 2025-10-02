# 🎬 Movie Recommendation System

A hybrid recommendation system that combines **Collaborative Filtering** and **Content-Based Filtering** to suggest movies to users.  
This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) and applies machine learning techniques like **cosine similarity** for personalized recommendations.

## 🚀 Features
- 📊 Hybrid approach (Collaborative + Content-based filtering)  
- 🔍 Movie similarity using **cosine similarity**  
- 👤 Personalized recommendations based on **user history**  
- 📂 Clean modular codebase (easy to extend)  

## 📁 Project Structure
```
movie-recommendation-system/
│── data/                 # Dataset (movies, ratings, tags, links)
│── scripts/                  # Source code
│   ├── movie_recommender.py
|
│── README.md             # Project documentation
│── requirements.txt      # Dependencies
```

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage
Run the main script to get recommendations:
```bash
python src/main.py
```

Example output:
```
Top 5 recommended movies for User 1:
1. The Matrix (1999)
2. Inception (2010)
3. Interstellar (2014)
...
```

## 📊 Dataset
- **Movies.csv** → movieId, title, genres  
- **Ratings.csv** → userId, movieId, rating, timestamp  
- **Tags.csv** → userId, movieId, tag, timestamp  
- **Links.csv** → movieId, imdbId, tmdbId  

Dataset: [MovieLens](https://grouplens.org/datasets/movielens/)

## 📌 Future Improvements
- Add **deep learning-based recommendations**  
- Build a simple **web app UI (Flask/Streamlit)**  
- Deploy using **Docker** or **Heroku**  

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss.  

## 📜 License
[MIT](LICENSE)
