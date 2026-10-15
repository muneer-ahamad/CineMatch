
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ======================================
# PAGE SETUP
# ======================================

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 CineMatch")
st.caption("Movie Recommendation System using Python, NumPy, Pandas & TMDB")

DATA_DIR = Path(__file__).parent / "data"

CURRENT_YEAR = 2026

SORT_OPTIONS = [
    "Best match",
    "Highest rated",
    "Lowest rated",
    "Newest first",
    "Oldest first",
    "Title A-Z",
    "Title Z-A"
]


# ======================================
# TMDB API KEY
# ======================================

def get_api_key():
    key = os.getenv("TMDB_API_KEY", "")

    if key:
        return key

    try:
        return st.secrets.get("TMDB_API_KEY", "")
    except (FileNotFoundError, KeyError):
        return ""


# ======================================
# LOAD OFFLINE MOVIELENS DATA
# ======================================

@st.cache_data
def load_movies():

    movies = pd.read_csv(DATA_DIR / "movies.csv")

    required = {"movieId", "title", "genres"}

    if not required.issubset(movies.columns):
        raise ValueError(
            "movies.csv must have movieId, title and genres"
        )

    ratings_file = DATA_DIR / "ratings.csv"

    if ratings_file.exists():

        ratings = pd.read_csv(ratings_file)

        summary = ratings.groupby("movieId")["rating"].agg(
            average_rating="mean",
            rating_count="count"
        ).reset_index()

        movies = movies.merge(
            summary,
            on="movieId",
            how="left"
        )

    else:
        movies["average_rating"] = np.nan
        movies["rating_count"] = 0

    movies["rating_count"] = (
        movies["rating_count"].fillna(0).astype(int)
    )

    movies["genres"] = movies["genres"].fillna("")

    # Extract release year from movie title
    movies["year"] = pd.to_numeric(
        movies["title"].str.extract(
            r"\((\d{4})\)\s*$"
        )[0],
        errors="coerce"
    )

    return movies


# ======================================
# GENRE VECTORIZATION USING NUMPY
# ======================================

def genre_vector(genres, all_genres):

    if isinstance(genres, str):
        selected = set(genres.split("|"))
    else:
        selected = set(genres)

    vector = np.array([
        int(genre in selected)
        for genre in all_genres
    ], dtype=float)

    return vector


# ======================================
# COSINE SIMILARITY
# ======================================

def similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ======================================
# OFFLINE RECOMMENDATION ENGINE
# ======================================

def offline_recommend(
    movies,
    all_genres,
    selected_genres=None,
    target_title=None
):

    if target_title:

        chosen = movies[
            movies["title"] == target_title
        ].iloc[0]

        user_vector = genre_vector(
            chosen["genres"],
            all_genres
        )

    else:

        user_vector = genre_vector(
            selected_genres or [],
            all_genres
        )

    results = []

    for _, movie in movies.iterrows():

        if target_title and movie["title"] == target_title:
            continue

        match = similarity(
            user_vector,
            genre_vector(movie["genres"], all_genres)
        )

        if match == 0:
            continue

        rating = movie["average_rating"]

        if pd.notna(rating):
            rating_bonus = float(rating) / 5
        else:
            rating_bonus = 0

        score = (
            0.85 * match +
            0.15 * rating_bonus
        ) * 100

        results.append({
            "Movie": movie["title"],
            "Year": movie["year"],
            "Genres": movie["genres"].replace("|", ", "),
            "Avg rating / 5": (
                round(float(rating), 2)
                if pd.notna(rating)
                else np.nan
            ),
            "Ratings": int(movie["rating_count"]),
            "Match score": round(score, 1)
        })

    return pd.DataFrame(results)


# ======================================
# FILTERING AND SORTING FUNCTION
# ======================================

def filter_and_sort(
    df,
    start_year,
    end_year,
    order,
    rating_col,
    limit=20
):

    if df.empty:
        return df

    # Filter by release year
    df = df[
        df["Year"].between(
            start_year,
            end_year
        )
    ].copy()

    # Sort by selected option
    if order == "Highest rated":

        df = df.sort_values(
            rating_col,
            ascending=False,
            na_position="last"
        )

    elif order == "Lowest rated":

        df = df.sort_values(
            rating_col,
            ascending=True,
            na_position="last"
        )

    elif order == "Newest first":

        df = df.sort_values(
            "Year",
            ascending=False
        )

    elif order == "Oldest first":

        df = df.sort_values(
            "Year",
            ascending=True
        )

    elif order == "Title A-Z":

        df = df.sort_values(
            "Movie",
            key=lambda x: x.str.lower()
        )

    elif order == "Title Z-A":

        df = df.sort_values(
            "Movie",
            key=lambda x: x.str.lower(),
            ascending=False
        )

    elif "Match score" in df.columns:

        df = df.sort_values(
            "Match score",
            ascending=False
        )

    return df.head(limit).reset_index(drop=True)


# ======================================
# TMDB API REQUEST
# ======================================

def tmdb_get(endpoint, key, params=None):

    options = dict(params or {})

    options["api_key"] = key

    response = requests.get(
        f"https://api.themoviedb.org/3/{endpoint}",
        params=options,
        timeout=12
    )

    response.raise_for_status()

    return response.json()


# ======================================
# CONVERT TMDB RESPONSE TO DATAFRAME
# ======================================

def tmdb_to_table(items):

    rows = []

    for movie in items:

        date = movie.get("release_date") or ""

        if re.match(r"^\d{4}", date):
            year = int(date[:4])
        else:
            year = np.nan

        rows.append({
            "Movie": movie.get("title", "Untitled"),
            "Year": year,
            "Rating / 10": movie.get("vote_average") or np.nan,
            "Votes": movie.get("vote_count", 0),
            "Overview": (
                movie.get("overview")
                or "No description available"
            ),
            "Poster": movie.get("poster_path")
        })

    return pd.DataFrame(rows)


# ======================================
# DISPLAY ONLINE MOVIES
# ======================================

def show_online(df):

    if df.empty:
        st.info(
            "No movies match these filters. "
            "Try a wider year range."
        )
        return

    for _, movie in df.iterrows():

        left, right = st.columns([1, 4])

        with left:

            if (
                isinstance(movie["Poster"], str)
                and movie["Poster"]
            ):

                poster_url = (
                    "https://image.tmdb.org/t/p/w185"
                    + movie["Poster"]
                )

                st.image(poster_url)

        with right:

            if pd.notna(movie["Year"]):
                year = int(movie["Year"])
            else:
                year = "Unknown year"

            st.subheader(
                f"{movie['Movie']} ({year})"
            )

            if pd.notna(movie["Rating / 10"]):
                rating = f"{movie['Rating / 10']:.1f}/10"
            else:
                rating = "Unrated"

            st.write(
                f"TMDB Rating: {rating} | "
                f"{int(movie['Votes'])} votes"
            )

            st.write(movie["Overview"])

        st.divider()


# ======================================
# SIDEBAR FILTERS
# ======================================

st.sidebar.header("Movie Filters")

mode = st.sidebar.radio(
    "Choose Mode",
    [
        "Offline - MovieLens",
        "Online - TMDB"
    ]
)

start_year, end_year = st.sidebar.slider(
    "Release Year Range",
    min_value=1900,
    max_value=CURRENT_YEAR,
    value=(1990, CURRENT_YEAR)
)

sort_choice = st.sidebar.selectbox(
    "Sort Movies By",
    SORT_OPTIONS
)

count = st.sidebar.slider(
    "Number of Results",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)


# ======================================
# OFFLINE MODE
# ======================================

if mode.startswith("Offline"):

    st.header("Offline Movie Recommendations")

    try:

        movies = load_movies()

        # Extract unique genres
        all_genres = sorted({
            genre
            for entry in movies["genres"]
            for genre in entry.split("|")
            if genre and genre != "(no genres listed)"
        })

        st.write(
            f"Movies available: **{len(movies):,}**"
        )

        method = st.radio(
            "Recommendation Method",
            [
                "By Genres",
                "Similar to a Movie"
            ],
            horizontal=True
        )

        # Genre recommendation
        if method == "By Genres":

            selected = st.multiselect(
                "Select Your Favorite Genres",
                all_genres,
                default=(
                    ["Sci-Fi"]
                    if "Sci-Fi" in all_genres
                    else []
                )
            )

            if st.button(
                "Recommend Movies",
                type="primary"
            ):

                if not selected:
                    st.warning(
                        "Please select at least one genre."
                    )

                else:

                    df = offline_recommend(
                        movies,
                        all_genres,
                        selected_genres=selected
                    )

                    result = filter_and_sort(
                        df,
                        start_year,
                        end_year,
                        sort_choice,
                        "Avg rating / 5",
                        count
                    )

                    st.dataframe(
                        result,
                        use_container_width=True,
                        hide_index=True
                    )

        # Similar movie recommendation
        else:

            chosen_title = st.selectbox(
                "Choose a Movie",
                sorted(movies["title"].tolist())
            )

            if st.button(
                "Find Similar Movies",
                type="primary"
            ):

                df = offline_recommend(
                    movies,
                    all_genres,
                    target_title=chosen_title
                )

                result = filter_and_sort(
                    df,
                    start_year,
                    end_year,
                    sort_choice,
                    "Avg rating / 5",
                    count
                )

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )

        st.caption(
            "Offline recommendations use MovieLens "
            "and NumPy cosine similarity."
        )

        with st.expander("Preview Movie Dataset"):

            st.dataframe(
                movies.head(20),
                use_container_width=True,
                hide_index=True
            )

    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError
    ) as error:

        st.error(
            f"Could not load offline data: {error}"
        )


# ======================================
# ONLINE MODE
# ======================================

else:

    st.header("Online Movie Discovery")

    key = get_api_key()

    if not key:

        st.warning(
            "Set TMDB_API_KEY in Streamlit secrets "
            "to use online mode."
        )

    else:

        method = st.radio(
            "Online Recommendation Method",
            [
                "Discover Movies",
                "Similar to a Movie"
            ],
            horizontal=True
        )

        # ----------------------------------
        # DISCOVER MOVIES
        # ----------------------------------

        if method == "Discover Movies":

            st.caption(
                "Discover movies using TMDB's catalog "
                "with release year and sorting filters."
            )

            if st.button(
                "Discover Movies",
                type="primary"
            ):

                tmdb_sort = {
                    "Best match": "popularity.desc",
                    "Highest rated": "vote_average.desc",
                    "Lowest rated": "vote_average.asc",
                    "Newest first": "primary_release_date.desc",
                    "Oldest first": "primary_release_date.asc",
                    "Title A-Z": "original_title.asc",
                    "Title Z-A": "original_title.desc"
                }[sort_choice]

                params = {
                    "primary_release_date.gte":
                        f"{start_year}-01-01",

                    "primary_release_date.lte":
                        f"{end_year}-12-31",

                    "sort_by": tmdb_sort,

                    "vote_count.gte": (
                        50
                        if sort_choice in (
                            "Highest rated",
                            "Lowest rated"
                        )
                        else 0
                    ),

                    "include_adult": "false",
                    "page": 1
                }

                try:

                    data = tmdb_get(
                        "discover/movie",
                        key,
                        params
                    )

                    table = tmdb_to_table(
                        data.get("results", [])
                    )

                    result = filter_and_sort(
                        table,
                        start_year,
                        end_year,
                        "Best match",
                        "Rating / 10",
                        count
                    )

                    show_online(result)

                    st.caption(
                        "Showing results from TMDB's "
                        "first page (up to 20 movies)."
                    )

                except requests.RequestException as error:

                    st.error(
                        f"TMDB request failed: {error}"
                    )

        # ----------------------------------
        # SIMILAR MOVIE RECOMMENDATIONS
        # ----------------------------------

        else:

            title = st.text_input(
                "Enter Movie Title",
                value="Inception"
            )

            st.caption(
                "Search for a movie and get recommendations "
                "from TMDB."
            )

            if st.button(
                "Find Similar Movies",
                type="primary"
            ):

                if not title.strip():

                    st.warning(
                        "Please enter a movie title."
                    )

                else:

                    try:

                        # Search for the movie
                        found = tmdb_get(
                            "search/movie",
                            key,
                            {"query": title}
                        )

                        matches = found.get("results", [])

                        if not matches:

                            st.info(
                                "Movie not found. Try another title."
                            )

                        else:

                            movie = matches[0]

                            st.write(
                                "Selected Movie:",
                                movie.get("title")
                            )

                            movie_id = movie["id"]

                            # Get TMDB recommendations
                            data = tmdb_get(
                                f"movie/{movie_id}/recommendations",
                                key
                            )

                            table = tmdb_to_table(
                                data.get("results", [])
                            )

                            result = filter_and_sort(
                                table,
                                start_year,
                                end_year,
                                sort_choice,
                                "Rating / 10",
                                count
                            )

                            show_online(result)

                    except requests.RequestException as error:

                        st.error(
                            f"TMDB request failed: {error}"
                        )


# ======================================
# FOOTER
# ======================================

st.divider()

st.caption(
    "Offline mode uses NumPy genre similarity. "
    "Online mode uses the TMDB API."
)

st.caption(
    "This product uses the TMDB API but is not "
    "endorsed or certified by TMDB."
)
