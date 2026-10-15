# CineMatch: Offline + Online Movie Recommendations

A beginner-friendly Python-only Streamlit project.

## Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Offline mode

The included `data/movies.csv` is a tiny illustrative starter dataset. For the full MovieLens data, download https://files.grouplens.org/datasets/movielens/ml-latest-small.zip and replace `data/movies.csv` with `ml-latest-small/movies.csv`, and add `ml-latest-small/ratings.csv` to `data/`.

Offline mode computes binary genre vectors and cosine similarity with NumPy; pandas loads movies and averages movie ratings from ratings.csv. Without ratings.csv, recommendations use genre matching alone.

## Online mode

1. Request a noncommercial TMDB API key: https://www.themoviedb.org/settings/api
2. Create `.streamlit/secrets.toml` locally:

```toml
TMDB_API_KEY = "YOUR_TMDB_V3_API_KEY"
```

3. **Do not commit your secrets file**. On Streamlit Community Cloud, add the same value in app Secrets settings.
4. Online mode searches TMDB for a title and displays TMDB's returned recommendations. It does not run our offline NumPy algorithm.

## Deployment

Push these files to GitHub (excluding secrets). At https://share.streamlit.io/ create an app pointing to `app.py` and set `TMDB_API_KEY` in advanced settings. To display posters, internet access is needed, even for offline mode if you later add them.

## Limitations

- Demo CSV is illustrative, not part of MovieLens.
- Ratings in MovieLens are on a 0.5–5.0 scale.
- Title lookup in online mode chooses the first TMDB result, which may not be the intended movie for ambiguous titles.
- TMDB requires attribution, including use of TMDB branding as described by its terms. Add the official TMDB logo to your deployed interface before making it public.
