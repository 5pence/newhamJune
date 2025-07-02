## Deployment

`pip freeze > requirements.txt`

Generate a Procfile (Procfile) is has NO extension and must start with a capital letter
Inside Procfile type the line below:

`web: streamlit run app.py --server.port=$PORT`

 You can down the Heroku CLI: https://devcenter.heroku.com/articles/heroku-clior deploy inside heroku by linking it to you Github

If you use the CLI - then inside your terminal:

`heroku login`

`heroku create my-streamlit-app`

`git push heroku main`

## Issues

If the App crashes - check logs with heroku - `heroku log --tail`

Streamlit version mismatch? Pin correct version in requirements.txt

CORS or public access? Ensure App.py allows external access (`server.enableCORS = False`) if needed in config.
