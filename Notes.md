Notes to Push new Data.md 

<!-- Connect to Server via Cloud Shell: -->
gcloud alpha cloud-shell ssh

<!-- Change user: -->
gcloud auth login

<!-- List & Change project: -->
gcloud projects list
gcloud config set project ilmapi

<!-- Shell Cloud SSH Password  -->
sls****1

<!-- Update Server -->
git pull

<!-- Reload Server with new features -->
gcloud app deploy app.yaml

<!-- Get the link to the API -->
gcloud app browse

<!-- Get logs -->
gcloud app logs tail -s default


<!-- Entrypoint changed -->
entrypoint: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app