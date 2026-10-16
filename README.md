# Iris Flower Classifier Deployment

A small end-to-end machine learning deployment project. It trains a Random Forest classifier on scikit-learn's Iris dataset, serves predictions through a Flask API, and includes a browser interface, Dockerfile, and Kubernetes Deployment and Service manifests.

## Project files

- `train_model.py` trains the classifier and saves `iris_model.joblib`.
- `app.py` serves the frontend and the prediction API.
- `frontend/` contains the web interface and stylesheet.
- `DockerFile` builds the container image.
- `ml-k8s.yaml` defines a Kubernetes Deployment and NodePort Service.
- `requirements.txt` lists the Python dependencies.

## Run locally

Use Python 3.9 or newer. From this directory, install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`. The health endpoint is `GET /health`.

Send four numeric measurements in centimeters as `sepal_length`, `sepal_width`, `petal_length`, `petal_width` order (the API accepts the values in a `features` array):

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

The response includes the predicted class (`setosa`, `versicolor`, or `virginica`) and the numeric prediction code.

## Train the model

To regenerate the model artifact:

```bash
python train_model.py
```

This writes `iris_model.joblib` in the current working directory. The Flask app loads that artifact at startup.

## Build and run with Docker

Run these commands from this directory:

```bash
docker build -f DockerFile -t iris-model-api:v1 .
docker run --rm -p 5000:5000 iris-model-api:v1
```

## Deploy to Kubernetes

The manifest uses the local image `ml-model-api:v1` and `imagePullPolicy: Never`. Build the image with that tag in the cluster's image runtime, then apply the manifest:

```bash
docker build -f DockerFile -t ml-model-api:v1 .
kubectl apply -f ml-k8s.yaml
kubectl get deployment,service
```

The Service is configured as a NodePort on port `30500`.
