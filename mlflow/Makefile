test_make:
	echo 123


activate_gcloud_service_account:
	gcloud auth activate-service-account --key-file=mlflow-for-gcp/secrets/mlops-zoomcamp-final-project-0e0f7f932f27.json

mlflow_container:
	cd mlflow-for-gcp/
	make -C mlflow-for-gcp/ docker-auth
	export GCP_PROJECT=mlops-zoomcamp-final-project && \
	make -C mlflow-for-gcp/ build && \
	make -C mlflow-for-gcp/ tag && \
	make -C mlflow-for-gcp/ push