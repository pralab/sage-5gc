from ml_models import DetectionRandomForest, DetectionIsolationForest, DetectionKnn

if __name__ == "__main__":
    df_train_csv = "data/train_set_all.csv"
    df_test_csv = "data/test_set_all.csv"
    model = "knn"

    if model =="isolation_forest":
        detection = DetectionIsolationForest()
        detection.run_isolation_forest(df_train_csv, df_test_csv)
    elif model == "knn":
        detection = DetectionKnn()
        detection.run_knn(df_train_csv, df_test_csv)
    elif model == "random_forest":
        detection = DetectionRandomForest()
        detection.run_random_forest(df_train_csv, df_test_csv)