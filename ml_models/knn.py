from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DetectionKnn:
    def __init__(self):
        # We create the encoders for IP_Flags & TCP_FLags
        self.one_hot_encoder_ip = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.one_hot_encoder_tcp = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )

    # function to encode a column values with frequency of apperance in the column
    def frequency_encode(self, df, column_name):
        freq = df[column_name].value_counts()
        encoding = freq.rank(method="dense").astype(int)
        return df[column_name].map(encoding)

    def prepare_data(self, df):
        df = df.copy()
        # One-Hot Encoding pour les IP_Flags
        encoded_flags_ip = self.one_hot_encoder_ip.fit_transform(df[["IP_Flags"]])
        df_encoded_flags_ip = pd.DataFrame(
            encoded_flags_ip,
            columns=self.one_hot_encoder_ip.get_feature_names_out(["IP_Flags"]),
        )
        # One-Hot Encoding pour les TCP_Flags
        encoded_flags_tcp = self.one_hot_encoder_tcp.fit_transform(df[["TCP_Flags"]])
        df_encoded_flags_tcp = pd.DataFrame(
            encoded_flags_tcp,
            columns=self.one_hot_encoder_tcp.get_feature_names_out(["TCP_Flags"]),
        )
        df["src_ip_encoded"] = self.frequency_encode(df, "src_ip")
        df["dst_ip_encoded"] = self.frequency_encode(df, "dst_ip")
        df["src_port_encoded"] = self.frequency_encode(df, "src_port")
        df["dst_port_encoded"] = self.frequency_encode(df, "dst_port")
        # Remplacer les colonnes non encodées par les colonnes encodées dans le DataFrame
        columns_to_drop = [
            "IP_Flags",
            "TCP_Flags",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "IP_Version",
            "IP_IHL",
            "IP_TOS",
            "IP_Chksum",
            "Chksum",
            "IP_ID",
            "TCP_Dataofs",
            "TCP_Urgent",
            "IP_Timestamp",
        ]
        df_dropped = df.drop(columns=columns_to_drop)
        df_final = pd.concat(
            [df_dropped, df_encoded_flags_ip, df_encoded_flags_tcp], axis=1
        )
        imputer = SimpleImputer(strategy="mean")
        # Imputer les valeures manquantes
        X_imputed = imputer.fit_transform(df_final)
        # Normaliser les données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        return X_scaled

    # Function to label anomalies
    def classify_anomalies(self, row):
        if row["IP_Timestamp"] == 0:  # DoS
            return 1
        elif row["IP_Timestamp"] == 1:  # deletion
            return 2
        elif row["IP_Timestamp"] == 2:  # modification
            return 3
        elif row["IP_Timestamp"] == 3:  # injection
            return 4
        elif row["IP_Timestamp"] == 4:  # Nmap
            return 5
        elif row["IP_Timestamp"] == 5:  # Reverse shell
            return 6
        else:
            return 0  # Normal

    def run_train(self, df_train: pd.DataFrame) -> KNeighborsClassifier:
        df_train = df_train.copy()
        x_train_prep = self.prepare_data(df_train)
        df_train["anomaly"] = df_train.apply(self.classify_anomalies, axis=1)
        y_train_prep = df_train["anomaly"]

        # SMOTE oversamples all minority classes until they reach the size of
        # the majority class.
        smote = SMOTE(random_state=42)
        x_train, y_train = smote.fit_resample(x_train_prep, y_train_prep)

        parameters = {
            "algorithm": "auto",
            "leaf_size": 30,
            "metric": "minkowski",
            "metric_params": None,
            "n_neighbors": 10,
            "p": 1,
            "weights": "distance",
        }

        knn = KNeighborsClassifier(**parameters)
        knn.fit(x_train, y_train)

        return knn

    def run_predict(self, df_test: pd.DataFrame, model: KNeighborsClassifier):
        df_test = df_test.copy()
        x_test = self.prepare_data(df_test)
        df_test["anomaly"] = df_test.apply(self.classify_anomalies, axis=1)
        y_test = df_test["anomaly"]

        y_pred = model.predict(x_test)

        return y_test, y_pred
