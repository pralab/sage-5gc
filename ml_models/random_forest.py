import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class DetectionRandomForest:
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
        # Encoding IP & Port src/dst
        df["src_ip_encoded"] = self.frequency_encode(df, "src_ip")
        df["dst_ip_encoded"] = self.frequency_encode(df, "dst_ip")
        df["src_port_encoded"] = self.frequency_encode(df, "src_port")
        df["dst_port_encoded"] = self.frequency_encode(df, "dst_port")
        # Remplacer les colonnes non encodées par les collonnes encodées dans le DataFrame
        columns_to_drop = [
            "IP_Flags",
            "src_ip",
            "dst_ip",
            "TCP_Flags",
            "IP_Version",
            "IP_IHL",
            "dst_port",
            "src_port",
            "IP_TOS",
            "IP_Chksum",
            "Chksum",
            "IP_Flags",
            "IP_ID",
            "TCP_Dataofs",
            "TCP_Urgent",
            "IP_Timestamp",
        ]
        df_dropped = df.drop(columns=columns_to_drop)
        df_final = pd.concat(
            [df_dropped, df_encoded_flags_ip, df_encoded_flags_tcp], axis=1
        )
        # Imputer les valeures manquantes
        self.imputer = SimpleImputer(strategy="mean")
        X_imputed = self.imputer.fit_transform(df_final)
        return X_imputed

    def classify_anomalies(self, row):
        if row["IP_Timestamp"] == 0:  # DoS
            return 1
        elif row["IP_Timestamp"] == 1:  # deletion
            return 2
        elif row["IP_Timestamp"] == 2:  # modification
            return 3
        elif row["IP_Timestamp"] == 3:  # modification
            return 4
        elif row["IP_Timestamp"] == 4:  # Nmap
            return 5
        elif row["IP_Timestamp"] == 5:  # Reverse shell
            return 6
        else:
            return 0  # Normal

    def run_train(self, df_train: pd.DataFrame) -> RandomForestClassifier:
        df_train = df_train.copy()
        x_train = self.prepare_data(df_train)
        df_train["anomaly"] = df_train.apply(self.classify_anomalies, axis=1)
        y_train = df_train["anomaly"]

        parameters = {
            "bootstrap": True,
            "ccp_alpha": 0.0,
            "class_weight": "balanced_subsample",
            "criterion": "gini",
            "max_depth": 15,
            "max_features": "sqrt",
            "max_leaf_nodes": 30,
            "max_samples": 0.9,
            "min_impurity_decrease": 0.0,
            "min_weight_fraction_leaf": 0.0,
            "n_estimators": 50,
            "oob_score": False,
            "random_state": 42,
            "verbose": 0,
            "warm_start": False,
        }

        rand_forest = RandomForestClassifier(**parameters)
        rand_forest.fit(x_train, y_train)

        return rand_forest

    def run_predict(self, df_test: pd.DataFrame, model: RandomForestClassifier):
        df_test = df_test.copy()
        x_test = self.prepare_data(df_test)
        df_test["anomaly"] = df_test.apply(self.classify_anomalies, axis=1)
        y_test = df_test["anomaly"]

        y_pred = model.predict(x_test)

        return y_test, y_pred
