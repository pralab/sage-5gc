import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest


class DetectionIsolationForest:
    def __init__(self):
        # We create the encoders for IP_Flags & TCP_FLags
        self.one_hot_encoder_ip = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.one_hot_encoder_tcp = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.imputer = SimpleImputer(strategy="mean")

    # function to encode a column values with frequency of apperance in the column
    def frequency_encode(self, df, column_name):
        freq = df[column_name].value_counts()
        encoding = freq.rank(method="dense").astype(int)
        return df[column_name].map(encoding)

    # function to prepare the data, input: dataframe, return: ndarray (X)
    def prepare_data(self, df):
        df = df.copy()
        df["src_ip_encoded"] = self.frequency_encode(df, "src_ip")
        df["dst_ip_encoded"] = self.frequency_encode(df, "dst_ip")

        df["src_port_encoded"] = self.frequency_encode(df, "src_port")
        df["dst_port_encoded"] = self.frequency_encode(df, "dst_port")

        # Fit and transform IP_Flags
        encoded_ip_flags = self.one_hot_encoder_ip.fit_transform(df[["IP_Flags"]])
        df_encoded_ip_flags = pd.DataFrame(
            encoded_ip_flags,
            columns=self.one_hot_encoder_ip.get_feature_names_out(["IP_Flags"]),
        )

        # Fit and transform TCP_Flags
        encoded_tcp_flags = self.one_hot_encoder_tcp.fit_transform(df[["TCP_Flags"]])
        df_encoded_tcp_flags = pd.DataFrame(
            encoded_tcp_flags,
            columns=self.one_hot_encoder_tcp.get_feature_names_out(["TCP_Flags"]),
        )

        # Drop original columns and concatenate
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
        ]
        if "IP_Timestamp" in df.columns:
            columns_to_drop.append("IP_Timestamp")
        df_dropped = df.drop(columns=columns_to_drop)
        df_final = pd.concat(
            [df_dropped, df_encoded_ip_flags, df_encoded_tcp_flags], axis=1
        )
        test = df_final.tail(4)
        test.to_csv("test.csv", index=False)
        X_imputed = self.imputer.fit_transform(df_final)
        return X_imputed

    # Function to label anomalies
    def tag_anomalies(self, row):
        if row["IP_Timestamp"] == 0:  # DoS
            return -1
        elif row["IP_Timestamp"] == 1:  # deletion
            return -1
        elif row["IP_Timestamp"] == 2:  # modification
            return -1
        elif row["IP_Timestamp"] == 3:  # injection
            return -1
        elif row["IP_Timestamp"] == 4:  # Nmap
            return -1
        elif row["IP_Timestamp"] == 5:  # Reverse shell
            return -1
        else:
            return 1  # Normal

    def classify_anomalies(self, row, prediction):
        if prediction == True:
            if row["anomaly"] == 2 and row["predictions"] == -1:  # DoS
                return 2
            elif row["anomaly"] == 3 and row["predictions"] == -1:  # deletion
                return 3
            elif row["anomaly"] == 4 and row["predictions"] == -1:  # modification
                return 4
            elif row["anomaly"] == 5 and row["predictions"] == -1:  # injection
                return 5
            elif row["anomaly"] == 6 and row["predictions"] == -1:  # Nmap
                return 6
            elif row["anomaly"] == 7 and row["predictions"] == -1:  # Reverse shell
                return 7
            elif row["anomaly"] == 1 and row["predictions"] == 1:  # Normal
                return 1
            else:
                return 0  # Missclassified
        else:
            if row["IP_Timestamp"] == 0:  # DoS
                return 2
            elif row["IP_Timestamp"] == 1:  # deletion
                return 3
            elif row["IP_Timestamp"] == 2:  # modification
                return 4
            elif row["IP_Timestamp"] == 3:  # injection
                return 5
            elif row["IP_Timestamp"] == 4:  # Nmap
                return 6
            elif row["IP_Timestamp"] == 5:  # Reverse shell
                return 7
            else:
                return 1  # Normal

    def run_isolation_forest(self, df_train_csv, df_test_csv):
        # prepare training data
        df_train = pd.read_csv(df_train_csv)
        sorted_columns = sorted(df_train.columns)
        df_train = df_train[sorted_columns]
        x_train = self.prepare_data(df_train)

        # prepare test data
        df_test = pd.read_csv(df_test_csv)
        sorted_columns = sorted(df_test.columns)
        df_test = df_test[sorted_columns]
        x_test = self.prepare_data(df_test)
        df_test["anomaly"] = df_test.apply(self.tag_anomalies, axis=1)
        y_test = df_test["anomaly"]

        parameters = {
            "bootstrap": True,
            "contamination": 0.04,
            "max_features": 0.5,
            "max_samples": 256,
            "n_estimators": 50,
            "n_jobs": None,
            "random_state": 1424966477,
            "verbose": 0,
            "warm_start": False,
        }

        iso_forest = IsolationForest(**parameters)

        iso_forest.fit(x_train)
        y_pred = iso_forest.predict(x_test)

        return y_test, y_pred
