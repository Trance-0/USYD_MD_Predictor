"""
This script is used for processing data to reduce redundency of code
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class USYD_data:
    def __init__(self, debug=False, drop=False) -> None:
        # depress warning: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        pd.options.mode.chained_assignment = None

        self.debug = debug
        # read csv data
        self.raw_df = pd.read_csv("./admission_data.csv")
        self.prediction = pd.read_csv("./prediction.csv")

        if self.debug:
            print("raw data:")
            print(self.raw_df.head())

        # data_cleaning, since the data is precious, we don't need to clean and just fill them with empty
        self.df = self.raw_df[
            [
                "MD / DMD",
                "Rurality",
                "Campus ( MD )",
                # "Places Selected ( DMD )",
                "USyd Over Other Offer ?",
                "Section 1",
                "Section 2",
                "Section 3",
                "Outcome",
            ]
        ]

        self.pred_x = self.prediction[
            [
                "MD / DMD",
                "Rurality",
                "Campus ( MD )",
                # "Places Selected ( DMD )",
                "USyd Over Other Offer ?",
                "Section 1",
                "Section 2",
                "Section 3",
            ]
        ]

        if drop:
            self.df = self.df.dropna()

        # vectorize dataframe
        # convert string to int
        self.MD_mapping = {k: v for v, k in enumerate(self.df["MD / DMD"].unique())}
        self.Rur_mapping = {k: v for v, k in enumerate(self.df["Rurality"].unique())}
        self.Camp_mapping = {
            k: v for v, k in enumerate(self.df["Campus ( MD )"].unique())
        }
        # the place data is incomplete and I believe they are irrelevent
        # self.Place_mapping = {k: v for v, k in enumerate(self.df["Places Selected ( DMD )"].unique())}
        self.Over_mapping = {
            k: v for v, k in enumerate(self.df["USyd Over Other Offer ?"].unique())
        }

        self.Res_mapping = {k: v for v, k in enumerate(self.df["Outcome"].unique())}
        self.Res_remapping = {v: k for v, k in enumerate(self.df["Outcome"].unique())}

        # map dataset
        self.df["MD / DMD"] = self.raw_df["MD / DMD"].map(self.MD_mapping)
        self.df["Rurality"] = self.raw_df["Rurality"].map(self.Rur_mapping)
        self.df["Campus ( MD )"] = self.raw_df["Campus ( MD )"].map(self.Camp_mapping)
        # self.df["Places Selected ( DMD )"] = self.raw_df["Places Selected ( DMD )"].map(Place_mapping)
        self.df["USyd Over Other Offer ?"] = self.raw_df["USyd Over Other Offer ?"].map(
            self.Over_mapping
        )

        # map prediction
        self.pred_x["MD / DMD"] = self.prediction["MD / DMD"].map(self.MD_mapping)
        self.pred_x["Rurality"] = self.prediction["Rurality"].map(self.Rur_mapping)
        self.pred_x["Campus ( MD )"] = self.prediction["Campus ( MD )"].map(
            self.Camp_mapping
        )
        # self.pred_x["Places Selected ( DMD )"] = self.prediction["Places Selected ( DMD )"].map(Place_mapping)
        self.pred_x["USyd Over Other Offer ?"] = self.prediction[
            "USyd Over Other Offer ?"
        ].map(self.Over_mapping)

        self.df["Outcome"] = self.raw_df["Outcome"].map(self.Res_mapping)

        # no need to introduce redundent message
        x = self.df[
            [
                "MD / DMD",
                "Rurality",
                "Campus ( MD )",
                # "Places Selected ( DMD )",
                "USyd Over Other Offer ?",
                "Section 1",
                "Section 2",
                "Section 3",
            ]
        ]
        y = self.df["Outcome"]

        self.x = x.to_numpy()
        self.y = y.to_numpy()

        self.pred_x = self.pred_x.to_numpy()

        if self.debug:
            print("vectorized data:")
            print(x[:5], x.shape)
            print(y[:5], y.shape)
            print(self.pred_x, self.pred_x.shape)
            print(self.Res_mapping)

    def get_data(self, split=0.3, random_state=None, normalization=False) -> tuple:
        if normalization:
            x = []
            y = self.y
            for i in self.x.astype("float32"):
                diff = max(i) - min(i)
                temp = []
                for j in i:
                    temp.append((j - min(i)) / diff)
                x.append(np.array(temp))
            x = np.array(x)
            if self.debug:
                print("normalized data:")
                print(x[:5], x.shape)
                print(y[:5], y.shape)
            return train_test_split(x, y, test_size=split, random_state=random_state)
        return train_test_split(self.x, self.y, test_size=split, random_state=random_state)
        
