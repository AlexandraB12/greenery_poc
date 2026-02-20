# check_addresses.py
import pandas as pd

df = pd.read_csv("adresses_final_streetview.csv")
df_ok = df[df["geocoding_source"].isin(["nominatim_ok", "manual_fix"])]

print(df_ok["geocoding_source"].value_counts())
print(f"TOTAL UTILISABLES : {len(df_ok)}")