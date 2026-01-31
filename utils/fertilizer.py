import pandas as pd

DATA_PATH = "fertilizer_data.csv"

df = pd.read_csv(DATA_PATH)

def get_crop_info(crop):
    row = df[df["crop"].str.lower() == crop.lower()]
    if row.empty:
        return None
    return row.iloc[0]

def recommend_fertilizer(crop, soil_N, soil_P, soil_K, soil_ph):
    info = get_crop_info(crop)
    if info is None:
        return "Crop not found", None

    ideal_N = (info.N_min + info.N_max) / 2
    ideal_P = (info.P_min + info.P_max) / 2
    ideal_K = (info.K_min + info.K_max) / 2

    dN = ideal_N - soil_N
    dP = ideal_P - soil_P
    dK = ideal_K - soil_K

    deficits = {"N": dN, "P": dP, "K": dK}
    biggest = max(deficits.keys(), key=lambda k: deficits[k])


    ferts = info.common_fertilizers.split(";")
    pests = info.common_pesticides.split(";")

    if biggest == "N":
        rec = f"Use Nitrogen rich fertilizer like {ferts[0]}"
    elif biggest == "P":
        rec = f"Use Phosphorus rich fertilizer like {ferts[1]}"
    else:
        rec = f"Use Potassium rich fertilizer like {ferts[-1]}"

    ph_note = ""
    if soil_ph < info.ph_min:
        ph_note = "Soil too acidic. Add lime."
    elif soil_ph > info.ph_max:
        ph_note = "Soil too alkaline. Add organic matter or sulfur."

    return rec + (f" | {ph_note}" if ph_note else ""), pests
