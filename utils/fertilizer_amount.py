import pandas as pd

req_df = pd.read_csv("fertilizer_requirements.csv")

FERTILIZERS = {
    "Urea": {"N": 0.46, "P": 0.0,  "K": 0.0},
    "DAP":  {"N": 0.18, "P": 0.46, "K": 0.0},
    "MOP":  {"N": 0.0,  "P": 0.0,  "K": 0.60},
    "NPK_20_20_20": {"N": 0.20, "P": 0.20, "K": 0.20},
}

def calculate_fertilizer_amount(crop, soil_N, soil_P, soil_K,
                                area_ha, fertilizer_name,
                                rainfall_mm=None):

    row = req_df[req_df["crop"].str.lower() == crop.lower()]
    if row.empty:
        return "Unknown crop"

    row = row.iloc[0]

    # required nutrients per hectare
    need_N = max(row.N_req - soil_N, 0)
    need_P = max(row.P_req - soil_P, 0)
    need_K = max(row.K_req - soil_K, 0)

    fert = FERTILIZERS.get(fertilizer_name)
    if fert is None:
        return "Unknown fertilizer"

    amounts = []

    # compute kg fertilizer needed per ha for each nutrient it supplies
    if fert["N"] > 0 and need_N > 0:
        amounts.append(need_N / fert["N"])
    if fert["P"] > 0 and need_P > 0:
        amounts.append(need_P / fert["P"])
    if fert["K"] > 0 and need_K > 0:
        amounts.append(need_K / fert["K"])

    if not amounts:
        return "Soil already sufficient for this fertilizer"

    kg_per_ha = max(amounts)  # ensure all deficits covered

    # simple weather adjustment
    if rainfall_mm is not None:
        if rainfall_mm > 200:
            kg_per_ha *= 1.10  # leaching loss
        elif rainfall_mm < 50:
            kg_per_ha *= 0.90  # avoid burn

    total_kg = kg_per_ha * area_ha

    return round(total_kg, 2)
