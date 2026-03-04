"""
Original classes:
- RV: right ventricle, 1
- MYO: myocardium, 2
- LV: left ventricle, 3
"""

# --- labels: important to unify them with the rest of the datasets ---
RV_LABEL = 1
MYO_LABEL = 2
LV_LABEL = 3
LABEL_TO_NAME = {
    RV_LABEL: "RV",
    MYO_LABEL: "MYO",
    LV_LABEL: "LV",
}

# --- phase: important to define the frame to be analyzed ---
