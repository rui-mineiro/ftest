import pandas as pd

data = {
    "Low": [21.233999, 20.112000, 19.965000, 20.906000, 20.997999]
}

dates = pd.to_datetime([
    "2025-09-24",
    "2025-09-25",
    "2025-09-26",
    "2025-09-29",
    "2025-09-30"
])

df = pd.DataFrame(data, index=dates)
df.index.name = "Date"

