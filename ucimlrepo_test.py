from ucimlrepo import fetch_ucirepo
import pandas as pd

# with api

wine_quality = fetch_ucirepo(id=186) 

X = wine_quality.data.features 
y = wine_quality.data.targets

print(len(X))
