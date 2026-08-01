from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)

# --------------------------------------------
# Load Dataset
# --------------------------------------------

df = pd.read_csv("Housing_SalePrice_Dataset_Updated_With_Nulls.csv")

# Features and Target
features = [
    'Lot Frontage',
    'Lot Area',
    'Overall Qual',
    'Year Built',
    'Year Remod/Add',
    'Mas Vnr Area',
    '1st Flr SF',
    'Gr Liv Area',
    'Garage Cars',
    'Garage Area'
]

target = 'SalePrice'

# Keep only required columns
df = df[features + [target]]

# Fill missing values
df = df.fillna(df.median(numeric_only=True))

# Prepare data
X = df[features]
y = df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)


# --------------------------------------------
# Home Page
# --------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        lot_frontage = float(request.form["lot_frontage"])
        lot_area = float(request.form["lot_area"])
        overall_qual = int(request.form["overall_qual"])
        year_built = int(request.form["year_built"])
        year_remod = int(request.form["year_remod"])
        mas_vnr_area = float(request.form["mas_vnr_area"])
        first_flr = float(request.form["first_flr"])
        gr_liv_area = float(request.form["gr_liv_area"])
        garage_cars = int(request.form["garage_cars"])
        garage_area = float(request.form["garage_area"])

        input_data = pd.DataFrame({
            'Lot Frontage': [lot_frontage],
            'Lot Area': [lot_area],
            'Overall Qual': [overall_qual],
            'Year Built': [year_built],
            'Year Remod/Add': [year_remod],
            'Mas Vnr Area': [mas_vnr_area],
            '1st Flr SF': [first_flr],
            'Gr Liv Area': [gr_liv_area],
            'Garage Cars': [garage_cars],
            'Garage Area': [garage_area]
        })

        prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)


# --------------------------------------------
# Run Flask Application
# --------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)