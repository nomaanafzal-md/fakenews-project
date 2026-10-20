import json
import joblib

def export():
    print("Loading model.pkl and vec.pkl...")
    model = joblib.load("model.pkl")
    vec = joblib.load("vec.pkl")

    print("Extracting weights, vocabulary, and parameters...")
    data = {
        "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0])
    }

    with open("model.json", "w") as f:
        json.dump(data, f)
    print("model.json saved successfully!")

if __name__ == "__main__":
    export()
