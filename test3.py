import joblib

data = joblib.load('/home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features.pkl')
scaled_features = data['scaled_features']
print(f"Scaled features shape: {scaled_features.shape}")
