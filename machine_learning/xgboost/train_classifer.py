import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

if __name__ == "__main__":
  # Load ds
  data, target = load_digits(return_X_y=True)
  # Split ds
  X_train, X_test, y_train, y_test = train_test_split(
                                  data, target, test_size=0.3, random_state=2)
  params = {'max_depth': 3,
           'learning_rate': 0.8,
           'num_parallel_tree': 50,
           'tree_method': 'gpu_hist'}

  model = xgb.XGBClassifier(**params)
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  acc = accuracy_score(y_test, pred)
  print(acc) # 0.9462962962962963
