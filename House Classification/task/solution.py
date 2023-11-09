from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from provide_data import load_house_data, save_predicted
from encode import one_hot_encode, ordinal_encode, target_encoder


def evaluate_model(data, encoder_name, X_train_encoded, y_train, X_test_encoded, y_test):
    classifier = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best',
                                        max_depth=6, min_samples_split=4, random_state=3)
    classifier = classifier.fit(X_train_encoded, y_train)
    y_test_predicted = classifier.predict(X_test_encoded)
    save_predicted(y_test_predicted, f'price_predicted_target_{encoder_name}.csv')
    report = classification_report(y_test, y_test_predicted, output_dict=True)
    f1_macro_average = report['macro avg']['f1-score']
    print(f"{encoder_name}:{f1_macro_average:.2f}")


def classify_one_hot_encoder():
    data = load_house_data()
    data = one_hot_encode(data)
    evaluate_model(data, "OneHotEncoder", data.X_train_encoded, data.y_train, data.X_test_encoded, data.y_test)


def classify_ordinal_encoder():
    data = load_house_data()
    data = ordinal_encode(data)
    evaluate_model(data, "OrdinalEncoder", data.X_train_encoded, data.y_train, data.X_test_encoded, data.y_test)


def classify_target_encoder():
    data = load_house_data()
    data = target_encoder(data)
    evaluate_model(data, "TargetEncoder", data.X_train_encoded, data.y_train, data.X_test_encoded, data.y_test)


if __name__ == '__main__':
    classify_one_hot_encoder()
    classify_ordinal_encoder()
    classify_target_encoder()
