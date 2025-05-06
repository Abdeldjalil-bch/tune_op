import pandas as pd
import numpy as np
import streamlit as st
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Streamlit page configuration
st.set_page_config(page_title="hyperparameter tuning", page_icon=":guardsman:", layout="wide")

# File uploader
file = st.file_uploader(
    label="Upload Train and Test Datasets",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="First upload train dataset, then test dataset (optional)"
)

# Initialize dataframes
train = pd.DataFrame()
test = pd.DataFrame()
submission=pd.DataFrame()
# File reading logic
if file:
    if len(file) == 1:
        try:
            # Read single file
            train = pd.read_csv(file[0]) if file[0].type == "text/csv" else pd.read_excel(file[0])
            st.write("Dataset provided:")
            st.write(train.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

    elif len(file) == 2:
        try:
            # Read train and test files
            train = pd.read_csv(file[0]) if file[0].type == "text/csv" else pd.read_excel(file[0])
            test = pd.read_csv(file[1]) if file[1].type == "text/csv" else pd.read_excel(file[1])

            st.write("Train Dataset:")
            st.write(train.head())
            st.write("Test Dataset:")
            st.write(test.head())
        except Exception as e:
            st.error(f"Error reading files: {e}")

    elif len(file) >= 3:
        try:
            # Read train and test files
            train = pd.read_csv(file[0]) if file[0].type == "text/csv" else pd.read_excel(file[0])
            test = pd.read_csv(file[1]) if file[1].type == "text/csv" else pd.read_excel(file[1])
            submission= pd.read_csv(file[2]) if file[1].type == "text/csv" else pd.read_excel(file[2])

            st.write("Train Dataset:")
            st.write(train.head())
            st.write("Test Dataset:")
            st.write(test.head())
            st.write("Submission Dataset")
            st.write(submission.head())
        except Exception as e:
            st.error(f"Error reading files: {e}")
    else:
        st.warning("Please upload at least one dataset")

# Feature and target selection
if not train.empty:
    target = st.selectbox(label="Select Target", options=train.columns)
    features = st.multiselect(label="Select Features", options=train.columns, default=[col for col in train.columns if col != target])
    
    submission_filename = st.text_input(
    label="Submission filename (optional)",
    value="submission.csv",
    help="Enter the desired filename for your submission file")
    if features and target:
        x = train[features]
        y = train[target]


        def create_custom_preprocessor(train):
            """
            Create a customizable preprocessing pipeline with Streamlit widgets
            """
            # Separate numeric and categorical columns
            numeric_columns = train.select_dtypes(include=[np.number]).columns
            categorical_columns = train.select_dtypes(exclude=[np.number]).columns

            st.subheader("Numeric Column Preprocessing")

            # Numeric Imputation Strategy
            numeric_imputation_options = [
                "Mean",
                "Median",
                "Most Frequent",
                "Constant",
                "KNN Imputer"
            ]
            num_impute_method = st.selectbox(
                "Numeric Imputation Method",
                numeric_imputation_options
            )

            # Numeric Scaling Strategy
            scaling_options = [
                "Standard Scaler",
                "MinMax Scaler",
                "Robust Scaler",
                "No Scaling"
            ]
            num_scaling_method = st.selectbox(
                "Numeric Scaling Method",
                scaling_options
            )

            st.subheader("Categorical Column Preprocessing")

            # Categorical Imputation Strategy
            cat_imputation_options = [
                "Most Frequent",
                "Constant",
                "Simple Imputer"
            ]
            cat_impute_method = st.selectbox(
                "Categorical Imputation Method",
                cat_imputation_options
            )

            # Categorical Encoding Strategy
            encoding_options = [
                "One-Hot Encoding",
                "Label Encoding",
                "No Encoding"
            ]
            cat_encoding_method = st.selectbox(
                "Categorical Encoding Method",
                encoding_options
            )

            # Construct Numeric Transformer
            numeric_transformer_steps = []

            # Imputation for Numeric Columns
            if num_impute_method == "Mean":
                numeric_transformer_steps.append(
                    SimpleImputer(strategy="mean")
                )
            elif num_impute_method == "Median":
                numeric_transformer_steps.append(
                    SimpleImputer(strategy="median")
                )
            elif num_impute_method == "Most Frequent":
                numeric_transformer_steps.append(
                    SimpleImputer(strategy="most_frequent")
                )
            elif num_impute_method == "Constant":
                constant_value = st.number_input(
                    "Constant Imputation Value",
                    value=0.0
                )
                numeric_transformer_steps.append(
                    SimpleImputer(strategy="constant", fill_value=constant_value)
                )
            elif num_impute_method == "KNN Imputer":
                knn_neighbors = st.slider(
                    "KNN Imputer Neighbors",
                    min_value=1,
                    max_value=50,
                    value=15
                )
                numeric_transformer_steps.append(
                    KNNImputer(n_neighbors=knn_neighbors)
                )

            # Scaling for Numeric Columns
            if num_scaling_method == "Standard Scaler":
                numeric_transformer_steps.append(StandardScaler())
            elif num_scaling_method == "MinMax Scaler":
                numeric_transformer_steps.append(MinMaxScaler())
            elif num_scaling_method == "Robust Scaler":
                numeric_transformer_steps.append(RobustScaler())

            # Construct Categorical Transformer
            categorical_transformer_steps = []

            # Imputation for Categorical Columns
            if cat_impute_method == "Most Frequent":
                categorical_transformer_steps.append(
                    SimpleImputer(strategy="most_frequent")
                )
            elif cat_impute_method == "Constant":
                constant_value = st.text_input(
                    "Constant Categorical Imputation Value",
                    value="Unknown"
                )
                categorical_transformer_steps.append(
                    SimpleImputer(strategy="constant", fill_value=constant_value)
                )

            # Encoding for Categorical Columns
            if cat_encoding_method == "One-Hot Encoding":
                categorical_transformer_steps.append(
                    OneHotEncoder(handle_unknown='ignore')
                )
            elif cat_encoding_method == "Label Encoding":
                categorical_transformer_steps.append(
                    LabelEncoder()
                )

            # Create pipelines
            numeric_transformer = make_pipeline(*numeric_transformer_steps) if numeric_transformer_steps else None
            categorical_transformer = make_pipeline(
                *categorical_transformer_steps) if categorical_transformer_steps else None

            # Create column transformer
            preprocessor = make_column_transformer(
                (numeric_transformer, make_column_selector(dtype_include=np.number)) if numeric_transformer else None,
                (categorical_transformer,
                 make_column_selector(dtype_exclude=np.number)) if categorical_transformer else None,
                remainder='passthrough'
            )

            return preprocessor


        # Example usage in main Streamlit app
        if not train.empty:
            # Call the function to create custom preprocessor
            preprocessor = create_custom_preprocessor(train)

        # Importations supplémentaires nécessaires
        from sklearn.svm import SVR, SVC
        from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        import lightgbm as lgb

        # Fonction pour créer et configurer un modèle avec Optuna
        def create_model_with_optuna(model_type, model_name, preprocessor):
            """
            Create a machine learning model and optimize parameters with Optuna
            """
            import optuna
            from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
            import streamlit as st
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.model_selection import cross_val_score
            
            st.subheader(f"Configuration d'Optuna pour {model_name}")
            
            # Paramètres généraux d'Optuna
            st.write("### Paramètres généraux d'optimisation")
            n_trials = st.number_input("Nombre d'essais", min_value=10, max_value=500, value=50, step=10)
            timeout = st.number_input("Timeout (secondes, 0 pour désactiver)", min_value=0, value=600, step=60)
            cv_folds = st.number_input("Nombre de folds pour validation croisée", min_value=2, value=5, step=1)
            
            # Définir les métriques d'évaluation selon le type de problème
            if model_type == "Regression":
                scoring_options = ["r2", "neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error", "neg_median_absolute_error"]
                scoring_display = ["R²", "MSE (négatif)", "RMSE (négatif)", "MAE (négatif)", "Median AE (négatif)"]
                default_index = 0  # R²
            else:  # Classification
                scoring_options = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc", "f1_macro"]
                scoring_display = ["Accuracy", "F1 Score (weighted)", "Precision (weighted)", "Recall (weighted)", "ROC AUC", "F1 Macro"]
                default_index = 0  # Accuracy
            
            # Créer un dictionnaire pour associer les noms affichés aux noms techniques
            scoring_dict = dict(zip(scoring_display, scoring_options))
            
            # Sélection des métriques d'évaluation
            selected_scoring_display = st.selectbox(
                "Métrique d'évaluation",
                options=scoring_display,
                index=default_index,
                help="Sélectionnez la métrique à optimiser pendant la recherche d'hyperparamètres"
            )
            selected_scoring = scoring_dict[selected_scoring_display]
            
            # Indiquer si c'est une métrique à maximiser ou minimiser
            maximize_metrics = ["r2", "accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc", "f1_macro"]
            direction = "maximize" if selected_scoring in maximize_metrics else "minimize"
            
            st.info(f"Direction d'optimisation: {'Maximiser' if direction=='maximize' else 'Minimiser'} la métrique {selected_scoring_display}")

            # Paramètres d'optimisation pour chaque modèle
            if model_type == "Regression":
                # Définition du modèle et des paramètres à optimiser selon le type de modèle
                if model_name == "Linear Regression":
                    st.write("### Paramètres pour Linear Regression")
                    fit_intercept = st.checkbox("Fit Intercept", value=True)
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', LinearRegression(fit_intercept=fit_intercept))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "KNN Regression":
                    st.write("### Paramètres pour KNN Regression")
                    n_neighbors_min = st.number_input("Minimum neighbors", min_value=1, value=3, step=1)
                    n_neighbors_max = st.number_input("Maximum neighbors", min_value=1, value=15, step=1)
                    
                    weights_options = st.multiselect("Types de pondération", ["uniform", "distance"], default=["uniform", "distance"])
                    metrics_options = st.multiselect("Métriques", ["euclidean", "manhattan", "chebyshev"], default=["euclidean", "manhattan"])
                    algorithm_options = st.multiselect("Algorithmes", ["auto", "ball_tree", "kd_tree", "brute"], default=["auto"])
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', KNeighborsRegressor(
                                n_neighbors=trial.suggest_int('n_neighbors', n_neighbors_min, n_neighbors_max),
                                weights=trial.suggest_categorical('weights', weights_options),
                                algorithm=trial.suggest_categorical('algorithm', algorithm_options),
                                metric=trial.suggest_categorical('metric', metrics_options)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "XGBoost Regression":
                    st.write("### Paramètres pour XGBoost Regression")
                    n_estimators_min = st.number_input("Minimum n_estimators", min_value=10, value=50, step=10)
                    n_estimators_max = st.number_input("Maximum n_estimators", min_value=10, value=300, step=10)
                    
                    learning_rate_min = st.number_input("Minimum learning_rate", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                    learning_rate_max = st.number_input("Maximum learning_rate", min_value=0.001, value=0.3, step=0.01, format="%.3f")
                    
                    max_depth_min = st.number_input("Minimum max_depth", min_value=1, value=3, step=1)
                    max_depth_max = st.number_input("Maximum max_depth", min_value=1, value=10, step=1)
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', xgb.XGBRegressor(
                                n_estimators=trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
                                learning_rate=trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max, log=True),
                                max_depth=trial.suggest_int('max_depth', max_depth_min, max_depth_max),
                                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                                gamma=trial.suggest_float('gamma', 0.0, 5.0)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "SVM Regression":
                    st.write("### Paramètres pour SVM Regression")
                    kernel_options = st.multiselect("Kernel", ["linear", "poly", "rbf", "sigmoid"], default=["linear", "rbf"])
                    
                    c_min = st.number_input("Minimum C", min_value=0.001, value=0.1, step=0.1, format="%.3f")
                    c_max = st.number_input("Maximum C", min_value=0.1, value=100.0, step=10.0)
                    
                    epsilon_min = st.number_input("Minimum epsilon", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                    epsilon_max = st.number_input("Maximum epsilon", min_value=0.01, value=1.0, step=0.1)
                    
                    def create_model(trial):
                        kernel = trial.suggest_categorical('kernel', kernel_options)
                        params = {
                            'kernel': kernel,
                            'C': trial.suggest_float('C', c_min, c_max, log=True),
                            'epsilon': trial.suggest_float('epsilon', epsilon_min, epsilon_max, log=True)
                        }
                        
                        # Ajouter des paramètres spécifiques au kernel
                        if kernel == 'poly':
                            params['degree'] = trial.suggest_int('degree', 2, 5)
                        
                        if kernel in ['rbf', 'poly', 'sigmoid']:
                            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto']) or \
                                            trial.suggest_float('gamma_value', 0.001, 1.0, log=True)
                        
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', SVR(**params))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "AdaBoost Regression":
                    st.write("### Paramètres pour AdaBoost Regression")
                    n_estimators_min = st.number_input("Minimum n_estimators", min_value=10, value=50, step=10)
                    n_estimators_max = st.number_input("Maximum n_estimators", min_value=10, value=300, step=10)
                    
                    learning_rate_min = st.number_input("Minimum learning_rate", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                    learning_rate_max = st.number_input("Maximum learning_rate", min_value=0.01, value=1.0, step=0.1)
                    
                    loss_options = st.multiselect("Fonctions de perte", ["linear", "square", "exponential"], default=["linear", "square"])
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', AdaBoostRegressor(
                                n_estimators=trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
                                learning_rate=trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max, log=True),
                                loss=trial.suggest_categorical('loss', loss_options)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "Random Forest Regression":
                    st.write("### Paramètres pour Random Forest Regression")
                    n_estimators_min = st.number_input("Minimum n_estimators", min_value=10, value=50, step=10)
                    n_estimators_max = st.number_input("Maximum n_estimators", min_value=10, value=300, step=10)
                    
                    max_depth_min = st.number_input("Minimum max_depth", min_value=1, value=5, step=1)
                    max_depth_max = st.number_input("Maximum max_depth", min_value=5, value=30, step=5)
                    use_none_max_depth = st.checkbox("Inclure None comme max_depth", value=True)
                    
                    min_samples_split_min = st.number_input("Minimum min_samples_split", min_value=2, value=2, step=1)
                    min_samples_split_max = st.number_input("Maximum min_samples_split", min_value=2, value=20, step=2)
                    
                    min_samples_leaf_min = st.number_input("Minimum min_samples_leaf", min_value=1, value=1, step=1)
                    min_samples_leaf_max = st.number_input("Maximum min_samples_leaf", min_value=1, value=10, step=1)
                    
                    def create_model(trial):
                        # Traitement spécial pour max_depth pour inclure None
                        if use_none_max_depth and trial.suggest_categorical('use_max_depth', [True, False]) == False:
                            max_depth = None
                        else:
                            max_depth = trial.suggest_int('max_depth', max_depth_min, max_depth_max)
                        
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(
                                n_estimators=trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
                                max_depth=max_depth,
                                min_samples_split=trial.suggest_int('min_samples_split', min_samples_split_min, min_samples_split_max),
                                min_samples_leaf=trial.suggest_int('min_samples_leaf', min_samples_leaf_min, min_samples_leaf_max),
                                bootstrap=trial.suggest_categorical('bootstrap', [True, False])
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "Gradient Boosting Regression":
                    st.write("### Paramètres pour Gradient Boosting Regression")
                    n_estimators_min = st.number_input("Minimum n_estimators", min_value=10, value=50, step=10)
                    n_estimators_max = st.number_input("Maximum n_estimators", min_value=10, value=300, step=10)
                    
                    learning_rate_min = st.number_input("Minimum learning_rate", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                    learning_rate_max = st.number_input("Maximum learning_rate", min_value=0.01, value=0.3, step=0.01, format="%.3f")
                    
                    max_depth_min = st.number_input("Minimum max_depth", min_value=1, value=3, step=1)
                    max_depth_max = st.number_input("Maximum max_depth", min_value=1, value=10, step=1)
                    
                    subsample_min = st.number_input("Minimum subsample", min_value=0.1, value=0.6, step=0.1)
                    subsample_max = st.number_input("Maximum subsample", min_value=0.1, value=1.0, step=0.1)
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', GradientBoostingRegressor(
                                n_estimators=trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
                                learning_rate=trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max, log=True),
                                max_depth=trial.suggest_int('max_depth', max_depth_min, max_depth_max),
                                subsample=trial.suggest_float('subsample', subsample_min, subsample_max),
                                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "LightGBM Regression":
                    st.write("### Paramètres pour LightGBM Regression")
                    n_estimators_min = st.number_input("Minimum n_estimators", min_value=10, value=50, step=10)
                    n_estimators_max = st.number_input("Maximum n_estimators", min_value=10, value=300, step=10)
                    
                    learning_rate_min = st.number_input("Minimum learning_rate", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                    learning_rate_max = st.number_input("Maximum learning_rate", min_value=0.01, value=0.3, step=0.01, format="%.3f")
                    
                    max_depth_min = st.number_input("Minimum max_depth", min_value=-1, value=-1, step=1)
                    max_depth_max = st.number_input("Maximum max_depth", min_value=-1, value=20, step=1)
                    
                    num_leaves_min = st.number_input("Minimum num_leaves", min_value=2, value=31, step=1)
                    num_leaves_max = st.number_input("Maximum num_leaves", min_value=2, value=200, step=10)
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('regressor', lgb.LGBMRegressor(
                                n_estimators=trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
                                learning_rate=trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max, log=True),
                                max_depth=trial.suggest_int('max_depth', max_depth_min, max_depth_max),
                                num_leaves=trial.suggest_int('num_leaves', num_leaves_min, num_leaves_max),
                                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
            else:  # Classification
                # Définition du modèle et des paramètres à optimiser pour les modèles de classification
                if model_name == "Logistic Regression":
                    st.write("### Paramètres pour Logistic Regression")
                    solver_options = st.multiselect("Solvers", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"], default=["lbfgs", "liblinear"])
                    
                    c_min = st.number_input("Minimum C", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                    c_max = st.number_input("Maximum C", min_value=0.01, value=100.0, step=10.0)
                    
                    penalty_options = st.multiselect("Pénalités", ["l1", "l2", "elasticnet", "none"], default=["l2"])
                    
                    max_iter_min = st.number_input("Minimum max_iter", min_value=10, value=100, step=10)
                    max_iter_max = st.number_input("Maximum max_iter", min_value=100, value=1000, step=100)
                    
                    st.info("Note: Certaines combinaisons ne sont pas valides: 'l1' uniquement avec 'liblinear' ou 'saga', 'elasticnet' uniquement avec 'saga'")
                    
                    def create_model(trial):
                        solver = trial.suggest_categorical('solver', solver_options)
                        
                        # Gestion des combinaisons invalides
                        penalty_choices = list(penalty_options)
                        if solver == 'liblinear':
                            penalty_choices = [p for p in penalty_choices if p in ['l1', 'l2']]
                        elif solver == 'saga':
                            # saga est compatible avec toutes les pénalités
                            pass
                        elif solver in ['newton-cg', 'sag', 'lbfgs']:
                            penalty_choices = [p for p in penalty_choices if p in ['l2', 'none']]
                        
                        if not penalty_choices:
                            penalty_choices = ['l2']  # Valeur par défaut si aucune option valide
                        
                        penalty = trial.suggest_categorical('penalty', penalty_choices)
                        
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(
                                solver=solver,
                                penalty=penalty,
                                C=trial.suggest_float('C', c_min, c_max, log=True),
                                max_iter=trial.suggest_int('max_iter', max_iter_min, max_iter_max)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "KNN Classification":
                    st.write("### Paramètres pour KNN Classification")
                    n_neighbors_min = st.number_input("Minimum neighbors", min_value=1, value=3, step=1)
                    n_neighbors_max = st.number_input("Maximum neighbors", min_value=1, value=15, step=1)
                    
                    weights_options = st.multiselect("Types de pondération", ["uniform", "distance"], default=["uniform", "distance"])
                    metrics_options = st.multiselect("Métriques", ["euclidean", "manhattan", "chebyshev"], default=["euclidean", "manhattan"])
                    algorithm_options = st.multiselect("Algorithmes", ["auto", "ball_tree", "kd_tree", "brute"], default=["auto"])
                    
                    p_min = st.number_input("Minimum p (pour Minkowski)", min_value=1, value=1, step=1)
                    p_max = st.number_input("Maximum p (pour Minkowski)", min_value=1, value=5, step=1)
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', KNeighborsClassifier(
                                n_neighbors=trial.suggest_int('n_neighbors', n_neighbors_min, n_neighbors_max),
                                weights=trial.suggest_categorical('weights', weights_options),
                                algorithm=trial.suggest_categorical('algorithm', algorithm_options),
                                metric=trial.suggest_categorical('metric', metrics_options),
                                p=trial.suggest_int('p', p_min, p_max)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "XGBoost Classification":
                    st.write("### Paramètres pour XGBoost Classification")
                    n_estimators_min = st.number_input("Minimum n_estimators", min_value=10, value=50, step=10)
                    n_estimators_max = st.number_input("Maximum n_estimators", min_value=10, value=300, step=10)
                    
                    learning_rate_min = st.number_input("Minimum learning_rate", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                    learning_rate_max = st.number_input("Maximum learning_rate", min_value=0.01, value=0.3, step=0.01, format="%.3f")
                    
                    max_depth_min = st.number_input("Minimum max_depth", min_value=1, value=3, step=1)
                    max_depth_max = st.number_input("Maximum max_depth", min_value=1, value=10, step=1)
                    
                    gamma_min = st.number_input("Minimum gamma", min_value=0.0, value=0.0, step=0.1)
                    gamma_max = st.number_input("Maximum gamma", min_value=0.0, value=5.0, step=0.5)
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', xgb.XGBClassifier(
                                n_estimators=trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
                                learning_rate=trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max, log=True),
                                max_depth=trial.suggest_int('max_depth', max_depth_min, max_depth_max),
                                gamma=trial.suggest_float('gamma', gamma_min, gamma_max),
                                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "SVM Classification":
                    st.write("### Paramètres pour SVM Classification")
                    kernel_options = st.multiselect("Kernel", ["linear", "poly", "rbf", "sigmoid"], default=["linear", "rbf"])
                    
                    c_min = st.number_input("Minimum C", min_value=0.001, value=0.1, step=0.1, format="%.3f")
                    c_max = st.number_input("Maximum C", min_value=0.1, value=100.0, step=10.0)
                    
                    gamma_options = st.multiselect("Options gamma", ["scale", "auto"], default=["scale", "auto"])
                    include_gamma_values = st.checkbox("Inclure valeurs gamma personnalisées", value=True)
                    
                    if include_gamma_values:
                        gamma_min = st.number_input("Minimum gamma value", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                        gamma_max = st.number_input("Maximum gamma value", min_value=0.01, value=1.0, step=0.1)
                    
                    def create_model(trial):
                        kernel = trial.suggest_categorical('kernel', kernel_options)
                        params = {
                            'kernel': kernel,
                            'C': trial.suggest_float('C', c_min, c_max, log=True),
                            'probability': True
                        }
                        
                        # Gamma handling
                        if kernel in ['rbf', 'poly', 'sigmoid']:
                            if gamma_options and include_gamma_values:
                                gamma_type = trial.suggest_categorical('gamma_type', ['predefined', 'custom'])
                                if gamma_type == 'predefined':
                                    params['gamma'] = trial.suggest_categorical('gamma', gamma_options)
                                else:
                                    params['gamma'] = trial.suggest_float('gamma_value', gamma_min, gamma_max, log=True)
                            elif gamma_options:
                                params['gamma'] = trial.suggest_categorical('gamma', gamma_options)
                            elif include_gamma_values:
                                params['gamma'] = trial.suggest_float('gamma_value', gamma_min, gamma_max, log=True)
                        
                        # Degree handling for poly kernel
                        if kernel == 'poly':
                            params['degree'] = trial.suggest_int('degree', 2, 5)
                        
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', SVC(**params))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "AdaBoost Classification":
                    st.write("### Paramètres pour AdaBoost Classification")
                    n_estimators_min = st.number_input("Minimum n_estimators", min_value=10, value=50, step=10)
                    n_estimators_max = st.number_input("Maximum n_estimators", min_value=10, value=300, step=10)
                    
                    learning_rate_min = st.number_input("Minimum learning_rate", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                    learning_rate_max = st.number_input("Maximum learning_rate", min_value=0.01, value=1.0, step=0.1)
                    
                    algorithm_options = st.multiselect("Algorithmes", ["SAMME", "SAMME.R"], default=["SAMME.R"])
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', AdaBoostClassifier(
                                n_estimators=trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
                                learning_rate=trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max, log=True),
                                algorithm=trial.suggest_categorical('algorithm', algorithm_options)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "Random Forest Classification":
                    st.write("### Paramètres pour Random Forest Classification")
                    n_estimators_min = st.number_input("Minimum n_estimators", min_value=10, value=50, step=10)
                    n_estimators_max = st.number_input("Maximum n_estimators", min_value=10, value=300, step=10)
                    
                    max_depth_min = st.number_input("Minimum max_depth", min_value=1, value=5, step=1)
                    max_depth_max = st.number_input("Maximum max_depth", min_value=5, value=30, step=5)
                    use_none_max_depth = st.checkbox("Inclure None comme max_depth", value=True)
                    
                    min_samples_split_min = st.number_input("Minimum min_samples_split", min_value=2, value=2, step=1)
                    min_samples_split_max = st.number_input("Maximum min_samples_split", min_value=2, value=20, step=2)
                    
                    min_samples_leaf_min = st.number_input("Minimum min_samples_leaf", min_value=1, value=1, step=1)
                    min_samples_leaf_max = st.number_input("Maximum min_samples_leaf", min_value=1, value=10, step=1)
                    
                    criterion_options = st.multiselect("Critères", ["gini", "entropy", "log_loss"], default=["gini", "entropy"])
                    
                    def create_model(trial):
                        # Traitement spécial pour max_depth pour inclure None
                        if use_none_max_depth and trial.suggest_categorical('use_max_depth', [True, False]) == False:
                            max_depth = None
                        else:
                            max_depth = trial.suggest_int('max_depth', max_depth_min, max_depth_max)
                        
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(
                                n_estimators=trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
                                max_depth=max_depth,
                                min_samples_split=trial.suggest_int('min_samples_split', min_samples_split_min, min_samples_split_max),
                                min_samples_leaf=trial.suggest_int('min_samples_leaf', min_samples_leaf_min, min_samples_leaf_max),
                                criterion=trial.suggest_categorical('criterion', criterion_options),
                                bootstrap=trial.suggest_categorical('bootstrap', [True, False])
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                
                elif model_name == "Gradient Boosting Classification":
                    st.write("### Paramètres pour Gradient Boosting Classification")
                    n_estimators_min = st.number_input("Minimum n_estimators", min_value=10, value=50, step=10)
                    n_estimators_max = st.number_input("Maximum n_estimators", min_value=10, value=300, step=10)
                    
                    learning_rate_min = st.number_input("Minimum learning_rate", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                    learning_rate_max = st.number_input("Maximum learning_rate", min_value=0.01, value=0.3, step=0.01, format="%.3f")
                    
                    max_depth_min = st.number_input("Minimum max_depth", min_value=1, value=3, step=1)
                    max_depth_max = st.number_input("Maximum max_depth", min_value=1, value=10, step=1)
                    
                    subsample_min = st.number_input("Minimum subsample", min_value=0.1, value=0.6, step=0.1)
                    subsample_max = st.number_input("Maximum subsample", min_value=0.1, value=1.0, step=0.1)
                    
                    criterion_options = st.multiselect("Critères", ["friedman_mse", "squared_error"], default=["friedman_mse", "squared_error"])
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', GradientBoostingClassifier(
                                n_estimators=trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
                                learning_rate=trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max, log=True),
                                max_depth=trial.suggest_int('max_depth', max_depth_min, max_depth_max),
                                subsample=trial.suggest_float('subsample', subsample_min, subsample_max),
                                criterion=trial.suggest_categorical('criterion', criterion_options),
                                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
                        
                elif model_name == "LightGBM Classification":
                    st.write("### Paramètres pour LightGBM Classification")
                    n_estimators_min = st.number_input("Minimum n_estimators", min_value=10, value=50, step=10)
                    n_estimators_max = st.number_input("Maximum n_estimators", min_value=10, value=300, step=10)
                    
                    learning_rate_min = st.number_input("Minimum learning_rate", min_value=0.001, value=0.01, step=0.01, format="%.3f")
                    learning_rate_max = st.number_input("Maximum learning_rate", min_value=0.01, value=0.3, step=0.01, format="%.3f")
                    
                    max_depth_min = st.number_input("Minimum max_depth", min_value=-1, value=-1, step=1)
                    max_depth_max = st.number_input("Maximum max_depth", min_value=-1, value=20, step=1)
                    
                    num_leaves_min = st.number_input("Minimum num_leaves", min_value=2, value=31, step=1)
                    num_leaves_max = st.number_input("Maximum num_leaves", min_value=2, value=200, step=10)
                    
                    boosting_type_options = st.multiselect("Type de boosting", ["gbdt", "dart", "goss", "rf"], default=["gbdt"])
                    
                    def create_model(trial):
                        return ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', lgb.LGBMClassifier(
                                n_estimators=trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
                                learning_rate=trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max, log=True),
                                max_depth=trial.suggest_int('max_depth', max_depth_min, max_depth_max),
                                num_leaves=trial.suggest_int('num_leaves', num_leaves_min, num_leaves_max),
                                boosting_type=trial.suggest_categorical('boosting_type', boosting_type_options),
                                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0)
                            ))
                        ])
                        
                    def objective(trial):
                        model = create_model(trial)
                        return cross_val_score(model, x, y, cv=cv_folds, scoring=selected_scoring).mean()
            
# Lancer l'optimisation avec Optuna
            run_optimization = st.button("Lancer l'optimisation des hyperparamètres")
            
            if run_optimization:
                st.write("### Démarrage de l'optimisation des hyperparamètres avec Optuna")
                
                # Configuration de timeout
                timeout_value = None if timeout == 0 else timeout
                
                # Création de l'étude avec visualisation en temps réel
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Conteneurs pour les visualisations
                history_plot = st.empty()
                param_importance_plot = st.empty()
                contour_plot = st.empty()
                
                # Pour stocker tous les essais
                trials_data = []
                
                class OptunaCallback:
                    def __init__(self, n_trials):
                        self.n_trials = n_trials
                        self.completed_trials = 0
                    
                    def __call__(self, study, trial):
                        self.completed_trials += 1
                        progress = min(self.completed_trials / self.n_trials, 1.0)
                        progress_bar.progress(progress)
                        
                        # Afficher le score correctement (positif pour les métriques négatives)
                        current_score = trial.value
                        display_score = -current_score if selected_scoring.startswith('neg_') else current_score
                        best_score = study.best_value
                        best_display_score = -best_score if selected_scoring.startswith('neg_') else best_score
                        
                        status_text.text(f"Essai {self.completed_trials}/{self.n_trials} complété. "
                                        f"Score actuel: {display_score:.4f}, "
                                        f"Meilleur score: {best_display_score:.4f}")
                        
                        # Collecter les données de cet essai
                        trial_data = {
                            'Essai': trial.number,
                            'Score': display_score,
                            **trial.params
                        }
                        trials_data.append(trial_data)
                        
                        # Mise à jour des graphiques tous les 5 essais ou au dernier essai
                        if self.completed_trials % 5 == 0 or self.completed_trials == self.n_trials:
                            try:
                                fig_history = plot_optimization_history(study)
                                history_plot.plotly_chart(fig_history, use_container_width=True)
                                
                                fig_importance = plot_param_importances(study)
                                param_importance_plot.plotly_chart(fig_importance, use_container_width=True)
                                
                                # Essayer de tracer le graphique de contour si assez d'essais
                                if self.completed_trials >= 10 and len(study.trials) >= 10:
                                    most_important_params = [p for p, _ in sorted(study.best_trial.params.items(), key=lambda x: x[1])[:2]]
                                    if len(most_important_params) >= 2:
                                        fig_contour = plot_contour(study, params=most_important_params[:2])
                                        contour_plot.plotly_chart(fig_contour, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Impossible de générer certains graphiques: {e}")
                
                try:
                    study = optuna.create_study(direction=direction)
                    callback = OptunaCallback(n_trials)
                    study.optimize(objective, n_trials=n_trials, timeout=timeout_value, callbacks=[callback])
                    
                    # Affichage des résultats finaux avec gestion des métriques négatives
                    best_score = study.best_value
                    if selected_scoring.startswith('neg_'):
                        display_score = -best_score  # Convertir en valeur positive pour l'affichage
                        st.success(f"Optimisation terminée! Meilleur score ({selected_scoring_display}): {display_score:.4f}")
                    else:
                        st.success(f"Optimisation terminée! Meilleur score ({selected_scoring_display}): {best_score:.4f}")
                    
                    st.write("### Meilleurs paramètres trouvés:")
                    
                    best_params_df = pd.DataFrame(study.best_params.items(), columns=["Paramètre", "Valeur"]).set_index("Paramètre")
                    st.dataframe(best_params_df, use_container_width=True)
                    
                    # Afficher tous les essais avec leurs combinaisons de paramètres et scores
                    st.write("### Tous les essais d'optimisation:")
                    
                    # Créer un DataFrame pour afficher tous les essais
                    trials_df = pd.DataFrame(trials_data)
                    
                    # Trier par score (du meilleur au pire)
                    trials_df = trials_df.sort_values(by='Score', ascending=False if direction == 'maximize' else True)
                    
                    # Réinitialiser l'index pour avoir un affichage propre
                    trials_df = trials_df.reset_index(drop=True)
                    
                    # Afficher le tableau avec tous les essais
                    st.dataframe(trials_df, use_container_width=True)
                    
                    # Option pour télécharger le tableau des essais
                    csv_trials = trials_df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger tous les essais d'optimisation (CSV)",
                        data=csv_trials,
                        file_name="optuna_trials_results.csv",
                        mime='text/csv',
                    )
                            
                    # Créer le modèle final avec les meilleurs paramètres
                    final_model = create_model(study.best_trial)
                    final_model.fit(x, y)
                    
                    # Prédictions et évaluation sur l'ensemble d'entraînement
                    y_pred = final_model.predict(x)
                    
                    if model_type == "Regression":
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        
                        mse = mean_squared_error(y, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y, y_pred)
                        r2 = r2_score(y, y_pred)
                        
                        metrics_df = pd.DataFrame({
                            "Métrique": ["MSE", "RMSE", "MAE", "R²"],
                            "Valeur": [mse, rmse, mae, r2]
                        }).set_index("Métrique")
                        
                        st.write("### Métriques d'évaluation (ensemble d'entraînement):")
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Visualisation des prédictions vs valeurs réelles
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y, y_pred, alpha=0.5)
                        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                        ax.set_xlabel('Valeurs réelles')
                        ax.set_ylabel('Prédictions')
                        ax.set_title('Prédictions vs Valeurs réelles')
                        st.pyplot(fig)
                        
                    else:  # Classification
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                        
                        # Calculer les métriques
                        accuracy = accuracy_score(y, y_pred)
                        
                        # Vérifier si c'est une classification binaire ou multiclasse
                        if len(np.unique(y)) == 2:
                            precision = precision_score(y, y_pred, average='binary')
                            recall = recall_score(y, y_pred, average='binary')
                            f1 = f1_score(y, y_pred, average='binary')
                        else:
                            precision = precision_score(y, y_pred, average='weighted')
                            recall = recall_score(y, y_pred, average='weighted')
                            f1 = f1_score(y, y_pred, average='weighted')
                        
                        metrics_df = pd.DataFrame({
                            "Métrique": ["Accuracy", "Precision", "Recall", "F1 Score"],
                            "Valeur": [accuracy, precision, recall, f1]
                        }).set_index("Métrique")
                        
                        st.write("### Métriques d'évaluation (ensemble d'entraînement):")
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Matrice de confusion
                        cm = confusion_matrix(y, y_pred)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        cax = ax.matshow(cm, cmap=plt.cm.Blues)
                        fig.colorbar(cax)
                        
                        classes = np.unique(y)
                        tick_marks = np.arange(len(classes))
                        plt.xticks(tick_marks, classes)
                        plt.yticks(tick_marks, classes)
                        
                        # Ajouter les valeurs dans la matrice
                        thresh = cm.max() / 2
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax.text(j, i, format(cm[i, j], 'd'),
                                        ha="center", va="center",
                                        color="white" if cm[i, j] > thresh else "black")
                        
                        ax.set_xlabel('Prédictions')
                        ax.set_ylabel('Valeurs réelles')
                        ax.set_title('Matrice de confusion')
                        st.pyplot(fig)
                    
                    # Visualisation avancée des paramètres
                    st.write("### Visualisation de la distribution des paramètres:")
                    
                    # Sélectionner les colonnes de paramètres (exclure Essai et Score)
                    param_columns = [col for col in trials_df.columns if col not in ['Essai', 'Score']]
                    
                    if len(param_columns) >= 2:
                        # Créer une matrice de pairplots pour visualiser les relations entre paramètres
                        try:
                            import plotly.express as px
                            
                            # Créer une figure avec des pairplots
                            if len(param_columns) <= 5:  # Limiter à 5 paramètres pour éviter une visualisation trop lourde
                                fig = px.scatter_matrix(
                                    trials_df,
                                    dimensions=param_columns,
                                    color='Score',
                                    color_continuous_scale='Viridis',
                                    title="Relations entre paramètres et scores"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Si trop de paramètres, ne montrer que les 5 plus importants
                                try:
                                    importances = optuna.importance.get_param_importances(study)
                                    top_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                                    top_param_names = [p[0] for p in top_params]
                                    
                                    fig = px.scatter_matrix(
                                        trials_df,
                                        dimensions=top_param_names,
                                        color='Score',
                                        color_continuous_scale='Viridis',
                                        title="Relations entre les 5 paramètres les plus importants et scores"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except:
                                    st.info("Trop de paramètres pour afficher une matrice complète. Seuls quelques paramètres sont visualisés.")
                                    sample_params = param_columns[:5]
                                    fig = px.scatter_matrix(
                                        trials_df,
                                        dimensions=sample_params,
                                        color='Score',
                                        color_continuous_scale='Viridis',
                                        title="Relations entre quelques paramètres et scores"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Impossible de générer la matrice de pairplots: {e}")
                    
                    # Si dataset de test disponible, faire des prédictions
                    if not test.empty:
                        st.write("### Prédictions sur l'ensemble de test")
                        
                        # Préparer les données de test (utiliser les mêmes features que pour l'entraînement)
                        x_test = test[features]
                        
                        # Faire des prédictions
                        test_predictions = final_model.predict(x_test)
                        
                        # Afficher un aperçu des prédictions
                        predictions_df = pd.DataFrame({
                            'ID': test.index,
                            'Prediction': test_predictions
                        })
                        st.dataframe(predictions_df.head(10), use_container_width=True)
                        
                        # Permettre le téléchargement des prédictions
                        if not submission.empty:
                            # Utiliser le format du fichier de soumission
                            submission_cols = submission.columns
                            
                            if len(submission_cols) >= 2:
                                id_col = submission_cols[0]
                                pred_col = submission_cols[1]
                                
                                final_submission = submission.copy()
                                final_submission[pred_col] = test_predictions
                            else:
                                final_submission = pd.DataFrame({
                                    'ID': test.index if 'id' not in test.columns else test['id'],
                                    'Prediction': test_predictions
                                })
                        else:
                            final_submission = pd.DataFrame({
                                'ID': test.index if 'id' not in test.columns else test['id'],
                                'Prediction': test_predictions
                            })
                        
                        # Convertir en CSV pour téléchargement
                        csv = final_submission.to_csv(index=False)
                        
                        st.download_button(
                            label="Télécharger le fichier de soumission",
                            data=csv,
                            file_name=submission_filename,
                            mime='text/csv',
                        )
                except Exception as e:
                    st.error(f"Erreur lors de l'optimisation: {e}")
                    st.error(f"Détails: {str(e)}")

            # Retourner le modèle configuré et la fonction objectif pour une utilisation externe si nécessaire
            return create_model, objective

        # Select the problem type
        problem_type = st.selectbox("Select Problem Type", ["Regression", "Classification"])

        # Select the model
        if problem_type == "Regression":
            model_options = [
                "Linear Regression",
                "KNN Regression",
                "SVM Regression",
                "Random Forest Regression",
                "XGBoost Regression",
                "AdaBoost Regression",
                "Gradient Boosting Regression",
                "LightGBM Regression"
            ]
        else:  # Classification
            model_options = [
                "Logistic Regression",
                "KNN Classification",
                "SVM Classification",
                "Random Forest Classification",
                "XGBoost Classification",
                "AdaBoost Classification",
                "Gradient Boosting Classification",
                "LightGBM Classification"
            ]

        model_name = st.selectbox("Select Model", model_options)

        # Create model with Optuna
        create_model_with_optuna(problem_type, model_name, preprocessor)
