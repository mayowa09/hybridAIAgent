# =========================================================================
# NEURO-SYMBOLIC AI FOR MEDICAL DIAGNOSIS
# =========================================================================
# This system combines neural networks (for pattern recognition) with 
# symbolic AI (for rule-based reasoning) to diagnose heart disease and diabetes
# while providing explainable recommendations including SHAP explanations.

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import make_scorer
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


# Visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    PLOTTING_AVAILABLE = True
    print("Matplotlib/Seaborn loaded successfully!")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Plotting libraries not available - running without visualizations")

try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP loaded successfully!")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - running without SHAP explanations")

# =========================================================================
# DATA EXPORT AND VISUALIZATION UTILITIES
# =========================================================================

def save_classification_report(y_true, y_pred, filename, disease_type):
    """Save detailed classification metrics to CSV"""
    try:
        report = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        
        # Add metadata
        df_report.index.name = 'Metric'
        df_report['Disease_Type'] = disease_type
        df_report['Total_Samples'] = len(y_true)
        
        df_report.to_csv(filename, index=True)
        print(f"✓ Classification report saved to {filename}")
        return df_report
    except Exception as e:
        print(f"✗ Failed to save classification report: {e}")
        return None

def save_feature_importance(model, feature_names, filename, disease_type):
    """Save feature importance rankings to CSV"""
    try:
        importance = model.feature_importances_
        df_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
            "Disease_Type": disease_type,
            "Rank": range(1, len(feature_names) + 1)
        }).sort_values(by="Importance", ascending=False)
        
        df_importance['Rank'] = range(1, len(df_importance) + 1)
        df_importance.to_csv(filename, index=False)
        print(f"✓ Feature importance saved to {filename}")
        return df_importance
    except Exception as e:
        print(f"✗ Failed to save feature importance: {e}")
        return None

def save_patient_explanations(explanations, test_indices, filename, disease_type):
    """Export patient-level explanations and recommendations to CSV"""
    try:
        export_data = []
        
        for i, exp in enumerate(explanations):
            # Flatten explanation data for CSV export
            row = {
                'Patient_ID': exp['patient_id'],
                'Test_Index': test_indices[i],
                'Disease_Type': disease_type,
                'Final_Prediction': exp['final_prediction'],
                'Confidence': exp['confidence'],
                'Model_Disagreement': exp['disagreement_flag'],
                
                # Individual model results
                'Neural_Prediction': exp['model_ensemble']['neural_network']['prediction'],
                'Neural_Confidence': exp['model_ensemble']['neural_network']['confidence'],
                'RF_Prediction': exp['model_ensemble']['random_forest']['prediction'],
                'RF_Confidence': exp['model_ensemble']['random_forest']['confidence'],
                'NB_Prediction': exp['model_ensemble']['naive_bayes']['prediction'],
                'NB_Confidence': exp['model_ensemble']['naive_bayes']['confidence'],
                
                # Top features and SHAP
                'Top_Feature_1': exp['feature_importance'][0]['feature'] if exp['feature_importance'] else '',
                'Top_Feature_1_Value': exp['feature_importance'][0]['value'] if exp['feature_importance'] else '',
                'Top_Feature_1_Importance': exp['feature_importance'][0]['importance'] if exp['feature_importance'] else '',
                
                # Medical recommendations (first 3)
                'Recommendation_1': exp['symbolic_recommendations'][0] if len(exp['symbolic_recommendations']) > 0 else '',
                'Recommendation_2': exp['symbolic_recommendations'][1] if len(exp['symbolic_recommendations']) > 1 else '',
                'Recommendation_3': exp['symbolic_recommendations'][2] if len(exp['symbolic_recommendations']) > 2 else '',
                'Total_Recommendations': len(exp['symbolic_recommendations'])
            }
            
            # Add SHAP information if available
            if isinstance(exp['shap_explanation'], list) and len(exp['shap_explanation']) > 0:
                row['SHAP_Top_Feature'] = exp['shap_explanation'][0]['feature']
                row['SHAP_Top_Value'] = exp['shap_explanation'][0]['shap_value']
                row['SHAP_Top_Impact'] = exp['shap_explanation'][0]['impact']
            else:
                row['SHAP_Top_Feature'] = 'Not Available'
                row['SHAP_Top_Value'] = 0
                row['SHAP_Top_Impact'] = 'None'
            
            export_data.append(row)
        
        df_explanations = pd.DataFrame(export_data)
        df_explanations.to_csv(filename, index=False)
        print(f"✓ Patient explanations saved to {filename}")
        return df_explanations
        
    except Exception as e:
        print(f"✗ Failed to save patient explanations: {e}")
        return None

def plot_feature_importance(model, feature_names, title, filename):
    """Generate and save feature importance visualization"""
    if not PLOTTING_AVAILABLE:
        print(f"✗ Plotting not available - skipping {filename}")
        return
    
    try:
        importance = model.feature_importances_
        sorted_idx = importance.argsort()[::-1][:10]  # Top 10 features
        
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in sorted_idx], importance[sorted_idx], color='skyblue')
        plt.xlabel("Feature Importance")
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Feature importance plot saved to {filename}")
        
    except Exception as e:
        print(f"✗ Failed to create feature importance plot: {e}")

def plot_ensemble_agreement(explanations, title, filename):
    """Visualize model agreement/disagreement patterns"""
    if not PLOTTING_AVAILABLE:
        print(f"✗ Plotting not available - skipping {filename}")
        return
    
    try:
        agreement_data = [exp['disagreement_flag'] for exp in explanations]
        
        plt.figure(figsize=(8, 5))
        agreement_counts = pd.Series(agreement_data).value_counts()
        labels = ['Models Agree', 'Models Disagree']
        colors = ['lightgreen', 'lightcoral']
        
        plt.pie(agreement_counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Ensemble agreement plot saved to {filename}")
        
    except Exception as e:
        print(f"✗ Failed to create ensemble agreement plot: {e}")

# =========================================================================
# SYMBOLIC REASONING LAYER - HEART DISEASE TREATMENT RULES
# =========================================================================

def suggest_heart_treatment(patient, prediction_confidence):
    """
    Generate evidence-based treatment suggestions for heart disease patients
    
    This function mimics how a cardiologist thinks through a case:
    1. Evaluate individual risk factors (age, cholesterol, chest pain, etc.)
    2. Calculate overall risk score
    3. Provide specific recommendations based on clinical guidelines
    
    Args:
        patient: Dictionary/Series with patient's clinical data
        prediction_confidence: How confident the AI model is (0.0 to 1.0)
    
    Returns:
        List of clinical recommendations and explanations
    """
    rules = []  
    
    try:
        # Handle both Series and array-like data structures
        if hasattr(patient, 'iloc'):
            # DataFrame/Series format
            age = float(patient.iloc[0])
            chol = float(patient.iloc[4]) 
            cp = int(float(patient.iloc[2]))
            exang = int(float(patient.iloc[8]))
            thalach = float(patient.iloc[7])
            oldpeak = float(patient.iloc[9])
            trestbps = float(patient.iloc[3])
        elif hasattr(patient, 'keys'):
            # Dictionary format
            age = float(patient['age'])
            chol = float(patient['chol'])
            cp = int(float(patient['cp']))
            exang = int(float(patient['exang']))
            thalach = float(patient['thalach'])
            oldpeak = float(patient['oldpeak'])
            trestbps = float(patient['trestbps'])
        else:
            # Array format - assume standard column order
            age = float(patient[0])
            chol = float(patient[4])
            cp = int(float(patient[2]))
            exang = int(float(patient[8]))
            thalach = float(patient[7])
            oldpeak = float(patient[9])
            trestbps = float(patient[3])
    except (KeyError, IndexError, ValueError) as e:
        return [f"Unable to extract patient data ({str(e)}) - recommend comprehensive clinical evaluation"]
    
    # Initialize risk scoring system (higher score = higher risk)
    risk_score = 0
    
    # AGE-BASED RISK ASSESSMENT
    if age > 65:
        risk_score += 2
        rules.append("Advanced age (>65): Increased cardiovascular risk.")
    elif age > 50:
        risk_score += 1
        rules.append("Middle age (>50): Moderate cardiovascular risk.")
    
    # CHOLESTEROL ANALYSIS
    if chol > 240:
        risk_score += 2
        rules.append("High cholesterol (>240): Immediate statin therapy and dietary intervention.")
    elif chol > 200:
        risk_score += 1
        rules.append("Borderline cholesterol (>200): Lifestyle modifications recommended.")
    
    # CHEST PAIN TYPE ANALYSIS
    if cp == 0:  # Typical angina
        risk_score += 3
        rules.append("Typical angina: Urgent cardiac catheterization recommended.")
    elif cp == 1:  # Atypical angina
        risk_score += 2
        rules.append("Atypical angina: Stress testing and cardiology consultation.")
    
    # EXERCISE-INDUCED SYMPTOMS
    if exang == 1:
        risk_score += 2
        rules.append("Exercise-induced angina: Avoid strenuous activity, cardiac evaluation needed.")
    
    # HEART RATE RESPONSE ANALYSIS
    if thalach < 100:
        risk_score += 1
        rules.append("Low maximum heart rate: Monitor cardiac function and fitness level.")
    
    # ECG FINDINGS (ST DEPRESSION)
    if oldpeak > 2.0:
        risk_score += 3
        rules.append("Significant ST depression (>2.0): Immediate cardiology referral required.")
    elif oldpeak > 1.0:
        risk_score += 1
        rules.append("Mild ST depression: Monitor with follow-up ECG.")
    
    # BLOOD PRESSURE ASSESSMENT
    if trestbps > 140:
        risk_score += 1
        rules.append("Hypertension detected: Blood pressure management required.")
    
    # OVERALL RISK STRATIFICATION
    if risk_score >= 6:
        rules.append("HIGH RISK PATIENT: Immediate comprehensive cardiac evaluation required.")
    elif risk_score >= 3:
        rules.append("MODERATE RISK: Regular monitoring and preventive measures needed.")
    else:
        rules.append("LOW RISK: Routine preventive care and lifestyle counseling.")
    
    # AI CONFIDENCE-BASED RECOMMENDATIONS
    if prediction_confidence > 0.8:
        rules.append(f"High model confidence ({prediction_confidence:.2f}): Recommendation reliability is high.")
    elif prediction_confidence < 0.6:
        rules.append(f"Low model confidence ({prediction_confidence:.2f}): Consider additional testing.")
    
    return rules

# =========================================================================
# SYMBOLIC REASONING LAYER - DIABETES TREATMENT RULES
# =========================================================================

def suggest_diabetes_treatment(patient, prediction_confidence):
    """
    Generate evidence-based treatment suggestions for diabetes patients
    
    This function applies endocrinology guidelines to evaluate:
    1. Blood sugar markers (HbA1c, glucose)
    2. Risk factors (BMI, age, comorbidities)
    3. Lifestyle factors (smoking, gender considerations)
    
    Args:
        patient: Dictionary/Series with patient's clinical data
        prediction_confidence: How confident the AI model is (0.0 to 1.0)
    
    Returns:
        List of clinical recommendations and explanations
    """
    rules = []
    
    try:
        # Extract patient features - handle different data formats
        if hasattr(patient, 'get'):
            # Dictionary-like access
            age = float(patient.get('age', 0))
            bmi = float(patient.get('bmi', 0))
            hba1c = float(patient.get('HbA1c_level', 0))
            glucose = float(patient.get('blood_glucose_level', 0))
            hypertension = int(patient.get('hypertension', 0))
            heart_disease = int(patient.get('heart_disease', 0))
            gender = patient.get('gender', 'Unknown')
            smoking = patient.get('smoking_history', 'unknown')
        else:
            # Series/DataFrame access
            age = float(patient['age'])
            bmi = float(patient['bmi'])
            hba1c = float(patient['HbA1c_level'])
            glucose = float(patient['blood_glucose_level'])
            hypertension = int(patient['hypertension'])
            heart_disease = int(patient['heart_disease'])
            gender = patient['gender']
            smoking = patient['smoking_history']
    except (KeyError, ValueError) as e:
        return [f"Unable to extract diabetes patient data ({str(e)}) - recommend comprehensive evaluation"]

    risk_score = 0
    
    # HbA1c ANALYSIS (GOLD STANDARD)
    if hba1c >= 6.5:
        risk_score += 3
        rules.append("HbA1c ≥6.5%: Diabetes diagnosis confirmed, immediate treatment required.")
    elif hba1c >= 5.7:
        risk_score += 2
        rules.append("HbA1c 5.7-6.4%: Pre-diabetes detected, intensive lifestyle intervention needed.")
    elif hba1c >= 5.0:
        risk_score += 1
        rules.append("HbA1c 5.0-5.6%: Normal-high range, monitor regularly.")
    
    # BLOOD GLUCOSE ANALYSIS
    if glucose >= 200:
        risk_score += 3
        rules.append("Very high glucose (≥200): Immediate medical attention, possible diabetic emergency.")
    elif glucose >= 140:
        risk_score += 2
        rules.append("High glucose (≥140): Diabetes likely, medical evaluation needed.")
    elif glucose >= 100:
        risk_score += 1
        rules.append("Elevated glucose (≥100): Pre-diabetic range, lifestyle changes recommended.")
    
    # BMI-BASED RECOMMENDATIONS
    if bmi >= 35:
        risk_score += 2
        rules.append("Severe obesity (BMI ≥35): Bariatric surgery consultation, intensive weight management.")
    elif bmi >= 30:
        risk_score += 1
        rules.append("Obesity (BMI ≥30): Structured weight loss program essential.")
    elif bmi >= 25:
        risk_score += 1
        rules.append("Overweight (BMI ≥25): Diet and exercise modifications needed.")
    
    # AGE-BASED RISK FACTORS
    if age >= 65:
        risk_score += 2
        rules.append("Age ≥65: High-risk group, regular diabetes screening essential.")
    elif age >= 45:
        risk_score += 1
        rules.append("Age ≥45: Increased diabetes risk, annual screening recommended.")
    
    # COMORBIDITY ANALYSIS
    if hypertension == 1:
        risk_score += 2
        rules.append("Hypertension present: Dual cardiovascular-metabolic risk, aggressive management needed.")
    
    if heart_disease == 1:
        risk_score += 3
        rules.append("Heart disease present: Very high risk, cardio-diabetic care coordination required.")
    
    # SMOKING CONSIDERATIONS
    if smoking in ['current', 'former']:
        risk_score += 1
        rules.append(f"Smoking history ({smoking}): Increased diabetes complications risk, cessation counseling.")
    
    # GENDER-SPECIFIC CONSIDERATIONS
    if gender == 'Female' and age >= 45:
        rules.append("Post-menopausal female: Increased diabetes risk, hormone considerations.")
    
    # OVERALL RISK STRATIFICATION
    if risk_score >= 8:
        rules.append("VERY HIGH RISK: Immediate comprehensive diabetes management required.")
    elif risk_score >= 5:
        rules.append("HIGH RISK: Urgent diabetes evaluation and treatment initiation.")
    elif risk_score >= 3:
        rules.append("MODERATE RISK: Regular monitoring and preventive interventions.")
    else:
        rules.append("LOW RISK: Routine preventive care and healthy lifestyle maintenance.")

    # AI MODEL CONFIDENCE INTERPRETATION
    if prediction_confidence > 0.9:
        rules.append(f"Very high model confidence ({prediction_confidence:.2f}): Diagnosis highly reliable.")
    elif prediction_confidence > 0.8:
        rules.append(f"High model confidence ({prediction_confidence:.2f}): Recommendation reliability is high.")
    elif prediction_confidence < 0.6:
        rules.append(f"Low model confidence ({prediction_confidence:.2f}): Additional testing recommended.")
    
    return rules


def bootstrap_accuracy(y_true, y_pred, n_bootstrap=1000):
    """Calculate confidence intervals for accuracy"""
    accuracies = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        acc = accuracy_score(y_true[indices], y_pred[indices])
        accuracies.append(acc)

    ci_lower = np.percentile(accuracies, 2.5)
    ci_upper = np.percentile(accuracies, 97.5)
    return ci_lower, ci_upper

def plot_learning_curves(agent, X, y, title="Learning Curves"):
    """Plot learning curves to detect overfitting"""
    train_sizes, train_scores, val_scores = learning_curve(
        agent.rf_model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    print(f"\n{title}:")
    print(f"Training accuracy: {train_mean[-1]:.3f} (+/- {train_std[-1]*2:.3f})")
    print(f"Validation accuracy: {val_mean[-1]:.3f} (+/- {val_std[-1]*2:.3f})")
    
    # Detect overfitting
    if train_mean[-1] - val_mean[-1] > 0.05:
        print("⚠️  Warning: Potential overfitting detected!")
    else:
        print("✅ Model appears to generalize well")

# =========================================================================
# HYBRID NEURO-SYMBOLIC AI AGENT CLASS
# =========================================================================

class HybridMedicalAgent:
    """
    Advanced AI system that combines:
    1. Neural Network (for complex pattern recognition)
    2. Random Forest (for interpretable feature importance)
    3. Naive Bayes (for probabilistic reasoning)
    4. SHAP (for local explanations)
    5. Symbolic Rules (for clinical reasoning)
    """
    
    def __init__(self, disease_type):
        """
        Initialize the hybrid medical AI system
        
        Args:
            disease_type: Either 'heart' or 'diabetes'
        """
        self.disease_type = disease_type
        
        # Initialize models with default parameters - hyperparameter optimization happens in fit()
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(50, 25, 10),
            alpha=0.001,
            max_iter=1000,
            random_state=42
        )
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
        
        self.nb_model = GaussianNB()
        
        # DATA PREPROCESSING
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # SHAP EXPLAINER
        self.shap_explainer = None     

    def save_models(self, base_path="models"):
        """Save trained models to disk"""
        import joblib
        import os
        
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        os.makedirs(base_path, exist_ok=True)
        
        joblib.dump(self.nn_model, f"{base_path}/{self.disease_type}_neural.pkl")
        joblib.dump(self.rf_model, f"{base_path}/{self.disease_type}_rf.pkl")
        joblib.dump(self.nb_model, f"{base_path}/{self.disease_type}_nb.pkl")
        joblib.dump(self.scaler, f"{base_path}/{self.disease_type}_scaler.pkl")
        joblib.dump(self.feature_names, f"{base_path}/{self.disease_type}_features.pkl")
        
        print(f"✓ {self.disease_type} models saved to {base_path}/")

    def load_models(self, base_path="models"):
        """Load pre-trained models from disk"""
        import joblib
        
        try:
            self.nn_model = joblib.load(f"{base_path}/{self.disease_type}_neural.pkl")
            self.rf_model = joblib.load(f"{base_path}/{self.disease_type}_rf.pkl")
            self.nb_model = joblib.load(f"{base_path}/{self.disease_type}_nb.pkl")
            self.scaler = joblib.load(f"{base_path}/{self.disease_type}_scaler.pkl")
            self.feature_names = joblib.load(f"{base_path}/{self.disease_type}_features.pkl")
            
            # Initialize SHAP explainer if available
            if SHAP_AVAILABLE:
                try:
                    print("  - Initializing SHAP explainer...")
                    # Use TreeExplainer for Random Forest (most reliable)
                    self.shap_explainer = shap.TreeExplainer(self.rf_model)
                    print("  - SHAP TreeExplainer ready!")
                except Exception as e:
                    print(f"  - SHAP initialization failed: {e}")
                    self.shap_explainer = None
            
            self.is_trained = True
            print(f"✓ {self.disease_type} models loaded successfully")
            return True
        except FileNotFoundError:
            print(f"No saved models found for {self.disease_type}")
            return False

    def diagnose_new_patient(self, patient_dict):
        """Diagnose a single new patient from form data"""
        if not self.is_trained:
            raise ValueError("Models must be trained first")
        
        # Convert form data to DataFrame with correct feature order
        df = pd.DataFrame([patient_dict])
        
        # Ensure we have all required features in correct order
        df = df.reindex(columns=self.feature_names, fill_value=0)
        
        predictions, confidences, explanations = self.predict_with_explanation(df)
        return explanations[0]
        
    def fit(self, X, y):
        """Train all models and initialize SHAP explainer with hyperparameter optimization"""
        print(f"Training hybrid models for {self.disease_type} disease...")
        
        # Store feature names
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Scale data for neural network and naive bayes
        X_scaled = self.scaler.fit_transform(X)
        
        # Hyperparameter optimization for Neural Network
        print("  - Optimizing Neural Network hyperparameters...")
        nn_params = {
            'hidden_layer_sizes': [(20,10,5), (50,25,10), (100,50,25)],
            'alpha': [0.0001, 0.001, 0.01]
        }
        nn_grid = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42), nn_params, cv=3, scoring='accuracy')
        nn_grid.fit(X_scaled, y)
        self.nn_model = nn_grid.best_estimator_
        
        # Hyperparameter optimization for Random Forest
        print("  - Optimizing Random Forest hyperparameters...")
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy')
        rf_grid.fit(X, y)
        self.rf_model = rf_grid.best_estimator_
        
        # Train Naive Bayes (no hyperparameter optimization needed)
        print("  - Training Naive Bayes...")
        self.nb_model.fit(X_scaled, y)
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            try:
                print("  - Initializing SHAP explainer...")
                self.shap_explainer = shap.TreeExplainer(self.rf_model)
                print("  - SHAP explainer ready!")
            except Exception as e:
                print(f"  - SHAP initialization failed: {e}")
                self.shap_explainer = None
        
        self.is_trained = True
        print("All models trained successfully!\n")
        return self
    
    def _ensemble_prediction(self, neural_pred, rf_pred, nb_pred, neural_proba, rf_proba, nb_proba):
        """
        Advanced ensemble method combining voting and weighted probabilities
        """
        # Individual model confidences
        neural_conf = np.max(neural_proba)
        rf_conf = np.max(rf_proba)
        nb_conf = np.max(nb_proba)
        
        # Weighted ensemble based on confidence
        total_weight = neural_conf + rf_conf + nb_conf
        if total_weight > 0:
            weighted_proba = (
                neural_conf * neural_proba + 
                rf_conf * rf_proba + 
                nb_conf * nb_proba
            ) / total_weight
        else:
            # Fallback to simple average
            weighted_proba = (neural_proba + rf_proba + nb_proba) / 3
        
        ensemble_pred = np.argmax(weighted_proba)
        ensemble_conf = np.max(weighted_proba)
        
        # Check for model disagreement
        predictions = [neural_pred, rf_pred, nb_pred]
        disagreement = len(set(predictions)) > 1
        
        return ensemble_pred, ensemble_conf, disagreement, {
            'neural': {'pred': neural_pred, 'conf': neural_conf},
            'rf': {'pred': rf_pred, 'conf': rf_conf},
            'nb': {'pred': nb_pred, 'conf': nb_conf}
        }
    
    def predict_with_explanation(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained first! Call .fit() before predicting.")
        
        print("Making predictions with ensemble...")
        X_scaled = self.scaler.transform(X)

        # Predictions from each model
        neural_pred = self.nn_model.predict(X_scaled)
        neural_proba = self.nn_model.predict_proba(X_scaled)
        rf_pred = self.rf_model.predict(X)
        rf_proba = self.rf_model.predict_proba(X)
        nb_pred = self.nb_model.predict(X_scaled)
        nb_proba = self.nb_model.predict_proba(X_scaled)

        final_predictions, final_confidences, explanations = [], [], []

        for i in range(len(X)):
            # Ensemble
            final_pred, final_conf, disagreement, model_details = self._ensemble_prediction(
                neural_pred[i], rf_pred[i], nb_pred[i],
                neural_proba[i], rf_proba[i], nb_proba[i]
            )

            # Compute SHAP for this row
            shap_exp = None
            if self.shap_explainer is not None:
                try:
                    # Get SHAP values for this specific instance - FIXED
                    shap_values = self.shap_explainer.shap_values(X.iloc[i:i+1])
                    
                    # Handle 3D SHAP output (samples, features, classes)
                    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                        # For binary classification, use class 1 SHAP values
                        shap_exp = shap_values[0, :, 1]  # First sample, all features, class 1
                    elif isinstance(shap_values, list) and len(shap_values) == 2:
                        # For binary classification, shap_values is a list of [class_0, class_1]
                        shap_exp = shap_values[1][0]  # Use class 1 SHAP values, first sample
                    else:
                        shap_exp = shap_values
                        
                except Exception as e:
                    shap_exp = f"SHAP explanation failed: {e}"

            explanation = self._generate_comprehensive_explanation(
                X.iloc[i], i, shap_exp, final_pred, final_conf,
                disagreement, model_details
            )
            final_predictions.append(final_pred)
            final_confidences.append(final_conf)
            explanations.append(explanation)

        return np.array(final_predictions), final_confidences, explanations


    def _generate_comprehensive_explanation(
        self, patient_data, patient_idx, shap_values,
        final_pred, final_conf, disagreement, model_details
    ):
        """Generate comprehensive explanation with all analysis types"""

        explanation = {
            'patient_id': f"Patient_{patient_idx + 1:04d}",
            'final_prediction': 'Positive' if final_pred == 1 else 'Negative',
            'confidence': float(final_conf),
            'model_ensemble': {
                'neural_network': {
                    'prediction': 'Positive' if model_details['neural']['pred'] == 1 else 'Negative',
                    'confidence': float(model_details['neural']['conf'])
                },
                'random_forest': {
                    'prediction': 'Positive' if model_details['rf']['pred'] == 1 else 'Negative',
                    'confidence': float(model_details['rf']['conf'])
                },
                'naive_bayes': {
                    'prediction': 'Positive' if model_details['nb']['pred'] == 1 else 'Negative',
                    'confidence': float(model_details['nb']['conf'])
                }
            },
            'disagreement_flag': bool(disagreement),
            'feature_importance': [],
            'shap_explanation': [],
            'symbolic_recommendations': []
        }

        # === Random Forest Feature Importance ===
        rf_importance = self.rf_model.feature_importances_
        top_features = np.argsort(rf_importance)[-5:][::-1]

        for idx in top_features:
            explanation['feature_importance'].append({
                'feature': self.feature_names[idx],
                'value': float(patient_data.iloc[idx]) if hasattr(patient_data, 'iloc') else float(patient_data[idx]),
                'importance': float(rf_importance[idx])
            })

        # === SHAP Explanation  ===
        if shap_values is not None and not isinstance(shap_values, str):
            try:
                # Extract patient data values for easier access
                if hasattr(patient_data, 'iloc'):
                    patient_values = patient_data.values
                else:
                    patient_values = patient_data
                
                # Handle different SHAP value formats
                patient_shap = None
                
                if isinstance(shap_values, np.ndarray):
                    if shap_values.ndim == 3:  # (samples, features, classes)
                        patient_shap = shap_values[0, :, 1]  # First sample, all features, class 1
                    elif shap_values.ndim == 2:  # (samples, features)
                        patient_shap = shap_values[0]  # First sample
                    else:
                        patient_shap = shap_values
                        
                elif isinstance(shap_values, list):
                    # Binary classifier - check if we have class-wise SHAP values
                    if len(shap_values) == 2:  # [class_0, class_1]
                        shap_array = shap_values[1]  # Use class 1 SHAP values
                        if shap_array.ndim == 2:
                            patient_shap = shap_array[0]  # First patient
                        else:
                            patient_shap = shap_array
                    else:
                        # Single array in list
                        patient_shap = shap_values[0] if shap_values else None
                else:
                    explanation['shap_explanation'] = f"Unexpected SHAP format: {type(shap_values)}"
                    # DON'T return here - continue to generate recommendations
                    explanation['shap_explanation'] = f"Unexpected SHAP format: {type(shap_values)}"
                    # Continue to generate recommendations instead of returning

                if patient_shap is None:
                    explanation['shap_explanation'] = "No SHAP values extracted"
                    # Continue to generate recommendations instead of returning

                # Ensure we have the right number of features
                elif len(patient_shap) != len(self.feature_names):
                    explanation['shap_explanation'] = f"SHAP dimension mismatch: {len(patient_shap)} vs {len(self.feature_names)}"
                    # Continue to generate recommendations instead of returning

                else:
                    # Get top 5 features by absolute SHAP value
                    abs_shap = np.abs(patient_shap)
                    top_indices = np.argsort(abs_shap)[-5:][::-1]  # Top 5 indices

                    for idx in top_indices:
                        explanation['shap_explanation'].append({
                            'feature': self.feature_names[idx],
                            'value': float(patient_values[idx]),
                            'shap_value': float(patient_shap[idx]),
                            'impact': 'Increases Risk' if patient_shap[idx] > 0 else 'Decreases Risk'
                        })

            except Exception as e:
                explanation['shap_explanation'] = f"SHAP explanation failed: {str(e)}"
        else:
            explanation['shap_explanation'] = shap_values  # Keep the error message

        # === Symbolic Medical Recommendations ===
        try:
            if self.disease_type == 'heart':
                recommendations = suggest_heart_treatment(patient_data, final_conf)
            elif self.disease_type == 'diabetes':
                recommendations = suggest_diabetes_treatment(patient_data, final_conf)
            else:
                recommendations = ["General medical evaluation recommended"]

            # Add disagreement warning
            if disagreement:
                recommendations.insert(
                    0, "⚠️ MODEL DISAGREEMENT: Models show conflicting predictions - additional clinical evaluation strongly recommended."
                )

            seen = set()
            unique_recs = []
            for rec in recommendations:
                if rec not in seen:
                    unique_recs.append(rec)
                    seen.add(rec)

            explanation['symbolic_recommendations'] = unique_recs

        except Exception as e:
            explanation['symbolic_recommendations'] = [f"Recommendation generation failed: {str(e)}"]

        return explanation

class MedicalDiagnosticService:
    """Persistent service managing trained medical AI agents"""
    
    def __init__(self):
        self.heart_agent = None
        self.diabetes_agent = None
        self.service_ready = False
    
    def initialize_service(self, retrain=False):
        """Initialize the diagnostic service with trained models"""
        print("Initializing Medical Diagnostic Service...")
        
        # Try to load existing models first
        if not retrain:
            if self._load_existing_models():
                print("Service ready with pre-trained models")
                return
        
        # Train new models if loading failed or retrain requested
        print("Training new models (this may take a few minutes)...")
        self._train_new_models()
        print("Service ready with newly trained models")
    
    def _load_existing_models(self):
        """Try to load pre-trained models"""
        self.heart_agent = HybridMedicalAgent('heart')
        self.diabetes_agent = HybridMedicalAgent('diabetes')
        
        heart_loaded = self.heart_agent.load_models()
        diabetes_loaded = self.diabetes_agent.load_models()
        
        if heart_loaded and diabetes_loaded:
            self.service_ready = True
            return True
        return False
    
    def _train_new_models(self):
        """Train and save new models"""
        # Train heart disease model
        X_heart, y_heart = load_and_prepare_heart_data()
        if X_heart is not None:
            self.heart_agent = HybridMedicalAgent('heart')
            self.heart_agent.fit(X_heart, y_heart)
            self.heart_agent.save_models()
        
        # Train diabetes model
        X_diabetes, y_diabetes, _ = load_and_prepare_diabetes_data()
        if X_diabetes is not None:
            self.diabetes_agent = HybridMedicalAgent('diabetes')
            self.diabetes_agent.fit(X_diabetes, y_diabetes)
            self.diabetes_agent.save_models()
        
        self.service_ready = True
    
    def diagnose_patient(self, patient_data, disease_type):
        """Diagnose a patient using the appropriate trained model"""
        if not self.service_ready:
            raise ValueError("Service not initialized. Call initialize_service() first.")
        
        if disease_type == 'heart':
            return self.heart_agent.diagnose_new_patient(patient_data)
        elif disease_type == 'diabetes':
            return self.diabetes_agent.diagnose_new_patient(patient_data)
        else:
            raise ValueError(f"Unknown disease type: {disease_type}")
        
# =========================================================================
# DATA LOADING AND PREPARATION FUNCTIONS
# =========================================================================

def load_and_prepare_heart_data():
    """Load and prepare heart disease data with robust error handling"""
    try:
        df = pd.read_csv("heart.csv")
        print("Heart disease data loaded successfully!")
        
        # Set column names
        df.columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        
        # Clean data
        df.replace("?", pd.NA, inplace=True)
        print(f"Data shape before cleaning: {df.shape}")
        
        df.dropna(inplace=True)
        print(f"Data shape after cleaning: {df.shape}")
        
        df = df.astype(float)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        X = df.drop("target", axis=1)
        y = df["target"]
        
        print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
        
    except FileNotFoundError:
        print("ERROR: heart.csv file not found!")
        return None, None
    except Exception as e:
        print(f"ERROR loading heart data: {str(e)}")
        return None, None

def load_and_prepare_diabetes_data():
    """Load and prepare diabetes data"""
    try:
        df = pd.read_csv("diabetes.csv")


        df = df.sample(n=20000, random_state=42)

        print("Diabetes data loaded successfully!")
        
        df_processed = df.copy()
        
        # Encode categorical variables
        gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
        df_processed['gender'] = df_processed['gender'].map(gender_map)
        
        smoking_map = {
            'never': 0, 'No Info': 1, 'former': 2,
            'not current': 3, 'ever': 4, 'current': 5
        }
        df_processed['smoking_history'] = df_processed['smoking_history'].map(smoking_map)
        
        X = df_processed.drop("diabetes", axis=1)
        y = df_processed["diabetes"]
        
        print(f"Diabetes dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y, df  # Return original for symbolic reasoning
        
    except FileNotFoundError:
        print("ERROR: diabetes.csv file not found!")
        return None, None, None
    except Exception as e:
        print(f"ERROR loading diabetes data: {str(e)}")
        return None, None, None

# =========================================================================
# ANALYSIS PIPELINE FUNCTIONS
# =========================================================================

def analyze_heart_disease(service=None):
    """Heart disease analysis with data export and visualization"""
    print("=" * 60)
    print("HEART DISEASE NEURO-SYMBOLIC ANALYSIS")
    print("=" * 60)
    
    X, y = load_and_prepare_heart_data()
    if X is None or y is None:
        print("Failed to load heart disease data.")
        return

    # USE PRE-TRAINED SERVICE IF AVAILABLE
    if service and service.heart_agent:
        print("Using pre-trained heart disease model...")
        predictions, confidences, explanations = service.heart_agent.predict_with_explanation(X)
        
        # Evaluate performance
        accuracy = accuracy_score(y, predictions)
        ci_lower, ci_upper = bootstrap_accuracy(y, predictions)
        print(f"Accuracy: {accuracy:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        print("\nClassification Report:")
        print(classification_report(y, predictions))
        
        # ✅ STORE ALL VARIABLES FOR EXPORT
        y_export = y
        predictions_export = predictions
        explanations_export = explanations
        agent_export = service.heart_agent
        test_indices_export = list(range(len(X)))
        
    else:
        # Fallback to original CV approach
        print("No pre-trained service available - using CV approach")
        
        cv_scores = []
        all_predictions = []
        all_true_labels = []
        all_explanations = []
        test_indices_list = []

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            agent = HybridMedicalAgent('heart')  
            agent.fit(X_train, y_train)
            predictions, _, explanations = agent.predict_with_explanation(X_test)
            
            fold_accuracy = accuracy_score(y_test, predictions)
            cv_scores.append(fold_accuracy)
            all_predictions.extend(predictions)
            all_true_labels.extend(y_test)
            all_explanations.extend(explanations)
            test_indices_list.extend(test_idx.tolist())

        print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores)*2:.3f})")

        # ✅ STORE ALL VARIABLES FOR EXPORT
        y_export = all_true_labels
        predictions_export = all_predictions
        explanations_export = all_explanations
        agent_export = agent  # Use the last trained agent
        test_indices_export = test_indices_list
        
        accuracy = accuracy_score(y_export, predictions_export)
        ci_lower, ci_upper = bootstrap_accuracy(y_export, predictions_export)
        print(f"Accuracy: {accuracy:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        print("\nClassification Report:")
        print(classification_report(y_export, predictions_export))

        final_agent = HybridMedicalAgent('heart')
        final_agent.fit(X, y)
        plot_learning_curves(final_agent, X, y, "Heart Disease Learning Curves")
    
    # =========================================================================
    # EXPORT ANALYSIS RESULTS (USE CONSISTENT VARIABLES)
    # =========================================================================
    print("\n" + "=" * 40)
    print("EXPORTING RESULTS")
    print("=" * 40)
    
    # Save classification metrics
    save_classification_report(y_export, predictions_export, "heart_classification_report.csv", "heart_disease")
    
    # Save feature importance
    save_feature_importance(agent_export.rf_model, agent_export.feature_names, "heart_feature_importance.csv", "heart_disease")
    
    # Save patient explanations
    save_patient_explanations(explanations_export, test_indices_export, "heart_patient_explanations.csv", "heart_disease")
    
    # Generate visualizations
    plot_feature_importance(agent_export.rf_model, agent_export.feature_names, 
                           "Heart Disease: Random Forest Feature Importance", 
                           "heart_feature_importance.png")
    
    plot_ensemble_agreement(explanations_export, 
                           "Heart Disease: Model Ensemble Agreement", 
                           "heart_model_agreement.png")
    
    # Show detailed analysis
    print("\n" + "=" * 60)
    print("HEART DISEASE PATIENT ANALYSIS")
    print("=" * 60)
    
    disagreement_count = sum([exp['disagreement_flag'] for exp in explanations_export])
    print(f"Cases with model disagreement: {disagreement_count}/{len(explanations_export)}")
    
    # Show first 3 patients with comprehensive analysis
    for i in range(min(3, len(explanations_export))):
        exp = explanations_export[i]
        print(f"\n{'-' * 50}")
        print(f"{exp['patient_id']}: {exp['final_prediction']} (Confidence: {exp['confidence']:.3f})")
        
        if exp['disagreement_flag']:
            print("  ⚠️ MODELS DISAGREE - CLINICAL REVIEW NEEDED!")
        
        print("\nModel Ensemble Results:")
        for model, results in exp['model_ensemble'].items():
            print(f"  • {model.replace('_', ' ').title()}: {results['prediction']} (conf: {results['confidence']:.3f})")
        
        print("\nTop Features (Random Forest Importance):")
        for item in exp['feature_importance']:
            print(f"  • {item['feature']}: {item['value']:.1f} (importance: {item['importance']:.3f})")
        
        if isinstance(exp['shap_explanation'], list) and len(exp['shap_explanation']) > 0:
            print("\nSHAP Local Explanations:")
            for item in exp['shap_explanation']:
                direction = "↑" if item['shap_value'] > 0 else "↓"
                print(f"  • {item['feature']}: {item['value']:.1f} {direction} {item['impact']} (SHAP: {item['shap_value']:+.3f})")
        
        print("\nClinical Recommendations:")
        for rec in exp['symbolic_recommendations'][:4]:
            print(f"  • {rec}")

def bootstrap_accuracy(y_true, y_pred, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for accuracy
    """
    # Convert to NumPy arrays to avoid pandas indexing issues
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n_samples = len(y_true)
    accuracies = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        # Generate bootstrap sample indices
        indices = rng.randint(0, n_samples, n_samples)
        
        # Calculate accuracy for bootstrap sample
        acc = accuracy_score(y_true[indices], y_pred[indices])
        accuracies.append(acc)
    
    # Calculate confidence interval
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(accuracies, 100 * alpha)
    upper = np.percentile(accuracies, 100 * (1 - alpha))
    
    return lower, upper

def analyze_diabetes(service=None):
    """Diabetes analysis with data export and visualization"""
    print("\n" + "=" * 60)
    print("DIABETES NEURO-SYMBOLIC ANALYSIS")
    print("=" * 60)
    
    X, y, df_original = load_and_prepare_diabetes_data()
    if X is None or y is None:
        print("Failed to load diabetes data.")
        return

    # USE PRE-TRAINED SERVICE IF AVAILABLE
    if service and service.diabetes_agent:
        print("Using pre-trained diabetes model...")
        predictions, confidences, explanations = service.diabetes_agent.predict_with_explanation(X)
        
        # Evaluate performance
        accuracy = accuracy_score(y, predictions)
        ci_lower, ci_upper = bootstrap_accuracy(y, predictions)
        print(f"Accuracy: {accuracy:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        print("\nClassification Report:")
        print(classification_report(y, predictions))

        y_export = y
        predictions_export = predictions
        explanations_export = explanations
        agent_export = service.diabetes_agent
        test_indices_export = list(range(len(X)))
        
    else:
        # Fallback to original CV approach
        print("No pre-trained service available - using CV approach")
    
        # Split data
        # Use 5-fold stratified cross-validation
        cv_scores = []
        all_predictions = []
        all_true_labels = []
        all_explanations = []
        test_indices_list = []

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            agent = HybridMedicalAgent('diabetes')  # Fresh agent each fold
            agent.fit(X_train, y_train)
            predictions, _, explanations = agent.predict_with_explanation(X_test)
            
            fold_accuracy = accuracy_score(y_test, predictions)
            cv_scores.append(fold_accuracy)
            all_predictions.extend(predictions)
            all_true_labels.extend(y_test)
            all_explanations.extend(explanations)
            test_indices_list.extend(test_idx.tolist())

        print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores)*2:.3f})")

        # Use final predictions for evaluation
        y_export = all_true_labels
        predictions_export = all_predictions
        explanations_export = all_explanations
        agent_export = agent  # Use the last trained agent
        test_indices_export = test_indices_list
        
        # Evaluate performance
        accuracy = accuracy_score(y_export, predictions_export)
        ci_lower, ci_upper = bootstrap_accuracy(y_export, predictions_export)
        print(f"Accuracy: {accuracy:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        print("\nClassification Report:")
        print(classification_report(y_export, predictions_export))

        # Analyze learning curves for overfitting
        final_agent = HybridMedicalAgent('diabetes')
        final_agent.fit(X, y)
        plot_learning_curves(final_agent, X, y, "Diabetes Learning Curves")

        y_export = all_true_labels
        predictions_export = all_predictions
        explanations_export = all_explanations
        agent_export = final_agent
        test_indices_export = test_indices_list
    
    # =========================================================================
    # EXPORT ANALYSIS RESULTS
    # =========================================================================
    print("\n" + "=" * 40)
    print("EXPORTING RESULTS")
    print("=" * 40)
    
    # Save classification metrics
    save_classification_report(y_export, predictions_export, "diabetes_classification_report.csv", "diabetes")
    
    # Save feature importance
    save_feature_importance(agent_export.rf_model, agent_export.feature_names, "diabetes_feature_importance.csv", "diabetes")
    
    # Save patient explanations
    save_patient_explanations(explanations_export, test_indices_export, "diabetes_patient_explanations.csv", "diabetes")
    
    # Generate visualizations
    plot_feature_importance(agent_export.rf_model, agent_export.feature_names, 
                           "Diabetes: Random Forest Feature Importance", 
                           "diabetes_feature_importance.png")
    
    plot_ensemble_agreement(explanations_export, 
                           "Diabetes: Model Ensemble Agreement", 
                           "diabetes_model_agreement.png")
    
    # Show detailed analysis
    print("\n" + "=" * 60)
    print("DIABETES PATIENT ANALYSIS")
    print("=" * 60)
    
    disagreement_count = sum([exp['disagreement_flag'] for exp in explanations_export])
    print(f"Cases with model disagreement: {disagreement_count}/{len(explanations_export)}")
    
    # Show first 3 patients with comprehensive analysis
    for i in range(min(3, len(explanations_export))):
        exp = explanations_export[i]
        print(f"\n{'-' * 50}")
        print(f"{exp['patient_id']}: {exp['final_prediction']} (Confidence: {exp['confidence']:.3f})")
        
        if exp['disagreement_flag']:
            print("  ⚠️ MODELS DISAGREE - CLINICAL REVIEW NEEDED!")
        
        print("\nModel Ensemble Results:")
        for model, results in exp['model_ensemble'].items():
            print(f"  • {model.replace('_', ' ').title()}: {results['prediction']} (conf: {results['confidence']:.3f})")
        
        print("\nTop Features (Random Forest Importance):")
        for item in exp['feature_importance']:
            print(f"  • {item['feature']}: {item['value']:.1f} (importance: {item['importance']:.3f})")
        
        if isinstance(exp['shap_explanation'], list) and len(exp['shap_explanation']) > 0:
            print("\nSHAP Local Explanations:")
            for item in exp['shap_explanation']:
                direction = "↑" if item['shap_value'] > 0 else "↓"
                print(f"  • {item['feature']}: {item['value']:.1f} {direction} {item['impact']} (SHAP: {item['shap_value']:+.3f})")
        
        print("\nClinical Recommendations:")
        for rec in exp['symbolic_recommendations'][:4]:
            print(f"  • {rec}")

# =========================================================================
# RESULTS SUMMARY FUNCTION
# =========================================================================

def generate_summary():
    """Generate comprehensive summary"""
    print("\n" + "=" * 60)
    print("NEURO-SYMBOLIC AI RESULTS SUMMARY")
    print("=" * 60)
    
    summary_data = {
        'Analysis_Component': [
            'Heart Disease Classification',
            'Diabetes Classification', 
            'Neural Network Ensemble',
            'Random Forest Interpretability',
            'Naive Bayes Probabilistic',
            'SHAP Local Explanations',
            'Symbolic Medical Rules',
            'Model Disagreement Detection'
        ],
        'Status': [
            'Completed', 'Completed', 'Implemented', 'Implemented',
            'Implemented', 'Available' if SHAP_AVAILABLE else 'Not Available',
            'Implemented', 'Implemented'
        ],
        'Output_Files': [
            'heart_*.csv, heart_*.png',
            'diabetes_*.csv, diabetes_*.png',
            'Included in patient explanations',
            'feature_importance.csv files',
            'Included in ensemble results',
            'Included in explanations' if SHAP_AVAILABLE else 'N/A',
            'Included in recommendations',
            'Flagged in patient explanations'
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("analysis_summary.csv", index=False)
    print("✓ Summary saved to analysis_summary.csv")
    
    print("\nGenerated Files:")
    files = [
        "heart_classification_report.csv",
        "heart_feature_importance.csv", 
        "heart_patient_explanations.csv",
        "diabetes_classification_report.csv",
        "diabetes_feature_importance.csv",
        "diabetes_patient_explanations.csv",
        "analysis_summary.csv"
    ]
    
    if PLOTTING_AVAILABLE:
        files.extend([
            "heart_feature_importance.png",
            "heart_model_agreement.png",
            "diabetes_feature_importance.png",
            "diabetes_model_agreement.png"
        ])
    
    for file in files:
        print(f"  • {file}")
    
    print(f"\nTotal exported files: {len(files)}")
    print("\nThese files contain all results, metrics, and visualizations")
    

# =========================================================================
# MAIN EXECUTION PIPELINE
# =========================================================================

if __name__ == "__main__":
    """
    Initialize the persistent medical diagnostic service
    """
    
    # Create the service
    service = MedicalDiagnosticService()
    
    # Initialize (will load existing models or train new ones)
    service.initialize_service(retrain=False)  # Set to True to force retraining
    
    print("\n" + "="*60)
    print("MEDICAL DIAGNOSTIC SERVICE IS READY")
    print("="*60)
    print("The service can now diagnose new patients using trained models.")
    print("Models are persistent and don't need retraining.")

    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE ANALYSIS WITH VISUALIZATIONS")
    print("="*60)
    
    analyze_heart_disease(service)      # This generates all heart visualizations/reports
    analyze_diabetes(service)           # This generates all diabetes visualizations/reports  
    generate_summary()           # This creates the summary file
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - ALL VISUALIZATIONS GENERATED")
    print("="*60)