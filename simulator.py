# new_g_simulator.py
import sys
import time
import random
from datetime import datetime

try:
    from neuro_symbolic_xai_agent import MedicalDiagnosticService
    ML_SYSTEM_AVAILABLE = True
except ImportError:
    ML_SYSTEM_AVAILABLE = False
    print("WARNING:  ML system not available. Run 'python _medical_ai.py' first.")

# =========================================================================
# MEDICAL AI SIMULATOR
# =========================================================================

class MedicalAISystem:
    def __init__(self):
        self.session_id = f"SESSION_{random.randint(1000, 9999)}"
        self.patient_counter = 1
        self.history = []
        if ML_SYSTEM_AVAILABLE:
            self.service = MedicalDiagnosticService()
            self.service.initialize_service(retrain=False)
        else:
            self.service = None

    def run(self):
        if not ML_SYSTEM_AVAILABLE or not self.service.service_ready:
            print("❌ ERROR: Cannot start simulator without trained ML models.")
            print("Please run 'python medical_ai.py' first to train models.")
            sys.exit(1)

        self._startup_banner()
        try:
            while True:
                self._show_menu()
                choice = input("\n🎯 Select option (1-3, q=quit): ").strip().lower()
                if choice == '1':
                    self._workflow("diabetes")
                elif choice == '2':
                    self._workflow("heart")
                elif choice == '3':
                    self._show_history()
                elif choice in ['q', 'quit', 'exit']:
                    self._exit_summary()
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")

    def _workflow(self, disease):
        print(f"\n🧾 Starting {disease.title()} diagnosis...")
        patient_data = self._collect_patient_data(disease)
        patient_data = self._encode_patient_data(patient_data, disease)

        print("🔄 Running patient data through trained neuro-symbolic AI...")
        result = self.service.diagnose_patient(patient_data, disease)  

        print("✅ Diagnosis complete.\n")
        self._display_result(result)

        self.history.append(result)
        self.patient_counter += 1

    def _display_result(self, result: dict):
        print("\n" + "=" * 70)
        print("🏥 CLINICAL ANALYSIS REPORT")
        print("=" * 70)

        print(f"👤 Patient ID: {result['patient_id']}")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🧾 Final Diagnosis: {result['final_prediction']} ({result['confidence']:.1%})")
        print(f"⚠️  Model Disagreement: {'Yes' if result['disagreement_flag'] else 'No'}")

        # Ensemble details
        print("\n🤖 MODEL ENSEMBLE RESULTS")
        for model, stats in result['model_ensemble'].items():
            print(f" • {model.replace('_', ' ').title():15}: {stats['prediction']} ({stats['confidence']:.1%})")

        # Feature importance
        print("\n📊 KEY FEATURES (Random Forest Importance)")
        for feat in result['feature_importance']:
            print(f" • {feat['feature']:20}: {feat['value']} (importance {feat['importance']:.3f})")

        # SHAP explanations
        if isinstance(result['shap_explanation'], list) and result['shap_explanation']:
            print("\n🔍 SHAP LOCAL EXPLANATIONS (per-patient)")
            for shap in result['shap_explanation']:
                direction = "↑ Increases Risk" if shap['shap_value'] > 0 else "↓ Decreases Risk"
                print(f" • {shap['feature']:20}: {shap['value']} | {direction} ({shap['shap_value']:+.3f})")
        else:
            print(f"\n🔍 SHAP Explanation: {result['shap_explanation']}")

        # Recommendations
        print("\n💡 CLINICAL RECOMMENDATIONS")
        for rec in result['symbolic_recommendations']:
            if any(tag in rec for tag in ["VERY HIGH", "URGENT", "⚠️"]):
                print(f" 🚨 {rec}")
            elif any(tag in rec for tag in ["HIGH", "Important"]):
                print(f" 🟡 {rec}")
            else:
                print(f" 🟢 {rec}")

        print("\n" + "=" * 70 + "\n")


    def _encode_patient_data(self, data, disease):
        if disease == "diabetes":
            gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
            smoking_map = {
                'never': 0, 'No Info': 1, 'former': 2,
                'not current': 3, 'ever': 4, 'current': 5
            }
            data["gender"] = gender_map.get(data["gender"], 0)
            data["smoking_history"] = smoking_map.get(data["smoking_history"], 0)
        return data


    def _collect_patient_data(self, disease):
        print("\n📝 Please enter patient details:")

        if disease == "diabetes":
            age = int(input("Age: "))
            bmi = float(input("BMI (kg/m², normal 18.5–24.9): "))
            hba1c = float(input("HbA1c level(%) of glycated hemoglobin, normal <5.7): "))
            glucose = float(input("Blood Glucose level (mg/dL, fasting 70–99 normal): "))
            hypertension = int(input("Hypertension (1=yes, 0=no): "))
            heart_disease = int(input("Heart disease (1=yes, 0=no): "))
            gender = input("Gender (Male/Female/Other): ").strip()
            smoking = input("Smoking history (never/former/current/ever/not current/No Info): ").strip()

            return {
                "age": age,
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "gender": gender,
                "smoking_history": smoking
            }

        elif disease == "heart":
            age = int(input("Age: "))
            sex = int(input("Sex (1=male, 0=female): "))
            cp = int(input("Chest pain type (0=asymptomatic, 1=atypical angina, 2=non-anginal pain, 3=typical angina): "))
            trestbps = int(input("Resting blood pressure(mmHg, normal ~120): "))
            chol = int(input("Cholesterol level(mg/dL, normal <200): "))
            fbs = int(input("Fasting blood sugar >120 (1=yes,0=no): "))
            restecg = int(input("Resting ECG result (0=normal, 1=ST-T abnormality, 2=LV hypertrophy): "))
            thalach = int(input("Max heart rate achieved(bpm, normal 150–200): "))
            exang = int(input("Exercise induced angina (1=yes,0=no): "))
            oldpeak = float(input("ST depression induced by exercise(1=yes,0=no): "))
            slope = int(input("Slope of ST segment (0=upsloping, 1=flat, 2=downsloping): "))
            ca = int(input("Number of major vessels (0–3): "))
            thal = int(input("Thalassemia (1=normal, 2=fixed defect, 3=reversible defect): "))

            return {
                "age": age, "sex": sex, "cp": cp,
                "trestbps": trestbps, "chol": chol, "fbs": fbs,
                "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }


    def _show_menu(self):
        print("\n📋 MAIN MENU")
        print("1. 🩺 Diagnose Diabetes")
        print("2. ❤️ Diagnose Heart Disease")
        print("3. 📜 View History")
        print("Q. 🚪 Quit")

    def _show_history(self):
        if not self.history:
            print("\n📜 No patient analyses in this session yet.")
            return

        print(f"\n📜 Session {self.session_id} Analysis History")
        for i, result in enumerate(self.history, 1):
            print(f"{i}. {result['patient_id']} | {result.get('disease_type','?').title()} | "
                f"{result['final_prediction']} ({result['confidence']:.1%})")

    def _startup_banner(self):
        print("=" * 70)
        print("🏥 NEURO-SYMBOLIC MEDICAL AI - SIMULATOR")
        print("=" * 70)
        print(f"Session: {self.session_id} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("System Ready with Trained Models")

    def _exit_summary(self):
        print("\n" + "=" * 60)
        print("👋 SESSION SUMMARY")
        print("=" * 60)
        print(f"Session ID: {self.session_id}")
        print(f"Total Patients Analyzed: {self.patient_counter - 1}")
        print("Thank you for using the Medical AI System!")

# =========================================================================
# ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    print("🚀 INITIALIZING MEDICAL AI SIMULATOR...")
    app = MedicalAISystem()
    app.run()
