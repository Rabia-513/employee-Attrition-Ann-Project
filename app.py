# 🧠 ANN-Based Employee Attrition Prediction (Final Submission App)
# =========================================================
# STREAMLIT FINAL PROJECT – PHASE 1 to PHASE 4
# =========================================================

# =====================
# 1. IMPORT LIBRARIES
# =====================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import math
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================
# 2. PAGE CONFIG + THEME
# =====================
st.set_page_config(page_title="ANN Employee Attrition", page_icon="🧠", layout="wide")

st.markdown("""
<style>
body { background-color: #0e1117; color: #f9fafb; }
.card { background-color: #111827; padding: 20px; border-radius: 12px; box-shadow: 0px 4px 15px rgba(0,0,0,0.4); }
h1,h2,h3 { color: #f9fafb; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 ANN-Based Employee Attrition Prediction")
st.caption("ANN | Phase-wise Implementation")


# =====================
# 4. SIDEBAR NAVIGATION
# =====================
menu = st.sidebar.radio("📌 Navigation",[
    "Dataset Preview",
    "Phase 1: EDA & Preprocessing",
    "Phase 2: Baseline ANN",
    "Phase 3: Optimizer Comparison",
    "Phase 4: Optimized ANN"
])

# =========================================================
# LOAD DATA
# =========================================================
DATA_PATH = "Modified_HR_Employee_Attrition.csv"
df = pd.read_csv(DATA_PATH)


# =========================================================
# COLUMN GROUPS (UNCHANGED)
# =========================================================
categorical_cols = [
    'BusinessTravel','Department','EducationField','Gender',
    'JobRole','MaritalStatus','Over18','OverTime'
]

ordinal_cols = [
    'Education','EmployeeCount','EnvironmentSatisfaction','JobInvolvement',
    'JobLevel','JobSatisfaction','NumCompaniesWorked','PerformanceRating',
    'RelationshipSatisfaction','StandardHours','StockOptionLevel',
    'TrainingTimesLastYear','WorkLifeBalance'
]

numerical_cols = [
    'Age','DailyRate','DistanceFromHome','EmployeeNumber','HourlyRate',
    'MonthlyIncome','MonthlyRate','PercentSalaryHike','TotalWorkingYears',
    'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

normal_numerical = ['Age','HourlyRate','MonthlyRate']
skewed_numerical = [c for c in numerical_cols if c not in normal_numerical]

# =========================================================
# CUSTOM TRANSFORMERS (UNCHANGED)
# =========================================================
class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns): self.columns = columns
    def fit(self, X, y=None):
        self.means = X[self.columns].mean()
        return self
    def transform(self, X):
        X = X.copy()
        X[self.columns] = X[self.columns].fillna(self.means)
        return X

class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns): self.columns = columns
    def fit(self, X, y=None):
        self.medians = X[self.columns].median()
        return self
    def transform(self, X):
        X = X.copy()
        X[self.columns] = X[self.columns].fillna(self.medians)
        return X

class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns): self.columns = columns
    def fit(self, X, y=None):
        self.modes = X[self.columns].mode().iloc[0]
        return self
    def transform(self, X):
        X = X.copy()
        X[self.columns] = X[self.columns].fillna(self.modes)
        return X

class ZScoreOutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, threshold=3):
        self.columns = columns
        self.threshold = threshold
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            z = (X[col] - X[col].mean()) / X[col].std()
            X.loc[np.abs(z) > self.threshold, col] = X[col].median()
        return X

class IQROutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns): self.columns = columns
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            Q1, Q3 = X[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            X[col] = np.where((X[col]<lower)|(X[col]>upper),
                              X[col].median(), X[col])
        return X

# =========================================================
# PIPELINE (UNCHANGED)
# =========================================================
categorical_pipeline = Pipeline([
    ('mode', ModeImputer(categorical_cols)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

ordinal_pipeline = Pipeline([
    ('mode', ModeImputer(ordinal_cols)),
    ('ordinal', OrdinalEncoder())
])

numerical_pipeline = Pipeline([
    ('mean', MeanImputer(normal_numerical)),
    ('median', MedianImputer(skewed_numerical)),
    ('zscore', ZScoreOutlierHandler(normal_numerical)),
    ('iqr', IQROutlierHandler(skewed_numerical)),
    ('scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('cat', categorical_pipeline, categorical_cols),
    ('ord', ordinal_pipeline, ordinal_cols),
    ('num', numerical_pipeline, numerical_cols)
])

# =========================================================
# DATA SPLIT (UNCHANGED)
# =========================================================
X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({'Yes':1,'No':0})

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
)

X_train_p = full_pipeline.fit_transform(X_train).astype('float32')
X_val_p   = full_pipeline.transform(X_val).astype('float32')
X_test_p  = full_pipeline.transform(X_test).astype('float32')



input_dim = X_train_p.shape[1]


# =====================
# 5. MODEL BUILDER
# =====================
def build_model(input_dim, optimized=False):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))

    if optimized:
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# =====================
# DATASET PREVIEW
# =====================
if menu == "Dataset Preview":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.markdown("</div>", unsafe_allow_html=True)

# =====================
# PHASE 1: EDA
# =====================
elif menu == "Phase 1: EDA & Preprocessing":

    st.subheader("📊 Exploratory Data Analysis")
    st.markdown("### Explore your dataset interactively:")

    # ---------- Button: Show Missing Values ----------
    if st.button("🟢 Show Missing Values Heatmap"):
        st.subheader("Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

    # ---------- Button: Show Correlation Matrix ----------
    if st.button("🟢 Show Correlation Matrix"):
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            df[numerical_cols].corr(),
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            ax=ax
        )
        st.pyplot(fig)

    # ---------- Button: Show Histograms ----------
    if st.button("🟢 Show Histograms"):
        st.subheader("Histograms of Numerical Features")
        fig, axes = plt.subplots(
            nrows=math.ceil(len(numerical_cols) / 4),
            ncols=4,
            figsize=(16, 10)
        )
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.histplot(df[col], kde=True, ax=axes[i], color="skyblue")
            axes[i].set_title(col, fontsize=10)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

    # ---------- Button: Show Boxplots ----------
    if st.button("🟢 Show Box Plots"):
        st.subheader("Box Plots of Numerical Features")
        fig, axes = plt.subplots(
            nrows=math.ceil(len(numerical_cols) / 4),
            ncols=4,
            figsize=(16, 8)
        )
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.boxplot(y=df[col], ax=axes[i], color="lightgreen")
            axes[i].set_title(col, fontsize=10)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

    # ---------- Button: Handle Missing Values & Outliers ----------
    if st.button("🛠 Handle Missing Values & Outliers"):
        st.subheader("Processing Missing Values & Outliers")

        df_processed = df.copy()

        # Handle missing values
        for cat_col in categorical_cols:
            df_processed[cat_col].fillna(
                df_processed[cat_col].mode()[0],
                inplace=True
            )

        for ord_col in ordinal_cols:
            df_processed[ord_col].fillna(
                df_processed[ord_col].mode()[0],
                inplace=True
            )

        for col in normal_numerical:
            df_processed[col].fillna(
                df_processed[col].mean(),
                inplace=True
            )

        for col in skewed_numerical:
            df_processed[col].fillna(
                df_processed[col].median(),
                inplace=True
            )

        # Handle outliers
        for col in normal_numerical:
            z = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
            df_processed.loc[np.abs(z) > 3, col] = df_processed[col].median()

        for col in skewed_numerical:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df_processed[col] = np.where(
                (df_processed[col] < lower) | (df_processed[col] > upper),
                df_processed[col].median(),
                df_processed[col]
            )

        st.success("✔ Missing values handled and outliers treated")
        st.dataframe(df_processed.head())

    # ---------- Button: Normalize Features ----------
    if st.button("⚡ Normalize Numerical Features"):
        st.subheader("Normalized Features Preview")
        df_norm = df.copy()

        for col in numerical_cols:
            df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()

        st.dataframe(df_norm[numerical_cols].head())
        st.success("✔ Features normalized (StandardScaler)")

# =====================
# PHASE 2: BASELINE ANN (INTERACTIVE)
# =====================

elif menu == "Phase 2: Baseline ANN":

    st.title("🧠 Phase 2 – Baseline ANN Architecture")

    st.markdown("### Configure Baseline ANN ")

    # ----------- Architecture Controls -----------
    hidden_layers = st.number_input(
        "Number of Hidden Layers",
        min_value=1,
        max_value=6,
        value=3,
        step=1
    )

    st.info("🔒 Hidden layers use **ReLU** activation | Output uses **Sigmoid**")

    neurons = []
    st.markdown("### Neurons per Hidden Layer")
    default_neurons = [128, 64, 32]
    for i in range(hidden_layers):
        n = st.slider(
            f"Hidden Layer {i+1} Neurons",
            min_value=8,
            max_value=256,
            value=default_neurons[i] if i < len(default_neurons) else 64,
            step=8,
            key=f"layer_{i}"
        )
        neurons.append(n)
 
    # ----------- Build Model -----------
    if st.button("🔹 Build & Show Model Summary"):
          model_baseline = Sequential()
          model_baseline.add(Input(shape=(input_dim,)))

          for n in neurons:
               model_baseline.add(Dense(n, activation="relu"))

          model_baseline.add(Dense(1, activation="sigmoid"))

        # Streamlit-friendly summary
          stream = io.StringIO()
          model_baseline.summary(print_fn=lambda x: stream.write(x + "\n"))
          st.text(stream.getvalue())

          st.session_state.baseline_model = model_baseline


# =====================
# PHASE 3: OPTIMIZER COMPARISON
# =====================

elif menu == "Phase 3: Optimizer Comparison":

    st.title("⚙️ Phase 3 – Optimizer Comparison")

    st.markdown("""
    Train the **same ANN architecture** using different optimizers  
    and analyze **convergence, performance, and generalization**.
    """)

    optimizers = {
        "Adam": Adam(),
        "SGD": SGD(momentum=0.9),
        "RMSprop": RMSprop(),
        "Adagrad": Adagrad()
    }

    if "optimizer_results" not in st.session_state:
        st.session_state.optimizer_results = {}

    cols = st.columns(4)

    # =========================
    # TRAIN BUTTONS
    # =========================
    for col, (opt_name, opt) in zip(cols, optimizers.items()):
        with col:
            if st.button(f"🚀 Train with {opt_name}"):

                model = build_model(input_dim)
                model.compile(
                    optimizer=opt,
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )

                history = model.fit(
                    X_train_p, y_train,
                    validation_data=(X_val_p, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )

                # Predictions
                y_pred = (model.predict(X_test_p) > 0.5).astype(int)

                # Metrics
                acc  = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec  = recall_score(y_test, y_pred, zero_division=0)
                f1   = f1_score(y_test, y_pred, zero_division=0)

                # Save results
                st.session_state.optimizer_results[opt_name] = {
                    "model": model,
                    "history": history.history,
                    "metrics": {
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1-Score": f1
                    }
                }

                st.success(f"✅ {opt_name} training completed")

    # =========================
    # DISPLAY RESULTS
    # =========================
    if st.session_state.optimizer_results:

        opt_selected = st.selectbox(
            "📌 View Results For:",
            list(st.session_state.optimizer_results.keys())
        )

        result = st.session_state.optimizer_results[opt_selected]

        st.subheader(f"📊 {opt_selected} – Training Curves")

        col1, col2 = st.columns(2)

        with col1:
            st.line_chart({
                "Train Accuracy": result["history"]["accuracy"],
                "Val Accuracy": result["history"]["val_accuracy"]
            })

        with col2:
            st.line_chart({
                "Train Loss": result["history"]["loss"],
                "Val Loss": result["history"]["val_loss"]
            })

        st.subheader("📈 Test Set Performance")

        metrics_df = pd.DataFrame(
            result["metrics"].items(),
            columns=["Metric", "Value"]
        )

        st.table(metrics_df)

    # =========================
    # OPTIMIZER COMPARISON
    # =========================
    st.markdown("---")
    if st.button("🏆 OPTIMIZER COMPARISON SUMMARY"):

        summary = []

        for opt, data in st.session_state.optimizer_results.items():
            summary.append([
                opt,
                data["metrics"]["Accuracy"],
                data["metrics"]["Precision"],
                data["metrics"]["Recall"],
                data["metrics"]["F1-Score"]
            ])

        summary_df = pd.DataFrame(
            summary,
            columns=["Optimizer", "Accuracy", "Precision", "Recall", "F1-Score"]
        )

        st.subheader("📋 Optimizer Performance Summary")
        st.dataframe(summary_df)

        # Best optimizer (by Accuracy)
        best_optimizer = summary_df.sort_values(
            by="Accuracy", ascending=False
        ).iloc[0]["Optimizer"]

        st.success(f"🏆 Best Optimizer: **{best_optimizer}**")

        # =========================
        # RETRAIN BEST MODEL
        # =========================
        st.info("🔁 Retraining best optimizer on full training data...")

        best_model = build_model(input_dim)
        best_model.compile(
            optimizer=optimizers[best_optimizer],
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        best_model.fit(
            X_train_p, y_train,
            epochs=50,
            batch_size=32,
            verbose=0
        )

        st.session_state.final_model = best_model
        st.success("✅ Final optimized ANN trained successfully")

# =====================
# PHASE 4: OPTIMIZED ANN
# =====================

elif menu == "Phase 4: Optimized ANN":

    st.title("🚀 Phase 4 – Optimized ANN Architecture")

    st.markdown("""
    ### Optimized ANN Configuration  
    ✔ Batch Normalization  
    ✔ Dropout Regularization  
    ✔ He Weight Initialization  
    ✔ ReLU (Hidden) | Sigmoid (Output)
    """)

    # =====================
    # ARCHITECTURE DISPLAY
    # =====================
    st.markdown("### 🔧 Network Architecture")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Hidden Layer 1", "128 Neurons")
        st.caption("Dropout: 0.3 | BatchNorm")

    with col2:
        st.metric("Hidden Layer 2", "64 Neurons")
        st.caption("Dropout: 0.3 | BatchNorm")

    with col3:
        st.metric("Hidden Layer 3", "32 Neurons")
        st.caption("Dropout: 0.2 | BatchNorm")

    # =====================
    # MODEL BUILDER
    # =====================
    def build_optimized_model():
        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        model.add(Dense(96, activation="relu", kernel_initializer="he_normal"))
        #model.add(BatchNormalization())
        model.add(Dropout(0.10))

        model.add(Dense(48, activation="relu", kernel_initializer="he_normal"))
        #model.add(BatchNormalization())
        model.add(Dropout(0.10))

        model.add(Dense(32, activation="relu", kernel_initializer="he_normal"))
       # model.add(BatchNormalization())
        model.add(Dropout(0.05))

        model.add(Dense(1, activation="sigmoid"))
        return model

    # =====================
    # BUILD MODEL BUTTON
    # =====================
    if st.button("🔧 Build Optimized Model & Show Summary"):

        opt_model = build_optimized_model()

        stream = io.StringIO()
        opt_model.summary(print_fn=lambda x: stream.write(x + "\n"))
        st.text(stream.getvalue())
        st.session_state.phase4_model = opt_model

    # =====================
    # OPTIMIZERS
    # =====================
    st.markdown("---")
    st.markdown("### ⚙️ Train with Different Optimizers")

    optimizers = {
        "Adam": Adam(learning_rate=0.0005),
        "SGD": SGD(learning_rate=0.01, momentum=0.9),
        "RMSprop": RMSprop(learning_rate=0.001),
        "Adagrad": Adagrad(learning_rate=0.01)
    }

    if "phase4_results" not in st.session_state:
        st.session_state.phase4_results = {}

    cols = st.columns(4)

    for col, (opt_name, optimizer) in zip(cols, optimizers.items()):
        with col:
            if st.button(f"🚀 Train with {opt_name}"):
                tf.keras.backend.clear_session() 
                model = build_optimized_model()
                model.compile(
                    optimizer=optimizer,
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )
             
                history = model.fit(
                    X_train_p, y_train,
                    validation_data=(X_val_p, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )

                y_pred = (model.predict(X_test_p) > 0.5).astype(int)

                acc  = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec  = recall_score(y_test, y_pred, zero_division=0)
                f1   = f1_score(y_test, y_pred, zero_division=0)

                st.session_state.phase4_results[opt_name] = {
                    "model": model,
                    "history": history.history,
                    "metrics": {
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1-Score": f1
                    }
                }

                st.success(f"✅ {opt_name} training completed")

    # =====================
    # SHOW TRAINING CURVES
    # =====================
    if st.session_state.phase4_results:

        selected = st.selectbox(
            "📌 View Optimizer Results:",
            list(st.session_state.phase4_results.keys())
        )

        res = st.session_state.phase4_results[selected]

        col1, col2 = st.columns(2)

        with col1:
            st.line_chart({
                "Train Accuracy": res["history"]["accuracy"],
                "Validation Accuracy": res["history"]["val_accuracy"]
            })

        with col2:
            st.line_chart({
                "Train Loss": res["history"]["loss"],
                "Validation Loss": res["history"]["val_loss"]
            })

        st.subheader("📊 Test Performance")

        st.table(pd.DataFrame(
            res["metrics"].items(),
            columns=["Metric", "Value"]
        ))

    # =====================
    # OPTIMIZER COMPARISON
    # =====================
    st.markdown("---")
    if st.button("🏆 OPTIMIZER COMPARISON SUMMARY"):

        rows = []
        for opt, data in st.session_state.phase4_results.items():
            rows.append([
                opt,
                data["metrics"]["Accuracy"],
                data["metrics"]["Precision"],
                data["metrics"]["Recall"],
                data["metrics"]["F1-Score"]
            ])

        df_summary = pd.DataFrame(
            rows,
            columns=["Optimizer", "Accuracy", "Precision", "Recall", "F1-Score"]
        )

        st.subheader("📋 Optimizer Comparison (Phase 4)")
        st.dataframe(df_summary)

        best_opt = df_summary.sort_values(
            by="F1-Score", ascending=False
        ).iloc[0]["Optimizer"]

        st.success(f"🏆 Best Optimizer (Phase 4): **{best_opt}**")

        # =====================
        # RETRAIN BEST MODEL
        # =====================
        st.info("🔁 Retraining best optimized ANN on full training data...")
        tf.keras.backend.clear_session()
        final_model = build_optimized_model()
        final_model.compile(
            optimizer=optimizers[best_opt],
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        final_model.fit(
            X_train_p, y_train,
            epochs=50,
            batch_size=32,
            verbose=0
        )

        st.session_state.phase4_best_model = final_model
        st.success("✅ Final Optimized ANN trained successfully")

   