import os
from pathlib import Path
import numpy as np
import cv2
import joblib
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from skimage.feature import local_binary_pattern
import mediapipe as mp

warnings.filterwarnings('ignore')

# --- CONFIG ---
DATASET_DIR = Path("/Users/seth/Downloads/human_emotion_dataset")
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
LABEL_MAP = {"happy": 0, "sad": 1, "fear": 2, "pain": 3, "anger": 4, "disgust": 5}
RANDOM_STATE = 42
TARGET_SIZE = (224, 224)

# -------------------------------
# 1. FINALIZED FEATURE EXTRACTION
# -------------------------------

def detect_and_crop_face(image, face_cascade):
    if image is None: return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0: return None
    (x, y, w, h) = sorted(faces, key=lambda a: a[2]*a[3], reverse=True)[0]
    return image[y:y+h, x:x+w]

def extract_landmark_features(image):
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks: return None
        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        origin = coords[1]
        coords_normalized = coords - origin
        inter_ocular_dist = np.linalg.norm(coords[130] - coords[359])
        if inter_ocular_dist < 1e-6: return None
        coords_scaled = coords_normalized / inter_ocular_dist
        mouth_opening = np.linalg.norm(coords_scaled[13] - coords_scaled[14])
        mouth_width = np.linalg.norm(coords_scaled[61] - coords_scaled[291])
        return np.concatenate([coords_scaled.flatten(), np.array([mouth_opening, mouth_width])])

def extract_lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def process_image(img_path):
    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    img = cv2.imread(str(img_path))
    cropped_face = detect_and_crop_face(img, face_cascade)
    if cropped_face is None: return None, None
    face_resized = cv2.resize(cropped_face, TARGET_SIZE)
    landmark_features = extract_landmark_features(face_resized)
    lbp_features = extract_lbp_features(face_resized)
    if landmark_features is None or lbp_features is None: return None, None
    combined_features = np.concatenate([landmark_features, lbp_features])
    label = LABEL_MAP.get(img_path.parent.name)
    return combined_features, label

def load_features_parallel(base_dir):
    all_img_paths = list(Path(base_dir).rglob("*.*"))
    print(f"Found {len(all_img_paths)} images in {base_dir}. Cropping faces and extracting features...")
    results = joblib.Parallel(n_jobs=-1, verbose=1)(
        joblib.delayed(process_image)(img_path) for img_path in all_img_paths
    )
    valid_results = [res for res in results if res[0] is not None and res[1] is not None]
    if not valid_results: return np.array([]), np.array([])
    X, y = zip(*valid_results)
    return np.array(X), np.array(y)

# -------------------------------
# 2. MODEL TRAINING
# -------------------------------
def train_stacking_model(X_train, y_train, X_test, y_test):
    estimators = [
        ('lgbm', lgb.LGBMClassifier(
            objective='multiclass', random_state=RANDOM_STATE,
            reg_lambda=0.1, reg_alpha=0.5, num_leaves=31,
            n_estimators=600, max_depth=7, learning_rate=0.1,
            colsample_bytree=0.9,
            verbose=-1  # Suppress warnings
        )),
        ('rf', RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, n_estimators=300,
            min_samples_split=5, min_samples_leaf=1,
            max_features='sqrt', max_depth=10
        )),
        ('svc', SVC(probability=True, random_state=RANDOM_STATE,
                   kernel='rbf', gamma=0.001, C=30))
    ]

    for name, model in estimators:
        print(f"\nTraining {name.upper()} model...")
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"🎯 {name.upper()} ACCURACY: {acc:.4f}")

    meta_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    stacking_classifier = StackingClassifier(
        estimators=estimators, final_estimator=meta_model, cv=5
    )

    final_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', stacking_classifier)
    ])

    print("\nTraining the ultimate STACKING pipeline on CROPPED faces...")
    final_pipeline.fit(X_train, y_train)
    return final_pipeline


def main():
    X_train, y_train = load_features_parallel(TRAIN_DIR)
    X_test, y_test = load_features_parallel(TEST_DIR)

    if len(X_train) == 0:
        print("❌ No features extracted. Check face detection or dataset paths. Exiting.")
        return

    model_pipeline = train_stacking_model(X_train, y_train, X_test, y_test)

    print("\nEvaluating final STACKING model on the test set...")
    y_pred = model_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 FINAL STACKING ACCURACY: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_MAP.keys(), zero_division=0))


if __name__ == "__main__":
    main()