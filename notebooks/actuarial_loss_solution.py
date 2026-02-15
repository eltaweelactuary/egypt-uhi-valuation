# %% [markdown]
# # 🏆 حل مسابقة Actuarial Loss Estimation
# ## التنبؤ بتكاليف مطالبات تعويضات العمال
# 
# هذا الكود يحل مسابقة Kaggle للتنبؤ بتكلفة المطالبات التأمينية النهائية

# %% [markdown]
# ## 📦 الخطوة 1: تثبيت وإستيراد المكتبات

# %%
# تثبيت المكتبات المطلوبة
!pip install -q kaggle xgboost lightgbm catboost

# %%
# إستيراد المكتبات الأساسية
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# مكتبات النماذج
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

print("✅ تم تحميل جميع المكتبات بنجاح!")

# %% [markdown]
# ## 📥 الخطوة 2: تحميل البيانات من Kaggle

# %%
# إعداد Kaggle API
# ارفع ملف kaggle.json الخاص بك
from google.colab import files
print("📤 ارفع ملف kaggle.json:")
# files.upload()  # قم بإلغاء التعليق لرفع الملف

# %%
# إعداد مجلد Kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/ 2>/dev/null || echo "تأكد من رفع kaggle.json"
!chmod 600 ~/.kaggle/kaggle.json

# تحميل بيانات المسابقة
!kaggle competitions download -c actuarial-loss-estimation
!unzip -q actuarial-loss-estimation.zip -d data/

print("✅ تم تحميل البيانات!")

# %% [markdown]
# ## 📊 الخطوة 3: استكشاف البيانات (EDA)

# %%
# تحميل البيانات
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"📈 حجم بيانات التدريب: {train.shape}")
print(f"📉 حجم بيانات الاختبار: {test.shape}")
print(f"\n🎯 المتغير المستهدف: UltimateIncurredClaimCost")

# %%
# عرض أول 5 صفوف
print("📋 عينة من البيانات:")
train.head()

# %%
# معلومات عن الأعمدة
print("📊 معلومات الأعمدة:")
train.info()

# %%
# إحصائيات وصفية
print("📈 الإحصائيات الوصفية:")
train.describe()

# %%
# توزيع المتغير المستهدف
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train['UltimateIncurredClaimCost'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('توزيع تكلفة المطالبات', fontsize=14)
axes[0].set_xlabel('التكلفة')

axes[1].hist(np.log1p(train['UltimateIncurredClaimCost']), bins=50, color='coral', edgecolor='white')
axes[1].set_title('توزيع التكلفة (بعد Log Transform)', fontsize=14)
axes[1].set_xlabel('log(التكلفة + 1)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🔧 الخطوة 4: هندسة الميزات (Feature Engineering)

# %%
def create_features(df, is_train=True):
    """
    إنشاء ميزات جديدة من البيانات
    """
    df = df.copy()
    
    # 1. تحويل التواريخ
    date_cols = ['DateTimeOfAccident', 'DateReported']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_Year'] = df[col].dt.year
            df[f'{col}_Month'] = df[col].dt.month
            df[f'{col}_DayOfWeek'] = df[col].dt.dayofweek
    
    # 2. تأخير الإبلاغ (بالأيام)
    if 'DateReported' in df.columns and 'DateTimeOfAccident' in df.columns:
        df['ReportingDelay'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.days
    
    # 3. ميزات الراتب
    if 'WeeklyWages' in df.columns:
        df['WeeklyWages_Log'] = np.log1p(df['WeeklyWages'])
        df['AnnualWages'] = df['WeeklyWages'] * 52
    
    # 4. ميزات ساعات العمل
    if 'HoursWorkedPerWeek' in df.columns:
        df['IsPartTime'] = (df['HoursWorkedPerWeek'] < 35).astype(int)
        if 'WeeklyWages' in df.columns:
            df['HourlyWage'] = df['WeeklyWages'] / df['HoursWorkedPerWeek'].replace(0, 1)
    
    # 5. ميزات من النص (وصف الحادث)
    text_col = None
    for col in ['ClaimDescription', 'AccidentDescription']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col:
        df[text_col] = df[text_col].fillna('')
        df['TextLength'] = df[text_col].apply(len)
        df['WordCount'] = df[text_col].apply(lambda x: len(str(x).split()))
        
        # كلمات تدل على شدة الإصابة
        severity_words = ['severe', 'serious', 'fracture', 'surgery', 'hospital', 'permanent']
        df['SeverityScore'] = df[text_col].apply(
            lambda x: sum(1 for w in severity_words if w.lower() in str(x).lower())
        )
    
    return df

# تطبيق هندسة الميزات
train = create_features(train, is_train=True)
test = create_features(test, is_train=False)

print("✅ تم إنشاء الميزات الجديدة!")
print(f"📊 عدد الأعمدة بعد هندسة الميزات: {train.shape[1]}")

# %% [markdown]
# ## 🏷️ الخطوة 5: تحضير البيانات للنماذج

# %%
# تحديد المتغير المستهدف
TARGET = 'UltimateIncurredClaimCost'
y = train[TARGET]
y_log = np.log1p(y)  # تحويل لوغاريتمي للتوزيع

# الأعمدة المستبعدة
EXCLUDE_COLS = [TARGET, 'ClaimNumber', 'ClaimDescription', 'AccidentDescription',
                'DateTimeOfAccident', 'DateReported', 'DateOfBirth']

# تحديد الميزات
feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
print(f"📋 عدد الميزات: {len(feature_cols)}")

# %%
# ترميز المتغيرات الفئوية
label_encoders = {}
categorical_cols = train[feature_cols].select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = train[col].fillna('MISSING').astype(str)
    test[col] = test[col].fillna('MISSING').astype(str)
    
    # دمج القيم للترميز الموحد
    all_values = pd.concat([train[col], test[col]]).unique()
    le.fit(all_values)
    
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

print(f"✅ تم ترميز {len(categorical_cols)} عمود فئوي")

# %%
# تحضير مصفوفات البيانات
X = train[feature_cols].fillna(-999)
X_test = test[feature_cols].fillna(-999)

print(f"📊 شكل بيانات التدريب: {X.shape}")
print(f"📊 شكل بيانات الاختبار: {X_test.shape}")

# %% [markdown]
# ## 🤖 الخطوة 6: تدريب النماذج

# %%
# إعدادات التدريب
N_FOLDS = 5
SEED = 42

# تخزين التنبؤات
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

pred_xgb = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))

# %%
# تدريب XGBoost
print("🚀 تدريب XGBoost...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"  Fold {fold+1}/{N_FOLDS}", end=" ")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
    
    model = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        early_stopping_rounds=50, eval_metric='mae', verbosity=0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_xgb[val_idx] = model.predict(X_val)
    pred_xgb += model.predict(X_test) / N_FOLDS
    
    mae = mean_absolute_error(y_val, oof_xgb[val_idx])
    print(f"MAE: {mae:.4f}")

xgb_score = mean_absolute_error(y_log, oof_xgb)
print(f"✅ XGBoost OOF MAE: {xgb_score:.4f}")

# %%
# تدريب LightGBM
print("\n🚀 تدريب LightGBM...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"  Fold {fold+1}/{N_FOLDS}", end=" ")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
        verbose=-1
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    
    oof_lgb[val_idx] = model.predict(X_val)
    pred_lgb += model.predict(X_test) / N_FOLDS
    
    mae = mean_absolute_error(y_val, oof_lgb[val_idx])
    print(f"MAE: {mae:.4f}")

lgb_score = mean_absolute_error(y_log, oof_lgb)
print(f"✅ LightGBM OOF MAE: {lgb_score:.4f}")

# %%
# تدريب CatBoost
print("\n🚀 تدريب CatBoost...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"  Fold {fold+1}/{N_FOLDS}", end=" ")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
    
    model = CatBoostRegressor(
        iterations=1000, depth=6, learning_rate=0.05,
        random_state=SEED, verbose=0
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
    
    oof_cat[val_idx] = model.predict(X_val)
    pred_cat += model.predict(X_test) / N_FOLDS
    
    mae = mean_absolute_error(y_val, oof_cat[val_idx])
    print(f"MAE: {mae:.4f}")

cat_score = mean_absolute_error(y_log, oof_cat)
print(f"✅ CatBoost OOF MAE: {cat_score:.4f}")

# %% [markdown]
# ## 🎯 الخطوة 7: إنشاء Ensemble وملف التقديم

# %%
# دمج تنبؤات النماذج (Ensemble)
oof_ensemble = (oof_xgb + oof_lgb + oof_cat) / 3
pred_ensemble = (pred_xgb + pred_lgb + pred_cat) / 3

ensemble_score = mean_absolute_error(y_log, oof_ensemble)
print(f"🏆 Ensemble OOF MAE: {ensemble_score:.4f}")

# تحويل التنبؤات للمقياس الأصلي
final_predictions = np.expm1(pred_ensemble)
final_predictions = np.maximum(final_predictions, 0)  # لا قيم سالبة

# %%
# إنشاء ملف التقديم
submission = pd.DataFrame({
    'ClaimNumber': test['ClaimNumber'],
    'UltimateIncurredClaimCost': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("✅ تم حفظ ملف التقديم: submission.csv")
submission.head()

# %%
# تحميل الملف
from google.colab import files
files.download('submission.csv')
print("📥 تم تحميل ملف التقديم!")

# %% [markdown]
# ## 📈 الخطوة 8: تحليل أهمية الميزات

# %%
# أهمية الميزات من آخر نموذج LightGBM
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importance['feature'][:20], importance['importance'][:20], color='steelblue')
plt.xlabel('الأهمية')
plt.title('أهم 20 ميزة')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 🎉 انتهى الكود!
# 
# ### ملخص النتائج:
# - تم تدريب 3 نماذج: XGBoost, LightGBM, CatBoost
# - تم دمجها في Ensemble للحصول على أفضل نتيجة
# - ملف التقديم جاهز للرفع على Kaggle
# 
# ### لتحسين النتيجة:
# 1. جرب hyperparameter tuning
# 2. أضف ميزات جديدة من النص (TF-IDF, embeddings)
# 3. جرب نماذج أخرى (Neural Networks)
