# %% [markdown]
# # 🏛️ أتمتة العملية الاكتوارية للتسعير
# ## Actuarial Pricing Process Automation
# 
# ### 📌 الهدف من هذا الكود:
# شرح كيفية أتمتة عملية التسعير الاكتوارية مع توضيح **نقاط التدخل واتخاذ القرار** في كل مرحلة
# 
# ### 🔄 مراحل العملية الاكتوارية:
# ```
# 1. جمع البيانات → 2. تنظيف البيانات → 3. التحليل الاستكشافي → 4. تقدير الخسائر
#                                    ↓
# 8. المراقبة المستمرة ← 7. التسعير النهائي ← 6. حساب الأقساط ← 5. عوامل الخطر
# ```

# %% [markdown]
# ---
# ## 📦 المرحلة 0: إعداد البيئة

# %%
# تثبيت المكتبات
!pip install -q xgboost lightgbm catboost shap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("✅ تم تحميل المكتبات")

# %% [markdown]
# ---
# ## 📥 المرحلة 1: جمع البيانات (Data Collection)
# 
# ### 🎯 الهدف:
# جمع بيانات المطالبات التاريخية من مصادر مختلفة
# 
# ### 🔧 نقطة القرار:
# - ما هي الفترة الزمنية المناسبة للبيانات؟
# - هل نحتاج بيانات خارجية (اقتصادية، ديموغرافية)؟
# - ما مستوى التفصيل المطلوب؟

# %%
# === نقطة قرار: تحديد مصادر البيانات ===
DATA_CONFIG = {
    'source': 'kaggle',  # يمكن تغييره إلى: 'database', 'api', 'file'
    'period_years': 5,   # عدد سنوات البيانات
    'include_external': True  # إضافة بيانات خارجية
}

print(f"📊 إعدادات جمع البيانات:")
print(f"   - المصدر: {DATA_CONFIG['source']}")
print(f"   - الفترة: {DATA_CONFIG['period_years']} سنوات")
print(f"   - بيانات خارجية: {'نعم' if DATA_CONFIG['include_external'] else 'لا'}")

# %%
# تحميل البيانات
# ملاحظة: قم برفع ملف kaggle.json أولاً
!mkdir -p ~/.kaggle
!kaggle competitions download -c actuarial-loss-estimation 2>/dev/null || echo "⚠️ تأكد من إعداد Kaggle API"
!unzip -q -o actuarial-loss-estimation.zip -d data/ 2>/dev/null || echo "جاري التحميل..."

try:
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print(f"✅ تم تحميل البيانات: {len(train)} مطالبة تدريب، {len(test)} مطالبة اختبار")
except:
    print("⚠️ يرجى تحميل البيانات يدوياً")

# %% [markdown]
# ---
# ## 🧹 المرحلة 2: تنظيف البيانات (Data Cleaning)
# 
# ### 🎯 الهدف:
# التأكد من جودة البيانات وتنظيفها
# 
# ### 🔧 نقاط القرار:
# - كيف نتعامل مع القيم المفقودة؟
# - هل نحذف الحالات الشاذة (Outliers)؟
# - ما هو حد القيم المتطرفة؟

# %%
def analyze_data_quality(df, name="البيانات"):
    """تحليل جودة البيانات"""
    print(f"\n{'='*50}")
    print(f"📋 تقرير جودة {name}")
    print(f"{'='*50}")
    
    total = len(df)
    print(f"📊 إجمالي السجلات: {total:,}")
    
    # القيم المفقودة
    missing = df.isnull().sum()
    missing_pct = (missing / total * 100).round(2)
    missing_report = pd.DataFrame({
        'عدد المفقود': missing,
        'النسبة %': missing_pct
    })
    missing_report = missing_report[missing_report['عدد المفقود'] > 0]
    
    if len(missing_report) > 0:
        print(f"\n⚠️ الأعمدة التي بها قيم مفقودة:")
        print(missing_report)
    else:
        print("✅ لا توجد قيم مفقودة")
    
    return missing_report

# تحليل جودة البيانات
quality_report = analyze_data_quality(train, "بيانات التدريب")

# %%
# === نقطة قرار: استراتيجية التعامل مع القيم المفقودة ===
MISSING_STRATEGY = {
    'numeric': 'median',      # median, mean, zero, drop
    'categorical': 'mode',    # mode, 'MISSING', drop
    'threshold_drop': 0.5     # حذف الأعمدة التي تزيد نسبة المفقود فيها عن 50%
}

print("🔧 استراتيجية معالجة القيم المفقودة:")
print(f"   - الأرقام: {MISSING_STRATEGY['numeric']}")
print(f"   - الفئات: {MISSING_STRATEGY['categorical']}")
print(f"   - حد الحذف: {MISSING_STRATEGY['threshold_drop']*100}%")

# %%
def clean_data(df, config):
    """تنظيف البيانات حسب الاستراتيجية المحددة"""
    df = df.copy()
    
    # حذف الأعمدة ذات المفقود الكثير
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df)
        if missing_pct > config['threshold_drop']:
            df = df.drop(columns=[col])
            print(f"🗑️ حذف العمود {col} (نسبة المفقود: {missing_pct:.1%})")
    
    # معالجة الأعمدة الرقمية
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if config['numeric'] == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif config['numeric'] == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif config['numeric'] == 'zero':
                df[col] = df[col].fillna(0)
    
    # معالجة الأعمدة الفئوية
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            if config['categorical'] == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'MISSING')
            else:
                df[col] = df[col].fillna('MISSING')
    
    return df

train_clean = clean_data(train, MISSING_STRATEGY)
test_clean = clean_data(test, MISSING_STRATEGY)
print(f"\n✅ تم تنظيف البيانات")

# %% [markdown]
# ---
# ## 📊 المرحلة 3: التحليل الاستكشافي (Exploratory Data Analysis)
# 
# ### 🎯 الهدف:
# فهم توزيع البيانات وتحديد الأنماط
# 
# ### 🔧 نقاط القرار:
# - هل التوزيع طبيعي أم يحتاج تحويل؟
# - ما هي المتغيرات الأكثر تأثيراً؟
# - هل هناك ارتباطات قوية؟

# %%
TARGET = 'UltimateIncurredClaimCost'

# تحليل المتغير المستهدف
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# التوزيع الأصلي
axes[0].hist(train_clean[TARGET], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('توزيع التكلفة الأصلي', fontsize=12)
axes[0].set_xlabel('التكلفة')

# التوزيع اللوغاريتمي
axes[1].hist(np.log1p(train_clean[TARGET]), bins=50, color='coral', edgecolor='white')
axes[1].set_title('توزيع log(التكلفة+1)', fontsize=12)
axes[1].set_xlabel('log(التكلفة)')

# Box plot
axes[2].boxplot(train_clean[TARGET])
axes[2].set_title('Box Plot - الكشف عن القيم المتطرفة', fontsize=12)

plt.tight_layout()
plt.show()

# %%
# === نقطة قرار: تحويل المتغير المستهدف ===
# بناءً على التحليل أعلاه، نقرر:
TARGET_TRANSFORM = 'log'  # 'none', 'log', 'sqrt', 'boxcox'

print(f"🎯 قرار تحويل المتغير المستهدف: {TARGET_TRANSFORM}")

if TARGET_TRANSFORM == 'log':
    y = np.log1p(train_clean[TARGET])
    print("   → سيتم استخدام التحويل اللوغاريتمي لتطبيع التوزيع")
elif TARGET_TRANSFORM == 'sqrt':
    y = np.sqrt(train_clean[TARGET])
else:
    y = train_clean[TARGET]

# %%
# === نقطة قرار: التعامل مع القيم المتطرفة ===
OUTLIER_CONFIG = {
    'method': 'iqr',        # 'iqr', 'zscore', 'percentile', 'none'
    'multiplier': 3.0,      # لـ IQR
    'action': 'cap'         # 'cap', 'remove', 'none'
}

print(f"\n🔧 استراتيجية القيم المتطرفة:")
print(f"   - الطريقة: {OUTLIER_CONFIG['method']}")
print(f"   - الإجراء: {OUTLIER_CONFIG['action']}")

if OUTLIER_CONFIG['method'] == 'iqr' and OUTLIER_CONFIG['action'] != 'none':
    Q1 = train_clean[TARGET].quantile(0.25)
    Q3 = train_clean[TARGET].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - OUTLIER_CONFIG['multiplier'] * IQR
    upper = Q3 + OUTLIER_CONFIG['multiplier'] * IQR
    
    outliers = ((train_clean[TARGET] < lower) | (train_clean[TARGET] > upper)).sum()
    print(f"   - عدد القيم المتطرفة: {outliers} ({outliers/len(train_clean)*100:.1f}%)")
    print(f"   - الحد الأدنى: {lower:,.0f}")
    print(f"   - الحد الأقصى: {upper:,.0f}")

# %% [markdown]
# ---
# ## 📈 المرحلة 4: تقدير الخسائر (Loss Estimation)
# 
# ### 🎯 الهدف:
# بناء نموذج للتنبؤ بتكلفة المطالبات النهائية
# 
# ### 🔧 نقاط القرار:
# - أي نموذج نستخدم؟
# - ما هي الـ Hyperparameters المناسبة؟
# - كم عدد الـ Folds للتحقق؟

# %%
# تحضير الميزات
EXCLUDE_COLS = [TARGET, 'ClaimNumber', 'ClaimDescription', 'AccidentDescription',
                'DateTimeOfAccident', 'DateReported', 'DateOfBirth']

feature_cols = [c for c in train_clean.columns if c not in EXCLUDE_COLS]

# ترميز المتغيرات الفئوية
cat_cols = train_clean[feature_cols].select_dtypes(include=['object']).columns
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train_clean[col], test_clean[col]]).astype(str).unique()
    le.fit(all_vals)
    train_clean[col] = le.transform(train_clean[col].astype(str))
    test_clean[col] = le.transform(test_clean[col].astype(str))
    encoders[col] = le

X = train_clean[feature_cols].fillna(-999)
X_test = test_clean[feature_cols].fillna(-999)

print(f"✅ تم تحضير {len(feature_cols)} ميزة للنمذجة")

# %%
# === نقطة قرار: اختيار النموذج وإعداداته ===
MODEL_CONFIG = {
    'models': ['xgboost', 'lightgbm', 'catboost'],  # النماذج المستخدمة
    'n_folds': 5,
    'ensemble_method': 'average',  # 'average', 'weighted', 'stacking'
    'hyperparams': {
        'xgboost': {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 1000},
        'lightgbm': {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 1000},
        'catboost': {'depth': 6, 'learning_rate': 0.05, 'iterations': 1000}
    }
}

print("🤖 إعدادات النمذجة:")
print(f"   - النماذج: {', '.join(MODEL_CONFIG['models'])}")
print(f"   - عدد الـ Folds: {MODEL_CONFIG['n_folds']}")
print(f"   - طريقة الدمج: {MODEL_CONFIG['ensemble_method']}")

# %%
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# تدريب النماذج
kf = KFold(n_splits=MODEL_CONFIG['n_folds'], shuffle=True, random_state=42)
results = {}

for model_name in MODEL_CONFIG['models']:
    print(f"\n🚀 تدريب {model_name}...")
    
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        if model_name == 'xgboost':
            model = xgb.XGBRegressor(**MODEL_CONFIG['hyperparams']['xgboost'], 
                                      random_state=42, verbosity=0)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif model_name == 'lightgbm':
            model = lgb.LGBMRegressor(**MODEL_CONFIG['hyperparams']['lightgbm'],
                                       random_state=42, verbose=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
        else:  # catboost
            model = CatBoostRegressor(**MODEL_CONFIG['hyperparams']['catboost'],
                                       random_state=42, verbose=0)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
        
        oof[val_idx] = model.predict(X_val)
        pred += model.predict(X_test) / MODEL_CONFIG['n_folds']
        
        mae = mean_absolute_error(y_val, oof[val_idx])
        print(f"   Fold {fold+1}: MAE = {mae:.4f}")
    
    overall_mae = mean_absolute_error(y, oof)
    results[model_name] = {'oof': oof, 'pred': pred, 'mae': overall_mae}
    print(f"   ✅ {model_name} OOF MAE: {overall_mae:.4f}")

# %% [markdown]
# ---
# ## ⚖️ المرحلة 5: تحليل عوامل الخطر (Risk Factors)
# 
# ### 🎯 الهدف:
# فهم العوامل المؤثرة في تكلفة المطالبات
# 
# ### 🔧 نقطة القرار:
# - ما هي أهم المتغيرات؟
# - هل نحتاج لتعديل الأوزان؟

# %%
# تحليل أهمية الميزات
import shap

# استخدام آخر نموذج للتفسير
print("📊 تحليل أهمية العوامل...")

# Feature Importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(len(feature_cols))
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance['feature'][:15], importance['importance'][:15], color='steelblue')
plt.xlabel('الأهمية')
plt.title('أهم 15 عامل مؤثر في تكلفة المطالبات')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %%
# === نقطة قرار: مراجعة عوامل الخطر ===
print("\n🔍 مراجعة الخبير الاكتواري لعوامل الخطر:")
print("   ← هل تتوافق هذه العوامل مع الخبرة العملية؟")
print("   ← هل هناك عوامل مفقودة يجب إضافتها؟")
print("   ← هل نحتاج لتعديل أوزان بعض العوامل؟")

RISK_FACTOR_ADJUSTMENTS = {
    'apply_manual_adjustments': False,  # True لتطبيق تعديلات يدوية
    'adjustments': {
        # 'feature_name': adjustment_factor
    }
}

# %% [markdown]
# ---
# ## 💰 المرحلة 6: حساب الأقساط (Premium Calculation)
# 
# ### 🎯 الهدف:
# تحويل تقديرات الخسارة إلى أقساط تأمين
# 
# ### 🔧 نقاط القرار:
# - نسبة التحميل (Loading)؟
# - هامش الربح؟
# - عوامل التعديل؟

# %%
# === نقطة قرار: معاملات التسعير ===
PRICING_CONFIG = {
    'expense_loading': 0.25,      # نسبة المصاريف الإدارية
    'profit_margin': 0.10,        # هامش الربح المستهدف
    'contingency_margin': 0.05,   # هامش الطوارئ
    'reinsurance_cost': 0.03,     # تكلفة إعادة التأمين
    'commission_rate': 0.15       # عمولة الوسطاء
}

print("💰 معاملات التسعير:")
for key, value in PRICING_CONFIG.items():
    print(f"   - {key}: {value*100:.1f}%")

total_loading = sum(PRICING_CONFIG.values())
print(f"\n   📊 إجمالي التحميل: {total_loading*100:.1f}%")

# %%
def calculate_premium(expected_loss, config):
    """
    حساب القسط بناءً على الخسارة المتوقعة ومعاملات التسعير
    
    القسط = الخسارة المتوقعة × (1 + مجموع التحميلات)
    """
    total_loading = sum(config.values())
    premium = expected_loss * (1 + total_loading)
    return premium

# حساب الأقساط للتنبؤات
ensemble_pred = np.mean([results[m]['pred'] for m in results], axis=0)

# تحويل من log إلى المقياس الأصلي
if TARGET_TRANSFORM == 'log':
    expected_loss = np.expm1(ensemble_pred)
else:
    expected_loss = ensemble_pred

calculated_premium = calculate_premium(expected_loss, PRICING_CONFIG)

print(f"\n📊 إحصائيات الأقساط المحسوبة:")
print(f"   - المتوسط: ${np.mean(calculated_premium):,.2f}")
print(f"   - الوسيط: ${np.median(calculated_premium):,.2f}")
print(f"   - الحد الأدنى: ${np.min(calculated_premium):,.2f}")
print(f"   - الحد الأقصى: ${np.max(calculated_premium):,.2f}")

# %% [markdown]
# ---
# ## ✅ المرحلة 7: التسعير النهائي (Final Pricing)
# 
# ### 🎯 الهدف:
# مراجعة واعتماد الأسعار النهائية
# 
# ### 🔧 نقاط القرار:
# - هل الأسعار تنافسية؟
# - هل تتوافق مع متطلبات الجهات الرقابية؟
# - هل هناك حاجة لتعديلات نهائية؟

# %%
# === نقطة قرار: المراجعة النهائية والتعديلات ===
FINAL_ADJUSTMENTS = {
    'apply_market_adjustment': True,
    'market_adjustment_factor': 0.95,  # خصم 5% للتنافسية
    'min_premium': 100,                # الحد الأدنى للقسط
    'max_premium': 1000000,            # الحد الأقصى للقسط
    'round_to': 10                     # تقريب إلى أقرب 10
}

print("📋 التعديلات النهائية:")
print(f"   - تعديل السوق: {'نعم' if FINAL_ADJUSTMENTS['apply_market_adjustment'] else 'لا'}")
if FINAL_ADJUSTMENTS['apply_market_adjustment']:
    print(f"   - معامل التعديل: {FINAL_ADJUSTMENTS['market_adjustment_factor']}")
print(f"   - الحد الأدنى: ${FINAL_ADJUSTMENTS['min_premium']:,}")
print(f"   - الحد الأقصى: ${FINAL_ADJUSTMENTS['max_premium']:,}")

# %%
def apply_final_adjustments(premium, config):
    """تطبيق التعديلات النهائية على الأقساط"""
    final_premium = premium.copy()
    
    # تعديل السوق
    if config['apply_market_adjustment']:
        final_premium = final_premium * config['market_adjustment_factor']
    
    # تطبيق الحدود
    final_premium = np.clip(final_premium, config['min_premium'], config['max_premium'])
    
    # التقريب
    final_premium = np.round(final_premium / config['round_to']) * config['round_to']
    
    return final_premium

final_premium = apply_final_adjustments(calculated_premium, FINAL_ADJUSTMENTS)

print(f"\n✅ الأقساط النهائية:")
print(f"   - المتوسط: ${np.mean(final_premium):,.2f}")
print(f"   - الوسيط: ${np.median(final_premium):,.2f}")

# %% [markdown]
# ---
# ## 📤 المرحلة 8: إنشاء ملف التقديم

# %%
# إنشاء ملف التقديم (للمسابقة: نستخدم الخسارة المتوقعة وليس القسط)
submission = pd.DataFrame({
    'ClaimNumber': test_clean['ClaimNumber'],
    'UltimateIncurredClaimCost': np.maximum(expected_loss, 0)  # لا قيم سالبة
})

submission.to_csv('submission.csv', index=False)
print("✅ تم حفظ ملف التقديم: submission.csv")
print(submission.head())

# %%
# تحميل الملف (في Colab)
try:
    from google.colab import files
    files.download('submission.csv')
    print("📥 تم تحميل الملف")
except:
    print("📁 الملف محفوظ في: submission.csv")

# %% [markdown]
# ---
# ## 📊 ملخص العملية الاكتوارية
# 
# | المرحلة | الهدف | نقاط القرار الرئيسية |
# |---------|-------|----------------------|
# | 1. جمع البيانات | جمع البيانات التاريخية | مصادر البيانات، الفترة الزمنية |
# | 2. تنظيف البيانات | ضمان جودة البيانات | معالجة المفقود، القيم المتطرفة |
# | 3. التحليل الاستكشافي | فهم الأنماط | تحويل المتغيرات، الارتباطات |
# | 4. تقدير الخسائر | بناء نموذج التنبؤ | اختيار النموذج، المعاملات |
# | 5. عوامل الخطر | تحديد المؤثرات | مراجعة الأوزان، التعديلات |
# | 6. حساب الأقساط | تحويل لأقساط | التحميلات، الهوامش |
# | 7. التسعير النهائي | الاعتماد النهائي | تعديلات السوق، الحدود |
# 
# ### 🎯 النتائج:
# - تم بناء نظام أتمتة متكامل للتسعير الاكتواري
# - كل مرحلة تتضمن نقاط قرار واضحة للتدخل البشري
# - النظام قابل للتخصيص حسب احتياجات الشركة
