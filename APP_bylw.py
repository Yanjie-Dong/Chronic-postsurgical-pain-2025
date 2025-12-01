#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 工作环境
import os
import warnings
current_directory = os.getcwd()
warnings.filterwarnings('ignore')
print("当前工作目录:", current_directory)


# In[2]:


import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# 设置 Matplotlib 的全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'


# In[13]:


# 
model = joblib.load("simple_model_bylw.pkl")
min_max_params = joblib.load("min_max_params_app_bylw.pkl")
selected_features = joblib.load("selected_features_app_bylw.pkl")  # 25


# In[14]:


feature_names = min_max_params["feature_names"]
selected_indices = [feature_names.index(f) for f in selected_features]


# In[21]:





# In[17]:


# 定义每个特征的输入范围
feature_ranges = {
    "Postoperative_pain_POD30": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Postoperative_pain_POD3": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Rehabilitation_at_POD30": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Treponema_Pallidum_Antibody": {"min": 1.0, "max": 12.0,"step": 1.0},
    "Postoperative_pain_POD1": {"min": 0.0, "max": 1.0,"step": 0.1},
    "Hospitalizing_expenses": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Hepatitis_B_e_Antigen": {"min": 0.0, "max": 10.0,"step": 1.0},
    "map__agg_linear_trend__attr_slope__chunk_len_10__f_agg_min": {"min": 0.0, "max": None,"step":0.1},
    "Pain_score_v0": {"min": 0.0, "max": None,"step": 0.1},
    #连续特征的范围
}


# In[ ]:


clinical_feature_names = {
      'Postoperative_pain_POD30': '术后30天亚急性疼痛NRS评分',
     'Postoperative_pain_POD3' : '术后3天急性疼痛NRS评分', 
    'Rehabilitation_at_POD30': '术后30天主观康复程度', 
    'Gender':'性别', 
    'Postoperative_pain_POD1': '术后1天急性疼痛NRS评分',
     'Gender' : '性别', 
    'Postoperative_drainage_Yes': '引流管安置', 
    'PHQ_trouble_in_sleep':'PHQ9-难以入睡或保持睡眠，或睡得太多', 
        'Open_surgery':'开放手术', 
        'Pain_score_v0':'术前疼痛NRS评分', 
        'Treponema_Pallidum_Antibody':'术前血清梅毒螺旋体抗体水平', 
        'Surgical_timing': '手术月份',
        'Surgery_site_Surface or limb': '浅表或四肢手术', 
        'Sleep_score_v0':'术前PSQI评分', 
    'PCA':'患者自控镇痛',
    'Thrombus_risk_No':'无血栓风险', 
    'map__agg_linear_trend__attr_slope__chunk_len_10__f_agg_min':'术中平均动脉压最低值的长期趋势', 
    'Surgery_site_Thoracic':'胸科手术', 
        'Hepatitis_B_e_Antigen':'术前血清乙型肝炎E抗原水平',
        'Operation_grading_III': 'III级手术', 
    'Hemoglobin':'术前全血血红蛋白水平', 
        'pulse__cwt_coefficients__coeff_0__w_2':'手术开始阶段心率中频波动强度',
    'HIV_Antigen.Antibody_Combination':'术前血清HIV抗原抗体联合检测值', 
    'GAD_easily_annoyed_or_irritable':'GAD7-变得容易烦恼或急躁',
    'Surgical_season_秋':'秋季手术',
    'PQSI_have_pain':'PSQI-疼痛不适',
    'IES_did_not_deal_with_them_No':'IES-R-处理本次疾病感受',
    'Surgery_site_Abdominal':'腹部手术',
    'Cancer_surgery': '肿瘤手术',
    'Thrombus_risk_Middle_or_high':'中高血栓风险',
    'Pain_history_No':'疼痛史',
    'Triglycerides':'术前血清甘油三酯水平',
    'Total_Bilirubin':'术前血清总胆红素水平',
    'PSQI_fell_too_hot':'PSQI-感觉热',
    'Education_Junior  school and below':'小学及以下文化水平',
    'Aspartate_Aminotransferase':"术前血清谷草转氨酶水平",
    'Hospitalizing_expenses':'住院费用'
}


# In[24]:


st.title("CPSP Prediction")
inputs = {}

# 定义哪些特征是二元的
binary_features = ["Gender", "Postoperative_drainage_Yes", "PHQ_trouble_in_sleep",
                  "Surgical_season_秋", 'Operation_grading_III',
                  'Surgery_site_Abdominal','Open_surgery',
                  'Thrombus_risk_No',
                  'Surgery_site_Surface or limb',
                  'PSQI_fell_too_hot',
                  'IES_did_not_deal_with_them_No']  # 示例：这些特征只能取0或1

for feature in selected_features:
    display_name = clinical_feature_names.get(feature, feature)  # 获取临床名称
    if feature in binary_features:
        # 二元特征，只能取0或1
        inputs[feature] = st.selectbox(
            display_name, 
            options=[0, 1],
            format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)'
        )
    else:
        # 连续特征，使用手动定义的范围和步长
        min_val = feature_ranges[feature]["min"]
        max_val = feature_ranges[feature]["max"]
        step = feature_ranges[feature]["step"]
        
        if max_val is None:
            # 如果没有设置最大值，允许用户输入任意大的值
            inputs[feature] = st.number_input(
                display_name, 
                min_value=min_val,
                step=step,
                value=min_val  # 默认值设置为最小值
            )
        else:
            # 如果设置了最大值，使用范围限制
            inputs[feature] = st.number_input(
                display_name,
                min_value=min_val,
                max_value=max_val,
                step=step,
                value=min_val  # 默认值设置为最小值
            )


# In[23]:


if st.button("Predict"):
    # (1)
    user_input = np.array([inputs[f] for f in selected_features])
    min_vals = min_max_params["min"][selected_indices]  # 25min
    max_vals = min_max_params["max"][selected_indices]  # 25max
    normalized_input = (user_input - min_vals) / (max_vals - min_vals)
                            
     # (2) 预测（注意输入是2D数组）
    prediction = model.predict([normalized_input])
    predicted_proba = model.predict_proba([normalized_input])[0]

    # 显示预测结果
    risk_probability = predicted_proba[1]  # 正类的概率
    st.success(f"Based on this model, the CPSP risk of this patient is {risk_probability * 100:.2f}%")

    # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([normalized_input], columns=selected_features))

    # 显示SHAP力图
    plt.figure(figsize=(20, 12))  # 调整图像大小
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([normalized_input], columns=selected_features), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_force_plot.png")


# In[ ]:




