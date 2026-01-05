#!/usr/bin/env python
# coding: utf-8

# 工作环境
import os
import warnings
current_directory = os.getcwd()
warnings.filterwarnings('ignore')
print("当前工作目录:", current_directory)

import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

# 设置页面配置
st.set_page_config(
    page_title="术后慢性疼痛预测模型",
    layout="wide"
)

# 简洁的CSS美化
st.markdown("""
<style>
    /* 主标题样式 */
    .main-header {
        color: #1a237e;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* 子标题样式 */
    .section-title {
        color: #37474f;
        font-size: 1.2rem;
        font-weight: 500;
        margin: 1rem 0 0.5rem 0;
    }
    
    /* 输入卡片样式 */
    .input-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    /* 结果卡片样式 */
    .result-container {
        background: #f5f7fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    
    /* 预测按钮样式 */
    .predict-button {
        width: 100%;
        padding: 0.75rem;
        border: none;
        border-radius: 8px;
        background: #1a237e;
        color: white;
        font-weight: 500;
        font-size: 1rem;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    
    .predict-button:hover {
        background: #283593;
    }
    
    /* 输入框统一样式 */
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* 特征标签 */
    .feature-label {
        font-weight: 500;
        color: #455a64;
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# 设置 Matplotlib 的全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# 加载模型
model = joblib.load("simple_model_bylw.pkl")
min_max_params = joblib.load("min_max_params_app_bylw.pkl")
selected_features = joblib.load("selected_features_app_bylw.pkl")  # 20

feature_names = min_max_params["feature_names"]
selected_indices = [feature_names.index(f) for f in selected_features]

# 定义每个特征的输入范围
feature_ranges = {
    "Postoperative_pain_POD30": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Postoperative_pain_POD3": {"min": 0.0, "max": 10.0,"step": 1.0},
    "Rehabilitation_at_POD30": {"min": 0.0, "max": 10.0,"step": 1.0},
    "pulse__agg_linear_trend__attr_rvalue__chunk_len_10__f_agg_mean": {"min": -1.0, "max": 1.0,"step": 0.1},
    "Postoperative_pain_POD1": {"min": 0.0, "max": 10.0,"step": 0.1},
    "rr__fft_coefficient__attr_angle__coeff_3": {"min": -180.0, "max": 180.0,"step": 0.1},
    "co2__agg_linear_trend__attr_rvalue__chunk_len_10__f_agg_mean": {"min": -1.0, "max": 1.0,"step": 0.1},
    "map__fft_coefficient__attr_angle__coeff_4": {"min": -180.0, "max": 180.0,"step":0.1}
}

clinical_feature_names = {
     'Postoperative_pain_POD1': '术后1天急性疼痛NRS评分',
     'Postoperative_pain_POD3': '术后3天急性疼痛NRS评分',
     'Postoperative_pain_POD30': '术后30天亚急性疼痛NRS评分',
    'Rehabilitation_at_POD30':'术后30天主观康复情况',
     'Gender' : '性别', 
    'Postoperative_drainage_No': '引流管安置', 
    'PHQ_trouble_in_sleep':'PHQ-9：难以入睡或保持睡眠，或睡得太多', 
        'Open_surgery':'开放手术', 
        'Pain_score_v0':'术前疼痛NRS评分', 
        'Treponema_Pallidum_Antibody':'术前血清梅毒螺旋体抗体水平', 
        'Surgical_timing': '手术月份',
        'Surgery_site_Surface or limb': '浅表或四肢手术', 
        'Sleep_score_v0':'术前PSQI评分', 
    'PCA':'患者自控镇痛',
    'Thrombus_risk_No':'无血栓风险', 
    'pulse__cwt_coefficients__coeff_0__w_2__widths_251020':'手术开始阶段心率中频波动强度', 
    'Surgery_site_Thoracic':'胸科手术', 
        'Hepatitis_B_e_Antigen':'术前血清乙型肝炎E抗原水平',
        'Operation_grading_III': 'III级手术', 
    'Hemoglobin':'术前全血血红蛋白水平', 
        'pulse__cwt_coefficients__coeff_0__w_2':'手术开始阶段心率中频波动强度',
    'HIV_Antigen.Antibody_Combination':'术前血清HIV抗原抗体联合检测值', 
    'GAD_easily_annoyed_or_irritable':'GAD-7：变得容易烦恼或急躁',
    'Surgical_season_秋':'秋季手术',
    'PQSI_have_pain':'PSQI：疼痛不适',
    'IES_did_not_deal_with_them_No':'IES-R：处理本次疾病感受',
    'Surgery_site_Abdominal':'腹部手术',
    'Cancer_surgery': '肿瘤手术',
    'Thrombus_risk_Middle_or_high':'中高血栓风险',
    'Pain_history_No':'疼痛史',
    'Triglycerides':'术前血清甘油三酯水平',
    'Total_Bilirubin':'术前血清总胆红素水平',
    'PSQI_fell_too_hot':'PSQI：感觉热',
    'Education_Junior  school and below':'小学及以下文化水平',
    'Aspartate_Aminotransferase':"术前血清谷草转氨酶水平",
    'Surgical_season_冬':'冬季手术',
    'Surgery_site_Head and neck':'头颈部手术',
    'PSSS_others_console_me_No_or_neutrality':'PSSS：当我有困难时，有些人能够安慰到我',
    'IES_other_things_kept_making_me_think_about_it_No':'IES-R：别的东西也会让我想起本次疾病',
    'IES_irritable_and_angry_No':'IES-R：我感觉我易受刺激、易发怒',
    'Sleep_PSQI_v0':'术前睡眠障碍',
    'map__first_location_of_minimum':'术中平均动脉压最低值出现时间',
    'SPO2__last_location_of_minimum':'术中脉搏血氧饱和度最低值出现时间',
    'pulse__agg_linear_trend__attr_rvalue__chunk_len_10__f_agg_mean':'术中心率变化线性趋势',
    'rr__fft_coefficient__attr_angle__coeff_3':'术中呼吸频率波动相位',
    'co2__agg_linear_trend__attr_rvalue__chunk_len_10__f_agg_mean':'术中呼气末二氧化碳变化线性趋势',
    'map__fft_coefficient__attr_angle__coeff_4':'术中平均动脉压波动相位'
}

# 主界面标题
st.markdown('<div class="main-header">术后慢性疼痛预测模型 (CSAC)</div>', unsafe_allow_html=True)

# 创建两列布局，添加一些间距
col1, col2 = st.columns(2, gap="large")

inputs = {}

# 定义哪些特征是二元的
binary_features = ["Gender", "Postoperative_drainage_No", "Operation_grading_III",
                  "PHQ_trouble_in_sleep", 'Surgical_season_冬',
                  'Surgery_site_Surface or limb','Thrombus_risk_No',
                  'IES_did_not_deal_with_them_No',
                  'Surgical_season_秋',
                  'Open_surgery',
                  'Surgery_site_Abdominal',
                  'GAD_easily_annoyed_or_irritable']

# 将特征分成两组
mid_point = len(selected_features) // 2
left_features = selected_features[:mid_point]
right_features = selected_features[mid_point:]

# 左侧特征输入
with col1:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">患者基本信息</div>', unsafe_allow_html=True)
    
    for feature in left_features:
        display_name = clinical_feature_names.get(feature, feature)
        st.markdown(f'<div class="feature-label">{display_name}</div>', unsafe_allow_html=True)
        
        if feature in binary_features:
            inputs[feature] = st.selectbox(
                label="",
                options=[0, 1],
                format_func=lambda x: '否' if x == 0 else '是',
                key=f"left_{feature}",
                label_visibility="collapsed"
            )
        else:
            if feature in feature_ranges:
                min_val = feature_ranges[feature]["min"]
                max_val = feature_ranges[feature]["max"]
                step = feature_ranges[feature]["step"]
                
                inputs[feature] = st.slider(
                    label="",
                    min_value=min_val,
                    max_value=max_val,
                    value=min_val,
                    step=step,
                    key=f"left_{feature}",
                    label_visibility="collapsed"
                )
            else:
                inputs[feature] = st.number_input(
                    label="",
                    min_value=0.0,
                    step=0.1,
                    value=0.0,
                    key=f"left_{feature}",
                    label_visibility="collapsed"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 右侧特征输入
with col2:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">临床与监测参数</div>', unsafe_allow_html=True)
    
    for feature in right_features:
        display_name = clinical_feature_names.get(feature, feature)
        st.markdown(f'<div class="feature-label">{display_name}</div>', unsafe_allow_html=True)
        
        if feature in binary_features:
            inputs[feature] = st.selectbox(
                label="", 
                options=[0, 1],
                format_func=lambda x: '否' if x == 0 else '是',
                key=f"right_{feature}",
                label_visibility="collapsed"
            )
        else:
            if feature in feature_ranges:
                min_val = feature_ranges[feature]["min"]
                max_val = feature_ranges[feature]["max"]
                step = feature_ranges[feature]["step"]
                
                inputs[feature] = st.slider(
                    label="",
                    min_value=min_val,
                    max_value=max_val,
                    value=min_val,
                    step=step,
                    key=f"right_{feature}",
                    label_visibility="collapsed"
                )
            else:
                inputs[feature] = st.number_input(
                    label="",
                    min_value=0.0,
                    step=0.1,
                    value=0.0,
                    key=f"right_{feature}",
                    label_visibility="collapsed"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 预测按钮
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button("开始预测", use_container_width=True, type="primary"):
        # 数据处理和预测
        user_input = np.array([inputs[f] for f in selected_features])
        min_vals = min_max_params["min"][selected_indices]
        max_vals = min_max_params["max"][selected_indices]
        normalized_input = (user_input - min_vals) / (max_vals - min_vals)
                                
        # 预测
        prediction = model.predict([normalized_input])
        predicted_proba = model.predict_proba([normalized_input])[0]
        risk_probability = predicted_proba[1]
        
        # 显示预测结果
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">预测结果</div>', unsafe_allow_html=True)
        st.markdown(f"### 术后慢性疼痛风险: **{risk_probability * 100:.2f}%**")
        st.markdown(f"预测类别: {'高风险' if prediction[0] == 1 else '低风险'}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # SHAP值分析
        st.markdown('<div class="section-title">特征贡献分析</div>', unsafe_allow_html=True)
        
        with st.spinner("正在生成分析图表..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pd.DataFrame([normalized_input], columns=selected_features))
            
            # 显示SHAP力图
            plt.figure(figsize=(18, 8))
            shap.force_plot(explainer.expected_value, shap_values[0], 
                           pd.DataFrame([normalized_input], columns=selected_features), 
                           matplotlib=True)
            plt.tight_layout()
            
            # 直接显示图表而不是保存为文件
            st.pyplot(plt)
            
            # 添加解释说明
            st.caption("特征贡献分析：红色表示增加风险，蓝色表示降低风险")

# 页脚信息
st.markdown("---")
st.caption("术后慢性疼痛预测模型 | 仅供临床研究使用")
