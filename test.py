import pandas as pd
from sklearn import tree
from IPython.display import Image
import numpy as np
import pydotplus
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data2021 = pd.read_spss("C:/Users/godsu/OneDrive/바탕 화면/논문/kdca/kyrbs2021.sav")
#data2020 = pd.read_spss("C:/Users/godsu/OneDrive/바탕 화면/논문/kdca/kyrbs2020.sav")
#data2019 = pd.read_spss("C:/Users/godsu/OneDrive/바탕 화면/논문/kdca/kyrbs2019.sav")
#data2018 = pd.read_spss("C:/Users/godsu/OneDrive/바탕 화면/논문/kdca/kyrbs2018.sav")
#data2017 = pd.read_spss("C:/Users/godsu/OneDrive/바탕 화면/논문/kdca/kyrbs2017.sav")

"""
SEX 성별
AGE 나이
E_S_RCRD 학업성적
E_SES 가정의 경제적 상태
E_RES 거주형태
E_FM_F_1 아버지
E_FM_SF_2 새아버지
E_FM_M_3 어머니
E_FM_SM_4 새어머니
E_LT_F 아빠랑 같이살고있다
E_LT_SF 새아빠랑 같이살고있다
E_LT_M 엄마랑 같이살고있다
E_LT_SM 새엄마랑 같이살고있다
E_EDU_F 아버지 학력
E_KRN_F 아버지 한국사람인가
E_BORN_F 아버지 어느나라사람
E_EDU_M 어머니 학력
E_KRN_M 어머니 한국사람인가
E_BORN_M 어머니 어느나라사람
M_SLP_EN 최근 7일동안 잠을 잔 시간이 피로회복에 충분한가? 1매우충분 2충분 3그저그렇다 4충분하지않다 5전혀충분하지않다
M_SLP_HR 주중 잠든 시간
M_SLP_MM 주중 잠든 분
M_WK_HR 주중 일어난 시간
M_WK_MM 주중 일어난 분
M_SLP_HR_k 주말 잠든 시간
M_SLP_MM_k 주말 잠든 분
M_WK_HR_k 주말 일어난 시간
M_WK_MM_k 주말 일어난 분
M_SAD 최근 2주내 슬프거나 절망감
M_SUI_CON 최근 12개월 내 자살생각
M_SUI_PLN 구체적인 자살계획 짰나
M_SUI_ATT 자살 시도
V_TRT 폭력당해서 병원에서 치료받은적
AC_LT 지금까지 1잔이상 음주
AC_FAGE 처음으로 1잔이상 마셔본 때
AC_DAYS 한달 내 1잔이상 술마신 날 며칠
AC_AMNT 술마실때 평균주량
TC_LT 흡연경험
TC_DAYS 30일동안 한개비라도 피운 날 며칠
TC_FAGE 처음으로 담배 한개비 피워본 때
TC_DAGE 매일 피우기 시작 한 때
TC_AMNT 하루 평균 흡연량
TC_QT_YR 금연시도
TC_SND_H 집안에서 담배냄새 맡은적 며칠
"""
col = [
    'SEX','AGE','E_S_RCRD','E_RES','E_SES','E_FM_F_1','E_FM_SF_2','E_FM_M_3','E_FM_SM_4',
    'E_LT_F','E_LT_SF','E_LT_M','E_LT_SM',
    'E_EDU_F','E_KRN_F','E_BORN_F','E_EDU_M','E_KRN_M','E_BORN_M',
    'M_SLP_EN','M_SLP_HR','M_SLP_MM','M_WK_HR','M_WK_MM','M_SLP_HR_K','M_SLP_MM_K','M_WK_HR_K','M_WK_MM_K',
    'M_SAD', 'M_SUI_CON', 'M_SUI_PLN', 'M_SUI_ATT', 'V_TRT',
    'AC_LT','AC_FAGE','AC_DAYS','AC_AMNT',
    'TC_LT','TC_DAYS','TC_FAGE','TC_DAGE','TC_AMNT','TC_QT_YR','TC_SND_H'
] 

print(data2021['TC_LT'])

x = np.array(pd.DataFrame(data2021, columns=['SEX']))
y = np.array(pd.DataFrame(data2021, columns=['TC_LT']))
X_train, X_test, y_train, y_test = train_test_split(x,y)

dt_clf = DecisionTreeClassifier()
dt_clf = dt_clf.fit(X_train, y_train)
dt_prediction = dt_clf.predict(X_test)


os.environ["PATH"] += os.pathsep + 'C:/Users/godsu/OneDrive/바탕 화면/논문/Graphviz/bin/'
feature_names = ['SEX']#data2021.columns.tolist()
feature_names = feature_names[0:1]



target_name = np.array(['Play No', 'Play Yes'])
dt_dot_data = tree.export_graphviz(dt_clf, out_file = None,
                                  feature_names = feature_names,
                                  class_names = target_name,
                                  filled = True, rounded = True,
                                  special_characters = True)

dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)
Image(dt_graph.create_png())

#print(data2020[col])
#print(data2019[col])
#print(data2018[col])
#print(data2017[col])

