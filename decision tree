import pandas as pd
import numpy as np
#데이터
data = pd.DataFrame({"no_prescription":["False","True","True","True","True","True","True","True","True","False"],
                     "no_treatment":["True","False","True","True","True","True","False","False","True","False"],
                     "no_wilting":["True","True","False","True","True","True","False","True","True","True"],
                     "no_diseases":["True","True","False","True","True","True","False","False","True","True"],
                     "tree_health":["Good","Good","Poor","Good","Good","Good","Poor","Poor","Good","Poor"]},
                    columns=["no_prescription","no_treatment","no_wilting","no_diseases","tree_health"])

#기술 속성(descriptive features)
features = data[["no_prescription","no_treatment","no_wilting","no_diseases","tree_health"]]

#대상 속성(target feature)
target = data["tree_health"]


#엔트로피
#np.unique - 고유한 원소들을 모은 뒤, 1차원 shape으로 변환하고 정렬
#고유한 원소의 등장하는 횟수
#엔트로피는 주어진 데이터셋의 불순도(impurity)를 측정하는데 사용된다. 불순도 : 다양한 범주들의 개체들이 얼마나 포함되어 있는가, 여러 가지의 클래스가 섞여 있는 정도
# 반대 순수도는 같은 클래스끼리 얼마나 많이 포함되어있는지를 말함.
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts = True)
    entropy = -np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

#정보 이득(information gain, IG)
#상위 노드의 엔트로피에서 하위 노드의 엔트로피를 뺀 값임.
def InfoGain(data, split_attribue_name, target_name):

    #전체 엔트로피 계산
    total_entropy = entropy(data[target_name])
    print("Entropy(D) =",round(total_entropy, 5))

    #가중 엔트로피 계산
    #where 조건 추출
    vals, counts = np.unique(data[split_attribue_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribue_name]==vals[i]).dropna()[target_name])
                               for i in range(len(vals))])
    print(f"H({split_attribue_name})={round(Weighted_Entropy,5)}")

    #정보이득 계산
    Information_Gain = total_entropy-Weighted_Entropy
    return Information_Gain

#ID3 알고리즘
def ID3(data, originaldata, features, target_attribue_name, parent_node_class = None):

    #중지기준 정의
    if len(np.unique(data[target_attribue_name])) <= 1:
        return np.unique(data[target_attribue_name])[0]

    #데이터가 없을 때 : 원본 데이터에서 최대값을 가지는 대상 속성 반환
    elif len(data) == 0:
        return np.unique(originaldata[target_attribue_name])\
            [np.argmax(np.unique(originaldata[target_attribue_name],return_counts=True)[1])]

    #기술 속성이 없을 때 : 부모 노드의 대상 속성 반환
    elif len(features) == 0:
        return parent_node_class

    #트리 성장
    else:
        #부모노드의 대상 속성 정의 (예 : Good)
        parent_node_class = np.unique(data[target_attribue_name])\
            [np.argmax(np.unique(data[target_attribue_name],return_counts=True)[1])]

        #데이터를 분할할 속성 선택
        item_values = [InfoGain(data,feature,target_attribue_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        #트리 구조 생성
        tree = {best_feature:{}}

        #최대 정보이득을 보인 기술 속성 제외
        features = [i for i in features if i != best_feature]

        #가지 성장
        for value in np.unique(data[best_feature]):
            #데이터 분할. dropna 결측값을 가진 행, 열 제거
            sub_data = data.where(data[best_feature]==value).dropna()

            #ID3 알고리즘
            subtree = ID3(sub_data, data, features, target_attribue_name, parent_node_class)
            tree[best_feature][value] = subtree
        return (tree)

# # numpy.unique: 고유값 반환
# print('numpy.unique: ', np.unique(data["tree_health"], return_counts = True)[1])
# # numpy.max: 최대값 반환
#
# print('numpy.max: ', np.max(np.unique(data["tree_health"], return_counts = True)[1]))
# # numpy.argmax: 최대값이 위치한  인덱스 반환
#
# print('numpy.argmax: ', np.argmax(np.unique(data["tree_health"], return_counts = True)[1]))

from pprint import pp
tree = ID3(data, data, ["no_prescription","no_wilting","no_diseases"], "tree_health")
pp(tree)
