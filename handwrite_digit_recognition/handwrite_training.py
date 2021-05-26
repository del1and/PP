import numpy as np
from PIL import Image
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
import pandas as pd


def preprocessing():
    image_list = []
    label_txt = open('./handwrite_image/label.txt')
    for filename in glob.glob('./handwrite_image/*.png'):  # open all images in folder
           images=Image.open(filename)
           image_list.append(images)
    print("이미지 개수:", len(image_list))

    label_all = label_txt.read(len(image_list))  # read label txt 0 to len(image_list) bytes.
    label = []
    for label_to_matrix in range(len(label_all)):  # label txt to matrix
        label.append(int(label_all[label_to_matrix]))
    print("레이블 수:", len(label))

    picture_number = []
    for images_number in range(len(image_list)):  # load all images
        picture_number.append(image_list[images_number].load)
    print("이미지 사이즈:", image_list[0].size[0], "*", image_list[0].size[1], "pixels")

    # extract image's rgb to int. use one color. one image: 768 pixels
    AbsImageRGB = np.zeros((len(image_list), image_list[0].size[0], image_list[0].size[1]))  # 3 dimensions.

    for numberofimage in range(len(image_list)):
        for i in range(image_list[0].size[0]):
            for j in range(image_list[0].size[1]):
                AbsImageRGB[numberofimage, i, j] = int(abs((image_list[numberofimage].getpixel((i, j))[0] +
                                                            image_list[numberofimage].getpixel((i, j))[1] +
                                                            image_list[numberofimage].getpixel((i, j))[2] - 765)/3))

    matrix = []  # loop for 1 line is 1 image, has 784 Characteristic value
    for reshape in range(len(image_list)):
        for k in range(image_list[0].size[0]):
            for l in range(image_list[0].size[1]):
                matrix.append(AbsImageRGB[reshape, k, l])
    matrix = np.reshape(matrix, (int(len(matrix)/784), 784))

    # split sets. All of classifiers use same set.
    X_train, X_test, y_train, y_test = train_test_split(matrix, label, test_size=0.20, random_state=42)

    return matrix, label, X_train, X_test, y_train, y_test


def MLP(X_train, X_test, y_train, y_test, image, label):
    clf = MLPClassifier(hidden_layer_sizes=(1500,), activation='relu', alpha=0.01, max_iter=15000, shuffle=True)
    clf.fit(X_train, y_train)  # MLP
    print("테스트 레이블 정답:", y_test[:10], "...")

    predict_result = clf.predict(X_test)
    print("MLP 테스트 예측 레이블:", predict_result[:10], "...")
    print("MLP 테스트 세트 정확도: {:.3f}".format(clf.score(X_test, y_test)))

    # --- for scaled MLP ---
    train_scaler    = MinMaxScaler().fit(X_train)
    test_scaler     = MinMaxScaler().fit(X_test)
    X_train_scaled  = train_scaler.transform(X_train)
    X_test_scaled   = test_scaler.transform(X_test)
    clf.fit(X_train_scaled, y_train)  # re-fitting set for scaled set.

    predict_scaled_result = clf.predict(X_test_scaled)

    print("Scaled된 MLP 테스트 예측 레이블:", predict_scaled_result[:10])
    print("Scaled된 MLP 테스트 세트 정확도: {:.3f}".format(clf.score(X_test_scaled, y_test)))

    # --- for cross validation ---
    clf.fit(image, label)  # re-fitting set for cross validation.
    cross_validation_score = cross_val_score(clf, image, label, cv=5)
    print("MLP의 5-fold Cross-Validation 점수: ", round(np.mean(cross_validation_score), 3))

    return predict_result, predict_scaled_result


def SVM(X_train, X_test, y_train, y_test, image, label):
    clf = svm.SVC(max_iter=20000)
    clf.fit(X_train, y_train)

    predict_result = clf.predict(X_test)
    test_score = clf.score(X_test, y_test)
    print("SVM 테스트 예측 레이블:", predict_result[:10], "...")
    print("SVM 테스트 세트 정확도: {:.3f}".format(test_score))

    # --- for scaled SVM ---
    train_scaler    = MinMaxScaler().fit(X_train)
    test_scaler     = MinMaxScaler().fit(X_test)
    X_train_scaled  = train_scaler.transform(X_train)
    X_test_scaled   = test_scaler.transform(X_test)
    clf.fit(X_train_scaled, y_train)

    predict_scaled_result = clf.predict(X_test_scaled)
    test_score = clf.score(X_test_scaled, y_test)
    print("Scaled된 SVM 테스트 예측 레이블:", predict_result[:10], "...")
    print("Scaled된 SVM 테스트 세트 정확도: {:.3f}".format(test_score))

    # --- for cross validation ---
    clf.fit(image, label)  # re-fitting set for cross validation.
    cross_validation_score = cross_val_score(clf, image, label, cv=5)
    print("SVM의 5-fold Cross-Validation 점수: ", round(np.mean(cross_validation_score), 3))

    return predict_result, predict_scaled_result


def LSVM(X_train, X_test, y_train, y_test, image, label):
    clf = LinearSVC(max_iter=20000)
    clf.fit(X_train, y_train)

    predict_result = clf.predict(X_test)
    test_score = clf.score(X_test, y_test)
    print("LSVM 테스트 예측 레이블:", predict_result[:10], "...")
    print("LSVM 테스트 세트 정확도: {:.3f}".format(test_score))

    # --- for scaled LSVM ---
    train_scaler    = MinMaxScaler().fit(X_train)
    test_scaler     =    MinMaxScaler().fit(X_test)
    X_train_scaled  = train_scaler.transform(X_train)
    X_test_scaled   = test_scaler.transform(X_test)
    clf.fit(X_train_scaled, y_train)

    predict_scaled_result = clf.predict(X_test_scaled)
    test_score = clf.score(X_test_scaled, y_test)
    print("LSVM-S 테스트 예측 레이블:", predict_result[:10], "...")
    print("LSVM-S 테스트 세트 정확도: {:.3f}".format(test_score))

    # --- for cross validation ---
    clf.fit(image, label)  # re-fitting set for cross validation.
    cross_validation_score = cross_val_score(clf, image, label, cv=5)
    print("LSVM의 5-fold Cross-Validation 점수: ", round(np.mean(cross_validation_score), 3))

    return predict_result, predict_scaled_result


def result_to_csv(answer, MLP_P, MLPS_P, SVM_P, SVMS_P, LSVM_P, LSVMS_P):
    df = pd.DataFrame({'answer': answer[:],
                       'MLP_predict': MLP_P[:],
                       'MLPS_predict': MLPS_P[:],
                       'SVM_predict': SVM_P[:],
                       'SVMS_predict': SVMS_P[:],
                       'LSVM_predict': LSVM_P[:],
                       'LSVMS_predict': LSVMS_P[:]
                       })
    df.to_csv('Classification result and answer.csv', index=False, encoding='utf-8')
    print('결과를 성공적으로 .csv 파일로 저장하였습니다.')


if __name__ == '__main__':
    image, label, X_train, X_test, y_train, y_test, = preprocessing()

    MLP_predict, MLPS_predict = MLP(X_train, X_test, y_train, y_test, image, label)
    SVM_predict, SVMS_predict = SVM(X_train, X_test, y_train, y_test, image, label)
    LSVM_predict, LSVMS_predict = LSVM(X_train, X_test, y_train, y_test, image, label)

    # result_to_csv(y_test, MLP_predict, MLPS_predict, SVM_predict, SVMS_predict, LSVM_predict, LSVMS_predict)
