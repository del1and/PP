# handwritten digit image recognition code(2020/11~2020/12)

images are selfmade 28*28 pngs. (special thanks: friends, family for make digit images)
use sklearn for modeling, pandas to make csv, PIL for RGB to gray.

model: SVM, LSVM, MLP

1. 개요: 여러 숫자가 적힌 큰 이미지를 28x28픽셀의 png로 분리 후 클래스를 지정, 훈련하는 모델
2. 제작 의도: Kaggle 등 공개적으로 배포되고 있는 데이터가 아닌 직접 수집한 데이터를 가지고 숫자 필기 구분 모델을 구현해보고자 함.

3. 데이터셋: 태블릿, 스마트폰, 종이 필기 등으로 이뤄진 숫자 필기 데이터를 수집, 숫자에 해당하는 부분만 남기고 모두 하얗게 처리한 28x28 사이즈의 1,211개의 png 이미지

![digit image](https://user-images.githubusercontent.com/62870912/119866754-1e38d380-bf58-11eb-8797-0d47e99ebb19.PNG)

5. 전처리: 모든 데이터를 불러와 색 반전을 시키고(배경이 255이므로 0으로 바꿔줌), (R+G+B)/3을 통해 흑백화. 이후 모두 정답 레이블이 존재하므로, sklearn을 통해 train, test set split
6. 결과

![ML image](https://user-images.githubusercontent.com/62870912/119866834-314ba380-bf58-11eb-997f-5edace7a6675.png)
