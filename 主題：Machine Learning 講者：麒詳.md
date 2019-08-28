# 主題：Machine Learning 講者：麒詳
###### tags: `MachineLearning`  `KNN` `KMeans` `SVM` `NaiveBayes` `Likelihood` `DecisionTree` `Regression` `Sigmoid` `Entropy`

* KNN(K-近鄰演算法)
1.K：自定義的常數，決定要參考K個周圍
2.作法：為一個沒有類別標籤的向量（查詢或測試點）找出和新數據附近的K個鄰居（資料），將其分類（標籤）

* K Means(k-平均演算法)
 1.非監督式學習
 2.意義：物以類聚、內聚力
 3.作法：
 Radom起始點或Set
 先預設要分成幾(k)群，沒有LABLE
 在feature space隨機給k個群心![](https://i.imgur.com/g40XTXy.png)
每筆資料都會和所有k個群心算歐式距離(直線距離)
將每筆資料分類到最近的群心![](https://i.imgur.com/OxG4Trb.png)
每個群心內都會有被分類過來的資料，用這些資料更新一次新的群心
重複3–5，直到所有群心不在有太大的變動(收斂)，結束
![](https://i.imgur.com/PFujCM2.png)

* Naive Bayes(貝式分類器)
1.前提：貝葉斯定理--使用條件機率：P（A|B）=P(B|A)×P(A)/P(B)
2.likelihood：https://wangcc.me/LSHTMlearningnote/likelihood-definition.html
![](https://i.imgur.com/xLEA0EJ.png)
![](https://i.imgur.com/Xg7k283.png)
3.例子
![](https://i.imgur.com/xoOsyya.png)
PS.機率不可為0，必須給他一個分子(因資料都是樣本非母群體)EX.黃色
![](https://i.imgur.com/Jyhi9yR.png)
50/150 * 70/230 * 250/500 * 190/191 * 410/660 * 270/530
(*390/1000)正 or (*610/1000)負

* Decision Tree(決策數)
1.資料必須是離散型
2.機率發生越小的事情-資訊量越大IG(E) = − logP(E)
3.Entropy(商)：亂度(好不好預測)(穩不穩定)，越大則越好預測(但也代表不確定性大)，可決定決策樹支點順序，PS.unfire好分
![](https://i.imgur.com/MST3CMj.png)

* Logistic Regression(羅吉斯回歸)
1.線性回歸：用來預測一個連續的值，羅吉斯回歸：用來分類
![](https://i.imgur.com/eIEgM9r.png)  ![](https://i.imgur.com/6TZGRK0.png)
![](https://i.imgur.com/QxQjSPI.png)
2.概念：將點帶進去回歸線，回歸線輸出值若是>=0，是一類(target)，值<0是另一類(non-target)
![](https://i.imgur.com/SIoxmgG.png)
![](https://i.imgur.com/NTbObKU.png)結果為0&1
PS.越靠近0的位置，理論上越容易分類錯誤，因此我們加上一個對數函數，使輸出就更有彈性，羅吉斯回歸用到的對數函數--Sigmoid
3.Sigmoid：返回的輸出值在0到1的範圍內
![](https://i.imgur.com/RvHBdAc.png)
![](https://i.imgur.com/7MFAxxI.png)

* Support Vector Machine
1.目標：找出一個超平面(hyperplane)，使之將兩個不同的集合分開
2.效果：希望這條線距離這兩個集合的邊界(margin)越大越好，這樣才能夠明確分辨某點是屬於何種集合，否則在計算上容易因精度產生誤差
3.Support Hyperplane![](https://i.imgur.com/ThA4hen.png)
指與 optimal separating hyperplane 平行，並且最靠近兩邊的超平面
虛線：optimal separating hyperplane 實線：support hyperplane
![](https://i.imgur.com/YKtRF9z.png)
![](https://i.imgur.com/yU5aM8N.png)
4.Using Kernel Function ex: linear, Polynomial, RBF
PS跟回歸比較，一個是線性、一個非線性
![](https://i.imgur.com/rrcPIwK.png)![](https://i.imgur.com/WUD94mp.png)

>
>https://docs.google.com/a/iir.csie.ncku.edu.tw/viewer?a=v&pid=sites&srcid=aWlyLmNzaWUubmNrdS5lZHUudHd8aWlyLWxhYnxneDozOWM0OGI5ZWZkODEwYWMy