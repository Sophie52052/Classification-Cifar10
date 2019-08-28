# 主題：Machine Learning 講者：陳育德
###### tags: `MachineLearning` `SVM` `Precition` `Recall` `F1score` `Accuray` `Loss` `Kerneltrick`
陳育德
> https://docs.google.com/a/iir.csie.ncku.edu.tw/viewer?a=v&pid=sites&srcid=aWlyLmNzaWUubmNrdS5lZHUudHd8aWlyLWxhYnxneDo3NjYyZDdjODViNDI4MGI0
> 
* 步驟
![](https://i.imgur.com/0CFnWIu.png)

* 例子
![](https://i.imgur.com/pX5Z9qY.png)

* 資料特性
![](https://i.imgur.com/PwnO8WB.png)
> 序列、時間、像素之間關連
> Clasfication：Multiclass-1個 Multilabel-2個以上
> 
* One-Hot-Encoding
1.原因：機器學習任務中，特徵並不是連續值，有可能是分類值
![](https://i.imgur.com/XDyw5lX.png)
上述資料不能直接用在我們的分類器中
因**分類器**往往預設資料資料是**連續**的、**有序**的
上述非有序，而是隨機分配的

  2.：使用N位狀態暫存器對N個狀態進行編碼，每個狀態都有它獨立的暫存器位，並且在任意時候，都只有一位有效
  ![](https://i.imgur.com/xLaluIX.png)
  
  3.例子
  ![](https://i.imgur.com/54TFm2t.png)
  4個特徵：
第一個特徵（第一行）為[0,1,2,1] ，其中三類特徵值[0,1,2]，因此One-Hot Code可將[0,1,2]表示為:[100,010,001]
同理第二個特徵列可將兩類特徵值[2,3]表示為[10,01]
第三個特徵將4類特徵值[1,2,4,5]表示為[1000,0100,0010,0001]
第四個特徵將2類特徵值[3,12]表示為[10,01]
因此最後可將[2,3,5,3]表示為[0,0,1,0,1,0,0,0,1,1,0]

  4.其他編碼
  ![](https://i.imgur.com/oJ7Z9Od.png)

* 標準化
 1.定義：兩者不同的尺寸和規格資料，使兩規格相同
 2.方法：往往按照某個統一的標準（如統一的內部構成）進行修正
 ![](https://i.imgur.com/mWYGExs.png)
 3.例子1：ZSCORE-代表著原始分數和母體平均值之間的距離，是以標準差為單位計算。在原始分數低於平均值時Z則為負數，反之則為正數。換句話說，Z值是從感興趣的點到均值之間有多少個標準差。
 ![](https://i.imgur.com/LI2G9Jy.png)
 4.例子2：feature scaling（特徵縮放）--將所選特徵的value都縮放到一個大致相似的範圍。目的是為了加快收斂，減少採用梯度下降算法迭代的次數

* Over- & under-fitting
 EX.總共有A~F，我喜歡ABCDE五樣產品，你推薦ABF給我
只看正確不考慮錯誤--Precition、Recall
Precition：推薦個數中分之幾個你要(2/3)
Recall：所有喜歡的個數分之猜中幾個(2/5)
考慮錯誤--Accuray
Accuray：全部分之判對幾個(2/6)

  F1 Score：![](https://i.imgur.com/EwhQY8N.png)
  
* Accuray & Loss
![](https://i.imgur.com/ttNqnYt.png)
1.Loss--模型預測值與真實值之間的差異，最常用的損失函數是交叉熵![](https://i.imgur.com/75uNB8U.png)![](https://i.imgur.com/30PMMNu.png)
2.Accuray--準確度是衡量模型性能的指標之一![](https://i.imgur.com/jv1ctLx.png)
3.大多數情況下，您會觀察到準確度隨著損失的減少而增加
![](https://i.imgur.com/3wowwX4.png)
因**準確性**和**損失**是測量兩種不同的東西
交叉熵損失將**損失降低**到更接近類別標籤的預測
準確度是**特定樣本的二進制真/假**，離散的
這裡的Loss是一個**連續**變量，即當預測接近1（對於真實標籤）和接近0（對於假標籤）時最好
> 模型：一種具有高精度和高損耗，另一種具有低精度和低損耗
選擇方法：基於關注什麼議題
> 

* Curse of dimensionality(維數災難)
1.意義：通常在越高維度越能分類的越好，易導致過度擬和(over-fitting)或。正確數據無法覆蓋整個特徵空間，這樣得到的分類器在對新數據進行預測時將會出現錯誤
PS.高維帶來的數據稀疏性問題
![](https://i.imgur.com/OyaLEbs.png)
隨著維度的增加，分類器性能逐步上升，到達某點之後，其性能便逐漸下降
2.例子：狗貓分類
一種特徵(一維)EX.紅色![](https://i.imgur.com/0pefJJX.png)
二種特徵(二維)EX.紅色、綠色![](https://i.imgur.com/xeDMeRs.png)
三種特徵(三維)![](https://i.imgur.com/GYe1E3d.png)
多維 找到最佳分類平面![](https://i.imgur.com/hjWExHx.png)
將高維空間向低維空間投影![](https://i.imgur.com/BUl3B6c.png)
過多特徵導致過度擬合：訓練良好，但對新數據缺乏泛化能力(機器學習演算法對新樣本的適應力)
3.高維數據稀疏問題
![](https://i.imgur.com/zdleTG1.png)
如果一直增加特徵維數，由於樣本分佈越來越稀疏，如果要避免過擬合的出現，就不得不持續增加樣本數量

* 參數(parameters)超參數(hyperparameters)
1.參數：模型根據數據自己學習出的變量。EX.深度學習的權重(Weight)，偏差(Bias)等
2.超參數：起始參數用來確定模型，一般就是根據經驗確定的變量(基於不同模型有區別，EX.CNN，層數不同則不同)。EX.學習速率Leraning Rate），迭代次數(epoch)，層數，每層神經元的個數等

* 交叉驗證(k-fold cross-validation)
1.意義：將樣本切割K個小子集，訓練階段定義一組用於「測試」模型的數據集，以便減少像過擬合的問題
2.作法：可以先在一個子集(訓練集train)上做分析，而其它子集則用來做後續對此分析的確認及驗證(測試集test)

* SVM
 1.意義：監督式學習，用統計法來估計一個分類的超平面(hyperplane)。找到一個決策邊界(decision boundary)讓兩類之間的邊界(margins)最大化，使其可以完美區隔開來。SVM就是在找參數w和b
![](https://i.imgur.com/U2b6c5H.png)
![](https://i.imgur.com/NbSqDJY.png)
![](https://i.imgur.com/0RvKCSz.png)

* LR & SVM
1.如果不考慮核函式，都是線性分類演算法、監督學習演算法，也就是說他們的分類決策面都是線性的
2.loss function不同
![](https://i.imgur.com/uGl2mV3.png)
邏輯迴歸方法：基於概率理論，假設樣本為1的概率可以用sigmoid函式來表示，然後通過極大似然估計的方法估計出引數的值http://blog.csdn.net/pakko/article/details/37878837
支援向量機：基於幾何間隔最大化原理，認為存在最大幾何間隔的分類面為最優分類面，具體細節參考http://blog.csdn.net/macyang/article/details/38782399
> https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E6%94%AF%E6%92%90%E5%90%91%E9%87%8F%E6%A9%9F-support-vector-machine-svm-%E8%A9%B3%E7%B4%B0%E6%8E%A8%E5%B0%8E-c320098a3d2e
> 
* Kernel trick
 1.意義：希望當不同類別的資料在原始空間中無法被線性分類器區隔開來時，經由非線性投影後的資料能在更高維度的空間中可以更區隔開
 ![](https://i.imgur.com/5hXUJbj.png)










