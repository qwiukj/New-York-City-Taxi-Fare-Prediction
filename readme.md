## 作者
**刘宇为**

## 文件说明
- train.csv - fare_amount训练集的输入要素和目标值（约55M行）。
- test.csv - 测试集的输入功能（约10K行）。您的目标是预测fare_amount每一行。
- sample_submission.csv - 格式正确的样本提交文件（列key和fare_amount）。这个文件'预测' 为所有行的fare_amount$ 11.35，这是fare_amount训练集的平均值。

## 数据字段
### ID
- key - 唯一string标识训练集和测试集中的每一行。由pickup_datetime加上一个唯一的整数组成，但这没关系，它应该只用作唯一的ID字段。您的提交CSV中需要。在训练集中不一定需要，但在训练集内进行交叉验证时可以模拟“提交文件”。

### 特征
- **pickup_datetime**:timestamp表示出租车开始的时间的值。
- **pickup_longitude**:float用于出租车开始的经度坐标。
- **pickup_latitude**:float用于出租车开始的纬度坐标。
- **dropoff_longitude**:float用于出租车行程结束的经度坐标。
- **dropoff_latitude**:float用于出租车行程结束的纬度坐标。
- **passenger_count**:integer表示出租车乘客的数量。

### 目标
- **fare_amount**:float乘坐出租车的费用的美元金额。该值仅在训练集中; 这是您在测试集中预测的内容，并且在您的提交CSV中需要它。