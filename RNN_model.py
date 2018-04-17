# -*- coding: utf-8 -*-
'''
二维数组(矩阵）为神经网络中的标准化接口
'''
import copy
import math
import random
random.seed(1)

#PCA降维
def PCA(inputs,d):
    # @inputs: 输入数据[samples,features]
    # @d:      降维后维度
    def zeroMean(inputs):
        #返回每个维度的平均值以及中心化后数据
        samples = len(inputs)
        features = len(inputs[0])
        means = [0 for i in range(features)]
        for sample in range(samples):
            for feature in range(features):
                means[feature] += inputs[sample][feature]
        for feature in range(features):
            means[feature] /= samples
        zeroData = matrixZerosLike(inputs)
        for sample in range(samples):
            for feature in range(features):
                zeroData[sample][feature] -= means[feature]
        return zeroData,means
    zeroData,means = zeroMean(inputs)


# activation function
def sigmoid(x):
    y = 1/(1+math.exp(-x))
    return y

def Relu(x):
    y = x if x>= 0 else 0
    return y

#gradent of sigmoid
def gradentOfSigmoid(y):
    return y*(1-y)

def gradentOfRelu(y):
    if y > 0:
        return 1
    else:
        return 0

def costFunc(y_true,y_pre):
    samples_num = len(y_pre)
    output_dim = len(y_pre[0])
    loss_square = [[0 for j in range(output_dim)] for i in range(output_dim)]
    for i in range(samples_num):
        for j in range(output_dim):
            loss_square[i][j] = math.pow((y_true[i][j] - y_pre[i][j]),2)
    loss_sum = 0
    for i in range(samples_num):
        loss_sum += sum(loss_square[i])
    loss = 0.5*loss_sum
    return loss


#矩阵加法
def matrixAdd(a,b):
    assert len(a) == len(b),"rows of matrix don't match"
    assert len(a[0]) == len(b[0]), "columns of matrix don'n match"
    ans = [[0 for j in range(len(a[0]))] for i in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            ans[i][j] = a[i][j] + b[i][j]
    return ans

#矩阵乘法
def matrixMultiply(A,B):
    assert len(A[0]) == len(B),"dimension of matrix doesn't match"
    ans = [[0 for i in range(len(B[0]))] for j in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(B)):
                sum += A[i][k]*B[k][j]
            ans[i][j] = sum
    return ans

#矩阵数乘
def matrixMultiplyNum(A,a):
    rows = len(A)
    columns = len(A[0])
    ans = matrixZerosLike(A)
    for i in range(rows):
        for j in range(columns):
            ans[i][j] = A[i][j]*a
    return ans

#矩阵点乘
def matrixDotMultiply(a,b):
    assert len(a) == len(b),"rows of matrix don't match"
    assert len(a[0]) == len(b[0]), "columns of matrix don'n match"
    ans = [[0 for j in range(len(a[0]))] for i in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            ans[i][j] = a[i][j] * b[i][j]
    return ans

#矩阵列和
def matrixSum(A):
    columns = len(A[0])
    sum = [0 for j in range(columns)]
    for row in A:
        for column in range(columns):
            sum[column] += row[column]
    return sum

#矩阵转置
def matrixTranpose(A):
    rows = len(A)
    columns = len(A[0])
    A_T = [[0 for i in range(rows)] for j in range(columns)]
    for i in range(columns):
        for j in range(rows):
            A_T[i][j] = A[j][i]
    return A_T

#生成相同shape的零矩阵
def matrixZerosLike(A):
    rows = len(A)
    columns = len(A[0])
    ans = [[0 for j in range(columns)] for i in range(rows)]
    return ans

def weights_init(input_dim,output_dim):
    #用来生成大小合适的初始化矩阵
    #满足均值为0，方差为2/(input_dim+output_dim)的均匀分布
    low = -math.sqrt(6.0/(input_dim + output_dim))
    high = math.sqrt(6.0/(input_dim + output_dim))
    weights = [[random.uniform(low,high) for j in range(output_dim)] for i in range(input_dim)]
    return weights



class DenseLayer(object):
    def __init__(self,neuron_num,inputs_dim,active_func = 'sigmoid',name = "DenseLayer",weights = None,bias = None):
        self.inputs_dim = inputs_dim    #输入维度，即特征数
        self.neuron_num = neuron_num    #神经元个数，即当前层输出
        self.active_func = active_func  #激活函数，默认是sigmoid
        self.weights = weights          # 权值矩阵[inputs_dim,neuron_num]
        self.bias = bias                #偏置[1,neuron_num]
        self.name = name
        if weights == None:
            self.weights = weights_init(inputs_dim,neuron_num)
        if bias == None:
            self.bias = [[0.1 for j in range(neuron_num)]]



    def forward(self,inputs):
        assert len(inputs[0]) == len(self.weights), 'inputs of '+ self.name +"don't match the dimension of weights"
        sample_num = len(inputs)
        bias = self.bias
        if sample_num >1:   #batchSize>1做加法需要broadcast
            bias = [self.bias[0] for i in range(sample_num)]
        temp = matrixAdd(matrixMultiply(inputs,self.weights),bias)
        if self.active_func == 'sigmoid':
            outputs = [map(sigmoid, temp[i]) for i in range(sample_num)]
        if self.active_func == 'Relu':
            outputs = [map(Relu, temp[i]) for i in range(sample_num)]
        if self.active_func == 'None':
            outputs = temp[:]
        return outputs  #[sample_num,neuron_num]

    def get_weights(self):
        return copy.deepcopy(self.weights)
    def get_bias(self):
        return copy.deepcopy(self.bias)
    def set_weights(self,weights):
        assert len(self.weights) == len(weights) and len(self.weights[0]) == len(weights[0]),"dimension of weights doesn't match"
        self.weights = copy.deepcopy(weights)
    def set_bias(self,bias):
        self.bias = copy.deepcopy(bias)


class RnnLayer(object):
    def __init__(self,neuron_num,inputs_dim,active_func = 'sigmoid',name = 'RNNLayer',weights0 = None,weights1 = None,bias = None):
        self.neuron_num = neuron_num
        self.inputs_dim = inputs_dim
        self.weights0 = weights0        #与输入连接的权值矩阵
        self.weights1 = weights1        #与当前层前一时刻输出连接的权值矩阵
        self.bias = bias
        self.active_func = active_func
        self.name = name
        if weights0 == None:
            self.weights0 = weights_init(inputs_dim,neuron_num)
        if weights1 == None:
            self.weights1 = weights_init(neuron_num,neuron_num)
        if bias == None:
            self.bias = [[0.1 for j in range(neuron_num)]]

    def forward(self,inputs,outputs_before):
        assert len(inputs[0]) == len(self.weights0), 'inputs of ' + self.name + "don't match the dimension of weights"
        sample_num = len(inputs)
        bias = self.bias
        if sample_num >1:   #batchSize>1做加法需要broadcast
            bias = [self.bias[0] for i in range(sample_num)]
        temp = matrixAdd(matrixAdd(matrixMultiply(inputs,self.weights0),matrixMultiply(outputs_before,self.weights1)),bias)
        outputs = temp[:]
        if self.active_func == 'sigmoid':
            outputs = [map(sigmoid,temp[i]) for i in range(sample_num)]
        if self.active_func == 'Relu':
            outputs = [map(Relu, temp[i]) for i in range(sample_num)]
        if self.active_func == 'None':
            pass
        return outputs       #[sample_num,neuron_num]

    def get_weights0(self):
        return copy.deepcopy(self.weights0)
    def get_weights1(self):
        return copy.deepcopy(self.weights1)
    def get_bias(self):
        return copy.deepcopy(self.bias)
    def set_weights0(self,weights):
        self.weights0 = copy.deepcopy(weights)
    def set_weights1(self,weights):
        self.weights1 = copy.deepcopy(weights)
    def set_bias(self,bias):
        self.bias = copy.deepcopy(bias)

def getDataSet(DataSet,flavorToUse,flavorToPredict,dayToPredict,timeSpan):
    # @DataSet: trainSet中读到的数据集，二维数组
    # @flavorToUse: 构造特征时考虑的flavor数
    # @flavorToPredict: 需要预测的规格，一维数组
    # @timeSpan：构造特征时考虑当天以前的时间跨度
    # @DayToPredict:需要预测的天数
    # @DataX: 前timeSpan天各个规格的数量和，二维数组，二维数组为标准化接口
    # @DataY: 当天后需要预测天数内各flavor规格的数量和，二维数组
    days = len(DataSet)
    DataSet = [DataSet[day][:flavorToUse] for day in range(days)]
    samples = days - timeSpan - dayToPredict + 1 #能够构造的样本数
    DataX = []; DataY = []
    for day in range(samples):
        today = day + timeSpan
        X = matrixSum(DataSet[day:today])
        DataX.append(X)
        Y = matrixSum([[DataSet[i][flavor-1] for flavor in flavorToPredict] for i in range(today,today+dayToPredict)])
        DataY.append(Y)
    return DataX, DataY


class RNN_network(object):
    def __init__(self,inputs_dim,hidden_num,outputs_dim):
        self.hidden_layer = RnnLayer(neuron_num=hidden_num,inputs_dim = inputs_dim,active_func="Relu")  #循环层
        self.output_layer = DenseLayer(neuron_num=outputs_dim,inputs_dim = hidden_num,active_func = "Relu") #输出全连接层

    def train(self,epochs,DataX,DataY,learning_rate = 0.5,lookBack = 7,optimizer = 'SGD'):
        samples = len(DataX)      #样本数
        train_num = 0
        for epoch in range(epochs):
            iter_num = samples - lookBack  #每一个epoch迭代的次数
            for it in range(iter_num):
                #定义更新偏导值
                hidden_weights0_update = matrixZerosLike(self.hidden_layer.weights0)
                hidden_weights1_update = matrixZerosLike(self.hidden_layer.weights1)
                hidden_bias_update = matrixZerosLike(self.hidden_layer.bias)
                output_weights_update = matrixZerosLike(self.output_layer.weights)
                output_bias_update = matrixZerosLike(self.output_layer.bias)

                overall_loss = 0
                output_layer_deltas = []  #输出层每个时刻的偏导
                hidden_layer_values = []  #隐层每个时刻的值
                hidden_layer_values.append([0 for _ in range(self.hidden_layer.neuron_num)]) #0时刻隐层初始值设为0
                for day in range(lookBack):
                    X = [DataX[it + day]]
                    Y = [DataY[it + day]]
                    hidden_layer_output = self.hidden_layer.forward(X,[hidden_layer_values[-1]])
                    hidden_layer_values.append(copy.deepcopy(hidden_layer_output[0]))
                    output_layer_output = self.output_layer.forward(hidden_layer_output)
                    temp_loss = costFunc(Y,output_layer_output)   #当前样本损失值
                    overall_loss += temp_loss
                    error = [[output_layer_output[0][i] - Y[0][i] for i in range(len(Y[0]))]]
                    output_layer_deltas.append(map(gradentOfRelu,error[0]))
                future_hidden_delta = [[0 for i in range(self.hidden_layer.neuron_num)]] #  t+1时刻隐层偏导
                for day in range(lookBack):
                    #从后往前基于时间的BP算法
                    X = [DataX[it+lookBack-day-1]] #当前时刻输入
                    hidden_layer_output = hidden_layer_values[-day-1] #当前时刻隐层输出
                    pre_hidden_output = hidden_layer_values[-day-2]   #前一时刻隐层输出
                    output_layer_delta = [output_layer_deltas[-day-1]]#当前时刻输出层的偏导
                    hidden_layer_delta = matrixDotMultiply(matrixAdd(matrixMultiply(future_hidden_delta,matrixTranpose(self.hidden_layer.weights1)),
                                  matrixMultiply(output_layer_delta,matrixTranpose(self.output_layer.weights))),
                                                           [map(gradentOfRelu,hidden_layer_output)])
                    output_weights_update = matrixAdd(output_weights_update,matrixMultiply(matrixTranpose([hidden_layer_output]),output_layer_delta)) #时间维度上的偏导累加
                    output_bias_update = matrixAdd(output_bias_update,output_layer_delta)
                    hidden_weights0_update = matrixAdd(hidden_weights0_update,matrixMultiply(matrixTranpose(X),hidden_layer_delta))
                    hidden_weights1_update = matrixAdd(hidden_weights1_update,matrixMultiply(matrixTranpose([pre_hidden_output]),hidden_layer_delta))
                    hidden_bias_update = matrixAdd(hidden_bias_update,hidden_layer_delta)
                    future_hidden_delta = hidden_layer_delta  #偏导在时间维度向前传播
                #梯度下降法更新参数
                hidden_weights0 = matrixAdd(self.hidden_layer.get_weights0(),
                                            matrixMultiplyNum(hidden_weights0_update,-1*learning_rate))
                self.hidden_layer.set_weights0(hidden_weights0)
                hidden_weights1 = matrixAdd(self.hidden_layer.get_weights1(),
                                            matrixMultiplyNum(hidden_weights1_update, -1*learning_rate))
                self.hidden_layer.set_weights1(hidden_weights1)
                hidden_bias = matrixAdd(self.hidden_layer.get_bias(),
                                            matrixMultiplyNum(hidden_bias_update, -1*learning_rate))
                self.hidden_layer.set_bias(hidden_bias)
                output_weights = matrixAdd(self.output_layer.get_weights(),
                                            matrixMultiplyNum(output_weights_update, -1*learning_rate))
                self.output_layer.set_weights(output_weights)
                output_bias = matrixAdd(self.output_layer.get_bias(),
                                        matrixMultiplyNum(output_bias_update,-1*learning_rate))
                self.output_layer.set_bias(output_bias)

                train_num += 1
                if train_num%10 == 0:
                    print "epoch: {}, train_num: {}, Total Average loss: {}".format(epoch+1,train_num,overall_loss/lookBack)
    def predict(self,DataX):
        days = len(DataX)
        hidden_layer_values = []
        hidden_layer_values.append([0 for _ in range(self.hidden_layer.neuron_num)])
        for day in range(days):
            hidden_layer_output = self.hidden_layer.forward([DataX[day]],[hidden_layer_values[-1]])
            hidden_layer_values.append(hidden_layer_output[0])
            output = self.output_layer.forward(hidden_layer_output)
        return output










