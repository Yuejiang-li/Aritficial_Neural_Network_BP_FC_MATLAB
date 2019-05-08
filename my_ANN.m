function [W, error, count] = my_ANN(node, M, X, T, alpha)
    % -----参数-----
    % node: 每层节点度数的序列
    % M: 网络的层数, 输入是第一层，最后一层是第M-1层
    % X: 输入数据向量
    % T: 输出数据（向量）
    
    [~, P] = size(X);
    % 输入的数据向量为竖向，n为向量维度，p是向量的个数
    % 由于每两层之间的神经元之间就会存在一个权重连接矩阵
    % 因此采用cell形式来存储相邻两层之间的链接权重矩阵
    W = cell(1, M-1);
    for i = 1:M-1
        % cell 中的每一个元素是一个矩阵，代表着第i层到第i+1层的链接权重
        W{1, i} = rand(node(i) + 1, node(i+1));
    end
    
    % 采用output_Neuron元胞组来记录每一层神经元的输出(第一层作为输入层)
    % 共有P行就意味着记录P个样本所对应每个神经元的输出
    output_Neuron = cell(P, M);
    for i = 1:P
        % 每层输出的个数应该等于
        for j = 1:M
            output_Neuron{i, j} = zeros(1, node(j) + 1);
            % 每层的第一个元素强行置为1，作为偏置
            output_Neuron{i, j}(1) = 1;
            % 第一层直接将输入填入
            if j == 1
                output_Neuron{i, j}(2:end) = X(: ,i).';
            end
        end
    end
    error = 0;
    count = 0;
    epsilon = 10^-5;
    % -----前向计算各个神经元的输出值-----
    % 对于每一个数据要单独计算
    for p = 1:P
        for j = 2:M
            output_Neuron{p, j}(2:end) = sigmoid(output_Neuron{p, j-1}*W{1, j-1});
            if j == M
                error = error + 0.5*sum((output_Neuron{p, j}(2:end) - T(:, p).').^2);
            end
        end
    end
    
    while (error >=epsilon)&&(count <= 1e5)
        
        % -----BP算法计算权值变化量-----
        % 需要开辟与W的维度完全一致的元胞数组d_W
        d_W = cell(1, M-1);
        for i = 1:M-1
            d_W{1, i} = zeros(node(i) + 1, node(i+1));
        end

        % 由于BP过程需要上一步得到的 delta_(p,j)^(k+1) = - partial E_p / partial y_(p,j)^(k+1)
        delta_table = cell(P, M-1);
        % 首先计算最后一层的链接权变化量
        % 采用矩阵相乘的形式直接得到对于第p个样本的全部delta_(p,j)^(M-1)以及y_(p,i)^(M-2)的乘积值
        for p = 1:P
            output_last_1 = output_Neuron{p, M}(2:end);
            delta = (T(:, p)' - output_last_1).*output_last_1.*(1 - output_last_1);
            delta_table{p, M-1} = delta;
            output_last_2 = output_Neuron{p, M-1};
            d_W{1, M-1} = d_W{1, M-1} + output_last_2.'*delta;
        end

        %计算倒数第二层以前的全部链接矩阵的增量
        for k = M-2:-1:1
            for p = 1:P
                output_this_layer = output_Neuron{p, k+1}(2:end);
                temp = delta_table{p, k+1}*W{k+1}(2:end, :).';
                delta = temp.*output_this_layer.*(1 - output_this_layer);
                delta_table{p, k} = delta;
                output_last_layer = output_Neuron{p, k};
                d_W{1, k} = d_W{1, k} + output_last_layer.'*delta;
            end
        end
        % ---------------------------------------------------
        
        % -----更新权值-----
        for i = 1:M-1
            W{1, i} = W{1, i} + alpha*d_W{1, i};
        end
        % ------------------
        
        % -----再次前向计算各个神经元的输出值-----
        error = 0;
        % 对于每一个数据要单独计算
        for p = 1:P
            for j = 2:M   
                output_Neuron{p, j}(2:end) = sigmoid(output_Neuron{p, j-1}*W{1, j-1});
                if j == M
                    error = error + 0.5*sum((output_Neuron{p, j}(2:end) - T(:, p).').^2);
                end
            end
        end
        count = count + 1;
    end
    
end