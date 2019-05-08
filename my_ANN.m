function [W, error, count] = my_ANN(node, M, X, T, alpha)
    % -----����-----
    % node: ÿ��ڵ����������
    % M: ����Ĳ���, �����ǵ�һ�㣬���һ���ǵ�M-1��
    % X: ������������
    % T: ������ݣ�������
    
    [~, P] = size(X);
    % �������������Ϊ����nΪ����ά�ȣ�p�������ĸ���
    % ����ÿ����֮�����Ԫ֮��ͻ����һ��Ȩ�����Ӿ���
    % ��˲���cell��ʽ���洢��������֮�������Ȩ�ؾ���
    W = cell(1, M-1);
    for i = 1:M-1
        % cell �е�ÿһ��Ԫ����һ�����󣬴����ŵ�i�㵽��i+1�������Ȩ��
        W{1, i} = rand(node(i) + 1, node(i+1));
    end
    
    % ����output_NeuronԪ��������¼ÿһ����Ԫ�����(��һ����Ϊ�����)
    % ����P�о���ζ�ż�¼P����������Ӧÿ����Ԫ�����
    output_Neuron = cell(P, M);
    for i = 1:P
        % ÿ������ĸ���Ӧ�õ���
        for j = 1:M
            output_Neuron{i, j} = zeros(1, node(j) + 1);
            % ÿ��ĵ�һ��Ԫ��ǿ����Ϊ1����Ϊƫ��
            output_Neuron{i, j}(1) = 1;
            % ��һ��ֱ�ӽ���������
            if j == 1
                output_Neuron{i, j}(2:end) = X(: ,i).';
            end
        end
    end
    error = 0;
    count = 0;
    epsilon = 10^-5;
    % -----ǰ����������Ԫ�����ֵ-----
    % ����ÿһ������Ҫ��������
    for p = 1:P
        for j = 2:M
            output_Neuron{p, j}(2:end) = sigmoid(output_Neuron{p, j-1}*W{1, j-1});
            if j == M
                error = error + 0.5*sum((output_Neuron{p, j}(2:end) - T(:, p).').^2);
            end
        end
    end
    
    while (error >=epsilon)&&(count <= 1e5)
        
        % -----BP�㷨����Ȩֵ�仯��-----
        % ��Ҫ������W��ά����ȫһ�µ�Ԫ������d_W
        d_W = cell(1, M-1);
        for i = 1:M-1
            d_W{1, i} = zeros(node(i) + 1, node(i+1));
        end

        % ����BP������Ҫ��һ���õ��� delta_(p,j)^(k+1) = - partial E_p / partial y_(p,j)^(k+1)
        delta_table = cell(P, M-1);
        % ���ȼ������һ�������Ȩ�仯��
        % ���þ�����˵���ʽֱ�ӵõ����ڵ�p��������ȫ��delta_(p,j)^(M-1)�Լ�y_(p,i)^(M-2)�ĳ˻�ֵ
        for p = 1:P
            output_last_1 = output_Neuron{p, M}(2:end);
            delta = (T(:, p)' - output_last_1).*output_last_1.*(1 - output_last_1);
            delta_table{p, M-1} = delta;
            output_last_2 = output_Neuron{p, M-1};
            d_W{1, M-1} = d_W{1, M-1} + output_last_2.'*delta;
        end

        %���㵹���ڶ�����ǰ��ȫ�����Ӿ��������
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
        
        % -----����Ȩֵ-----
        for i = 1:M-1
            W{1, i} = W{1, i} + alpha*d_W{1, i};
        end
        % ------------------
        
        % -----�ٴ�ǰ����������Ԫ�����ֵ-----
        error = 0;
        % ����ÿһ������Ҫ��������
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