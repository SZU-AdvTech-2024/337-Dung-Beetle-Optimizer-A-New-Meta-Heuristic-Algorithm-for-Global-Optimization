function [fMin, bestX, Convergence_curve] = IDBO(pop, M, c, d, dim, fobj)
    P_percent = 0.2; % 生产者种群大小占总种群大小的百分比
    pNum = round(pop * P_percent); % 生产者种群大小
    crossover_rate = 0.9; % 交叉率
    step_size = 0.01; % 梯度下降步长

    lb = c .* ones(1, dim); % 下界
    ub = d .* ones(1, dim); % 上界

    % 初始化种群
    x = lb + (ub - lb) .* rand(pop, dim);
    fit = arrayfun(@(i) fobj(x(i, :)), 1:pop);

    pFit = fit; % 个体最优适应度
    pX = x; % 个体最优位置
    XX = pX; % 用于更新的种群

    [fMin, bestI] = min(fit); % 全局最优适应度
    bestX = x(bestI, :); % 全局最优位置

    Convergence_curve = zeros(1, M); % 收敛曲线

    % 开始迭代更新
    for t = 1:M
        [fmax, B] = max(fit);
        worse = x(B, :); % 最差位置

        % 动态调整太阳光引导系数
        a_coeff = 1 - (t / M)^2;

        for i = 1:pNum
            % 交叉操作
            idx = randi([1, pop]);
            child = crossover(pX(i, :), pX(idx, :), crossover_rate);
            child = Bounds(child, lb, ub);
            child_fit = fobj(child);
            
            % 局部搜索
            local_sol = local_search(pX(i, :), fobj, lb, ub, step_size);
            local_fit = fobj(local_sol);
            
            % 选择更好的解
            if child_fit < fit(i) || local_fit < fit(i)
                if child_fit < local_fit
                    x(i, :) = child;
                    fit(i) = child_fit;
                else
                    x(i, :) = local_sol;
                    fit(i) = local_fit;
                end
            end
            
            % 更新个体最优
            if fit(i) < pFit(i)
                pFit(i) = fit(i);
                pX(i, :) = x(i, :);
            end
        end

         % 更新全局最优和个体最优
        [fMMin, bestII] = min(fit);
        bestXX = x(bestII, :);

        R = 1 - t / M;
        Xnew1 = bestXX .* (1 - R);
        Xnew2 = bestXX .* (1 + R);
        Xnew1 = Bounds(Xnew1, lb, ub);
        Xnew2 = Bounds(Xnew2, lb, ub);

        Xnew11 = bestX .* (1 - R);
        Xnew22 = bestX .* (1 + R);
        Xnew11 = Bounds(Xnew11, lb, ub);
        Xnew22 = Bounds(Xnew22, lb, ub);

        % 更新其他解
        for i = (pNum + 1):pop
            if i <= 12
                x(i, :) = bestXX + (rand(1, dim) .* (pX(i, :) - Xnew1)) + (rand(1, dim) .* (pX(i, :) - Xnew2));
                x(i, :) = Bounds(x(i, :), Xnew1, Xnew2);
            elseif i <= 19
                x(i, :) = pX(i, :) + (randn(1) .* (pX(i, :) - Xnew11)) + (rand(1, dim) .* (pX(i, :) - Xnew22));
                x(i, :) = Bounds(x(i, :), lb, ub);
            else
                x(i, :) = bestX + randn(1, dim) .* ((abs((pX(i, :) - bestXX)) + abs((pX(i, :) - bestX))) /2);
                x(i, :) = Bounds(x(i, :), lb, ub);
            end
            fit(i) = fobj(x(i, :));
        end

        % 更新个体最优和全局最优
        for i = 1:pop
            if fit(i) < pFit(i)
                pFit(i) = fit(i);
                pX(i, :) = x(i, :);
            end
            if pFit(i) < fMin
                fMin = pFit(i);
                bestX = pX(i, :);
            end
        end

        XX = pX; % 更新用于更新的种群

        % 更新收敛曲线
        Convergence_curve(t) = fMin;
    end
end

% 交叉操作函数
function child = crossover(parent1, parent2, crossover_rate)
    if rand() < crossover_rate
        % 随机选择单点交叉或多点交叉
        if rand() < 0.5
            % 单点交叉
            idx = randi(length(parent1));
            child = [parent1(1:idx), parent2(idx+1:end)];
        else
            % 多点交叉
            % 随机选择两个不同的交叉点
            idx1 = randi(length(parent1));
            idx2 = randi(length(parent1));
            % 确保idx1 < idx2
            if idx1 > idx2
                [idx1, idx2] = deal(idx2, idx1);
            end
            % 执行多点交叉
            child = [parent1(1:idx1), parent2(idx1+1:idx2), parent1(idx2+1:end)];
        end
    else
        % 不进行交叉，直接复制父代
        child = parent1;
    end
end


% 局部搜索函数
function x = local_search(x, fobj, lb, ub, step_size)
    grad = numerical_gradient(fobj, x); % 使用数值梯度
    x = x - step_size * grad;
    x = Bounds(x, lb, ub);
end

% 数值梯度计算函数
function grad = numerical_gradient(f, x)
    grad = zeros(size(x));
    h = 1e-4;
    for i = 1:numel(x)
        x1 = x;
        x2 = x;
        x1(i) = x1(i) - h;
        x2(i) = x2(i) + h;
        grad(i) = (f(x2) - f(x1)) / (2*h);
    end
end

% 边界处理函数
function s = Bounds(s, Lb, Ub)
    % 应用下界向量
    temp = s;
    I = temp < Lb;
    temp(I) = Lb(I);

    % 应用上界向量
    J = temp > Ub;
    temp(J) = Ub(J);
    % 更新新位置
    s = temp;
end