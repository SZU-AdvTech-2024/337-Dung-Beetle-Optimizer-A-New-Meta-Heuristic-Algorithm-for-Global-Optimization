clear
clc
close all

%%  参数设置
SearchAgents = 30;                                   
Max_iterations = 1000;                                  
number = 'F1';                                          % F1~F23
[lb,ub,dim,fobj] = Get_Functions_details(number);       % [lb,ub,D,y]：下界、上界、维度、目标函数表达式

%%  调用算法
[DBO_pos,DBO_score,DBO_curve]=DBO(SearchAgents,Max_iterations,lb,ub,dim,fobj);      
[IDBO_pos,IDBO_score,IDBO_curve]=IDBO(SearchAgents,Max_iterations,lb,ub,dim,fobj); 
%% Figure
figure1 = figure('Color',[1 1 1]);
G1=subplot(1,2,1,'Parent',figure1);
func_plot(number)
title(number)
xlabel('x')
ylabel('y')
zlabel('z')
subplot(1,2,2)
G2=subplot(1,2,2,'Parent',figure1);
CNT=20;
k=round(linspace(1,Max_iterations,CNT)); %随机选CNT个点
iter=1:1:Max_iterations;
if ~strcmp(number,'F16')&&~strcmp(number,'F9')&&~strcmp(number,'F11')  %这里是因为这几个函数收敛太快，不适用于semilogy，直接plot
    semilogy(iter(k),IDBO_curve(k),'b-*','linewidth',1);
    hold on
    semilogy(iter(k),DBO_curve(k),'r-p','linewidth',1);
else
    plot(iter(k),IDBO_curve(k),'b-*','linewidth',1);
    hold on
    plot(iter(k),DBO_curve(k),'r-p','linewidth',1);
end
grid on;
title('收敛曲线')
xlabel('迭代次数');
ylabel('适应度值');
box on
legend('IDBO','DBO')
set (gcf,'position', [300,300,800,330])

