close all
clear
A =0 ;
n = 50;
m = 100;
lamda = 1;
N = 1;
iteras = 100;
beta = 100;
alpha = lamda;
% generate data
theta_opt = ones(n,1); %n*1
x = randn(m,n); %m*n
z = x*theta_opt; %m*1
p = 1./(1+exp(-z)); %m*1
y = binornd(1,p);
for j=1:length(y)
    if y(j) == 0
        y(j) = -1;
    else
    end
end
theta_0 = randn(n,1);
theta_initial = theta_0;
a0 = 1;
value_list_grad = [];
value_list_fastgrad = [];
value_list_simpfastgrad = [];
% gradient descent
for i=1:iteras
    [value] = f(A,theta_initial,x,y,lamda);
    x_next = gradient_descent(A,theta_initial,x,y,lamda,beta);
    value_list_grad = [value_list_grad,value];
    theta_initial = x_next;
end
% plot f value
figure
semilogy(value_list_grad)
title('Gradient Descent')
xlabel('iterations')
ylabel('function value')

% fast gradient descent
x_current = theta_0;
x_past = theta_0;
a_current = a0;
a_past = a0;
for i=1:iteras
    [value] = f(A,x_current,x,y,lamda);
    [x_next,a_next] = fast_gradient(A,x_current,x_past,x,y,lamda,beta,a_current,a_past);
    value_list_fastgrad = [value_list_fastgrad,value];
    x_past = x_current;
    x_current = x_next;
    a_past = a_current;
    a_current = a_next;
end
% plot f value
figure
semilogy(value_list_fastgrad)
title('Fast Gradient Descent(Algorithm 2)')
xlabel('iterations')
ylabel('function value')

% simplified fast gradient descent
t0 = 0.5;
x_current = theta_0;
y_current = theta_0;
t_current = t0;
for i=1:iteras
    [value] = f(A,x_current,x,y,lamda);
    [x_next,t_next,y_next] = simpfast_gradient(A,x_current,x,y,y_current,lamda,beta,t_current,alpha);
    value_list_simpfastgrad = [value_list_simpfastgrad,value];
    x_current = x_next;
    t_current = t_next;
    y_current = y_next;
end
% plot f value
figure
semilogy(value_list_simpfastgrad)
title('Simplified Fast Gradient Descent(Algorithm 5)')
xlabel('iterations')
ylabel('function value')

function  [result] = delta_f(A,theta,x,y,lamda)
L = length(y);
temp_result1 = zeros(1,L);
temp_result2 = zeros(50,L);
for i= 1:L
    temp_result1(i) = 1+exp(-y(i)*theta'*x(i,:)');
end
for i = 1:L
    temp_result2(:,i) = -y(i)*x(i,:)';
end
    result = sum(1./temp_result1.*(temp_result1-1).*(temp_result2),2) + lamda*theta;
end

function [value] = f(A,theta,x,y,lamda)
L = length(y);
temp_result = zeros(1,L);
for i= 1:L
    temp_result(i) = log(1+exp(-y(i)*theta'*x(i,:)'));
end
value =  sum(temp_result) + lamda/2 * norm(theta)^2;
end

function [t_next] = compute_tk(t_k,alpha,beta)
    t = [0:0.01:1];
    result  = t.^2 - ((1-t)*t_k^2 + alpha/beta*t);
    [~,index] = min(abs(result));
    t_next = t(index);
end
function [theta_next] = gradient_descent(A,theta,x,y,lamda,beta)
    theta_next = theta - 1/beta * delta_f(A,theta,x,y,lamda);
end

function [x_next,a_next] = fast_gradient(A,x_k,x_k_minus_1,x,y,lamda,beta,a_k,a_k_minus_1)
    y_k = x_k + a_k*(a_k_minus_1^(-1)-1)*(x_k - x_k_minus_1);
    x_next = y_k - 1/beta * delta_f(A,y_k,x,y,lamda);
    a_next = (sqrt(a_k^4 + 4 * a_k^2)-a_k^2)/2;
end

function [x_next,t_next,y_next] = simpfast_gradient(A,x_k,x,y,y_k,lamda,beta,t_k,alpha)
    x_next = y_k - 1/beta * delta_f(A,y_k,x,y,lamda);
    t_next = compute_tk(t_k,alpha,beta);
    y_next = x_next + t_k*(1 - t_k)/(t_k^2+t_next)*(x_next-x_k);
end