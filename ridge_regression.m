close all
clear

n = 100;
m = 80;
lamda = 1;
N = 1;
iteras = 50;
beta = 4*(m+n)+lamda;
alpha = lamda;
% generate data
x_opt = mvnrnd(zeros(1,n),eye(n),N)'; %n*N
A = mvnrnd(zeros(1,n),eye(n),m); % m*n
eplison = 0.25 * randn(m,N); %m*N
y = A * x_opt + eplison;

x0 = zeros(n,1);
x_initial = x0;
a0 = 1;
value_list_grad = [];
value_list_fastgrad = [];
value_list_simpfastgrad = [];
% gradient descent
for i=1:iteras
    [value] = f(A,x_initial,y,lamda);
    x_next = gradient_descent(A,x_initial,y,lamda,beta);
    value_list_grad = [value_list_grad,value];
    x_initial = x_next;
end
% plot f value
figure
semilogy(value_list_grad)
title('Gradient Descent')
xlabel('iterations')
ylabel('function value')

% fast gradient descent
x_current = x0;
x_past = x0;
a_current = a0;
a_past = a0;
for i=1:iteras
    [value] = f(A,x_current,y,lamda);
    [x_next,a_next] = fast_gradient(A,x_current,x_past,y,lamda,beta,a_current,a_past);
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
x_current = x0;
y_current = x0;
t_current = t0;
for i=1:iteras
    [value] = f(A,x_current,y,lamda);
    [x_next,t_next,y_next] = simpfast_gradient(A,x_current,y,y_current,lamda,beta,t_current,alpha);
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

function  [result] = delta_f(A,x,y,lamda)
     result = A'*(A*x-y) + lamda*x;
end

function [value] = f(A,x,y,lamda)
    value = 1/2 * norm(A*x-y)^2 + lamda/2 * norm(x)^2;
end

function [t_next] = compute_tk(t_k,alpha,beta)
    t = [0:0.01:1];
    result  = t.^2 - ((1-t)*t_k^2 + alpha/beta*t);
    [~,index] = min(abs(result));
    t_next = t(index);
end
function [x_next] = gradient_descent(A,x,y,lamda,beta)
    x_next = x - 1/beta * delta_f(A,x,y,lamda);
end

function [x_next,a_next] = fast_gradient(A,x_k,x_k_minus_1,y,lamda,beta,a_k,a_k_minus_1)
    y_k = x_k + a_k*(a_k_minus_1^(-1)-1)*(x_k - x_k_minus_1);
    x_next = y_k - 1/beta * delta_f(A,y_k,y,lamda);
    a_next = (sqrt(a_k^4 + 4 * a_k^2)-a_k^2)/2;
end

function [x_next,t_next,y_next] = simpfast_gradient(A,x_k,y,y_k,lamda,beta,t_k,alpha)
    x_next = y_k - 1/beta * delta_f(A,y_k,y,lamda);
    t_next = compute_tk(t_k,alpha,beta);
    y_next = x_next + t_k*(1 - t_k)/(t_k^2+t_next)*(x_next-x_k);
end