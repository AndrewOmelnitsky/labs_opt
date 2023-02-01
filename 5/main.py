self.alpha = 0.5
self.beta = 1.5
self.gamma = 1.5
self.delta = 0.1
self.nu = 5
self.mu = 20
self.lambda_ = 20
self.rho = 10
self.A0 = 1
self.L0 = 1
self.D0 = 1
self.tau = 0.4
self.sigma = 0.2
self.theta = (1 + self.alpha * (self.beta - 1)) ** (-1)


//claer;
aph = 0.5; bt = 1.5; gm = 1.5; dt = 0.1; nu = 5;
mu = 20; lmd = 20; ro = 10;
A0 = 1; L0 = 1; D0 = 1;
tau = 0.4; sigma = 0.15;
theta = (1 + aph * (bt - 1)) ^ (-1);
initial = [0; 0.5; 0.25; 0.1; 0; 0.5; 0.25; 0.1]

left = 0;
right = 600; resolution = 10;
t = linspace(left, right, resolution * (right - left + 1));

function [result]=L1(data)
    result(1) = data(4) * ((1 - aph) * A0 * data(2) / data(3)) ^ (1 / aph);
endfunction

function [result]=Q1(data)
    result(1) = A0 * data(4) ^ aph * L1(data) ^ (1 - aph);
endfunction

function [result]=D1(data)
    result(1) = D0 * exp(-bt * data(2)) * data(6) / (data(2) + data(6));
endfunction

function [result]=S1(data)
    result(1) = L0 * (1 - exp(-gm * data(3))) * data(3) / (data(3) + data(7));
endfunction

function [result]=I1(data)
    result(1) = (1 - tau) * (1 - theta) * data(1);
endfunction

function [result]=G1(data)
    result(1) = (1 - tau) * theta * data(1);
endfunction

function [result]=L2(data)
    result(1) = data(8) * ((1 - aph) * A0 * data(6) / data(7)) ^ (1 / aph);
endfunction

function [result]=Q2(data)
    result(1) = A0 * data(8) ^ aph * L2(data) ^ (1 - aph);
endfunction

function [result]=D2(data)
    result(1) = D0 * exp(-bt * data(6)) * data(2) / (data(2) + data(6));
endfunction

function [result]=S2(data)
    result(1) = L0 * (1 - exp(-gm * data(7))) * data(7) / (data(3) + data(7));
endfunction

function [result]=I2(data)
    result(1) = (1 - theta) * data(5);
endfunction

function [result]=G2(data)
    result(1) = theta * data(5);
endfunction

function [result]=T(data)
    result(1) = tau * data(1);
endfunction

function [result]=G(data)
    result(1) = (1 - sigma) * tau * data(1);
endfunction

function [result]=calculate(t, data)
    result(1) = (data(2) * min(Q1(data), D1(data)) - data(3) * min(L1(data), S1(data)) - data(1)) / nu;
    result(2) = (D1(data) - Q1(data)) / mu;
    result(3) = (L1(data) - S1(data)) / lmd;
    result(4) = -dt * data(4) + I1(data);
    result(5) = (exp(-ro * sigma * T(data)) * data(6) * min(Q2(data), D2( data)) - data(7) * min(L2(data), S2(data)) - data(5)) / nu;
    result(6) = (D2(data) - Q2(data)) / mu;
    result(7) = (L2(data) - S2(data)) / lmd;
    result(8) = -dt * data(8) + I2(data)
endfunction

// disp(initial);
// disp('\n');
// disp(initial(4));
// disp('\n');

ode_result = ode(initial, left, t, calculate);
legal_part = []; black_part = []; black_coef = [];
legal_profit_part = []; government_profit_part = []; black_profit_part = [];
legal_price = []; black_price = [];
legal_salary = []; black_salary = [];
legal_salary_coef = []; black_salary_coef = [];
legal_foundation_coef = []; black_foundation_coef = [];
g_result = []; t_result = []; g1_result = []; g2_result = [];

[height , width] = size(ode_result); // disp(width , height)

for i = 1: width
    temp_ode_result = ode_result(:,i)
    legal_part($ + 1) = Q1(temp_ode_result) ./ (Q1(temp_ode_result) + Q2(temp_ode_result));
    black_part($ + 1) = Q2(temp_ode_result) ./ (Q1(temp_ode_result) + Q2(temp_ode_result));
    black_coef($ + 1) = Q1(temp_ode_result) ./ Q2(temp_ode_result);
    //legal_profit_part($ + 1) = G1(temp_ode_result) ./ (G1(temp_ode_result) + G2(temp_ode_result) + G(temp_ode_result));
    //government_profit_part($ + 1) = G(temp_ode_result) ./ (G1(temp_ode_result) + G2(temp_ode_result) + G2(temp_ode_result));
    //black_profit_part($ + 1) = G2(temp_ode_result) ./ (G1(temp_ode_result) + G2(temp_ode_result) + G(temp_ode_result));
    //disp('\n');
    //disp(temp_ode_result(1));
    //disp('\n');
    legal_price($ + 1) = temp_ode_result(2);
    black_price($ + 1) = temp_ode_result(6);
    legal_salary($ + 1) = temp_ode_result(3);
    black_salary($ + 1) = temp_ode_result(7);
    legal_salary_coef($ + 1) = L1(temp_ode_result) ./ S1(temp_ode_result);
    black_salary_coef($ + 1) = L2(temp_ode_result) ./ S2(temp_ode_result);

	//disp('\n');
    //disp(temp_ode_result(4));
	//disp(initial(4));
    //disp('\n');
    legal_foundation_coef($ + 1) = temp_ode_result(4) ./ initial(4);
    black_foundation_coef($ + 1) = temp_ode_result(8) ./ initial(8);
    g_result($ + 1) = G(temp_ode_result); t_result($ + 1) = T(temp_ode_result);
    g1_result($ + 1) = G1(temp_ode_result); g2_result($ + 1) = G2(temp_ode_result);
end

plot(t', [g_result'; t_result'; g1_result'; g2_result']);
legend(prettyprint(["Pro t"; "Taxes"; "Legal"; "Black"],"latex","",%t),4);
xtitle("", "t", "");

/*
scf(1);
plot(t', [legal_part'; black_part'; black_coef']);
legend(prettyprint(["Legal production part"; "Black production part"; "Ratio"],"latex","",%t), 3);
xtitle("", "t", "");

scf(2);
plot(t', [legal_profit_part'; government_profit_part'; black_profit_part']);
legend(prettyprint(["Legal pro t part"; "Government pro t part"; "Black profit part"],"latex","",%t), 3);
xtitle("", "t", "");

scf(3);
plot(t', [legal_price'; black_price'; legal_salary'; black_salary']);
legend(prettyprint(["Legal price"; "Black price"; "Legal salary"; "Black salary"],"latex","",%t), 4);
xtitle("", "t", "");

scf(4);
plot(t', [legal_salary_coef'; black_salary_coef'; legal_foundation_coef'; black_foundation_coef']);
legend(prettyprint(["Legal salary ratio"; "Black salary ratio"; "Legal foundation ratio"; "Black foundation ratio"],"latex","",%t), 4);
xtitle("", "t", "");
*/