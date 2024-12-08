%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AAE 561 Convex Optimization
% Final Project: Optimal Station Keeping of a GEO Satellite
%
% Ryan Herberg December 7, 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% Parameters
global nE a_GEO V_GEO J2 muE RE muS muM AU emiss area mass JD0...
    alpha0 t0 f1 f2 f3 f4 N

AU = 1.496E8; %km
J2 = 1.082635E-3;
muE = 3.986*10^5; %km^3/s^2
muS = 1.327*10^11; %km^3/s^2
muM = 4902.8; %km^3/s^2
RE = 6378.137; %km
nE = 2*pi/(23*60*60 + 56*60 * 4); %rad/s
a_GEO = 35786; %km
V_GEO = nE*a_GEO; %km/s
emiss = 0.21;
area = 0.026*0.002; %km^2
mass = 1750; % kg
alpha0 = 0; %rad
year = 2025;
month = 8; % August
day = 15;
JD0 = juliandate(year,month,day);

%% Nonsmooth Constrained Optimization using Subgradient Method:

% Discretization for Optimal Control
t0 = 0; % s
tf = 86400; % s (one day)
h_c = 10000; % s
tspan_c = 0:h_c:tf;
N = length(tspan_c);

% Affine Dynamics Matrices
A = zeros(6,6);
A(6,1) = 1;
At = (1/24*A^4*h_c^4)+(1/6*A^3*h_c^3)+(1/2*A^2*h_c^2)+(A*h_c)+eye(6);
F = zeros(6*(N-1),6);
D = zeros(6*(N-1),1);
J = eye(6*(N-1),6*(N-1));
H = zeros(6*(N-1),3*(N-1));
H_B = zeros(6*(N-1),3*(N-1));
for i = 1:N-1
    t = tspan_c(i);
    alpha_k1 = RA_GEO(t);
    alpha_k2 = RA_GEO(t+(h_c/2));
    alpha_k3 = RA_GEO(t+h_c);

    B_k1 = B_alpha(alpha_k1);
    B_k2 = B_alpha(alpha_k2);
    B_k3 = B_alpha(alpha_k3);

    ud_k1 = GEO_pert_accel(t);
    ud_k2 = GEO_pert_accel(t+(h_c/2));
    ud_k3 = GEO_pert_accel(t+h_c);

    Bt = ((1/24*A^3*h_c^4)+(1/12*A^2*h_c^3)+(1/6*A*h_c^2)+(1/6*h_c*eye(6)))*B_k1...
    + ((1/12*A^2*h_c^3)+(1/3*A*h_c^2)+(2/3*h_c*eye(6)))*B_k2 + (1/6*h_c)*B_k3;
    d = ((1/24*A^3*h_c^4)+(1/12*A^2*h_c^3)+(1/6*A*h_c^2)+(1/6*h_c*eye(6)))*B_k1*ud_k1...
    + ((1/12*A^2*h_c^3)+(1/3*A*h_c^2)+(2/3*h_c*eye(6)))*B_k2*ud_k2 + (1/6*h_c)*B_k3*ud_k3;
    
    F(6*i-5:6*i,:) = At^i;
    D(6*i-5:6*i) = d;
    for col = 1:N-1-i
        J(6*(i+col)-5:6*(i+col),6*col-5:col*6) = At^i;
    end
    for index = i:N-1
        H_B(6*index-5:6*index,3*i-2:3*i) = Bt;
    end
end
for i = 1:N-1
    for j = 1:N-1
        H(6*i-5:6*i,3*j-2:3*j) = J(6*i-5:6*i,6*j-5:6*j)*H_B(6*i-5:6*i,3*j-2:3*j);
    end
end

epsilonLB = -6E-2;
epsilonUB = 6E-2;
Xlb_set = [-100;-3E-4;-3E-4;-2.5E-4;-2.5E-4;epsilonLB];
Xub_set = -1.*Xlb_set;
Xlb = zeros(6*(N-1),1);
Xub = zeros(6*(N-1),1);
for i = 1:N-1
    Xlb(6*i-5:6*i,1) = Xlb_set;
    Xub(6*i-5:6*i,1) = Xub_set;
end
Umin = 0; % N
Umax = 0.05; % N

x0 = zeros(6,1);

% Optimization Constraint Functions:
f1 = @(U) -F*x0 - H*U - J*D + Xlb;
f2 = @(U) F*x0 + H*U + J*D - Xub;
f3 = @(U) -norm(U) + Umin;
f4 = @(U) norm(U) - Umax;

% Subgradient method parameters
R = 0.001; % Step size parameter
max_iter = 500; % Maximum number of iterations
tol = 1e-6; % Convergence tolerance
u0 = zeros(3*(N-1),1); % initial point
u_k = zeros(3*(N-1),1); % Initialize uk
step = @(k) R / sqrt(k + 0.5);

% Iteration
for k = 1: max_iter
    % Compute function values
    f_val = fcn_nonsmooth_obj(u_k); % objective function values
    [c, ~] = fcn_nonsmooth_cons(u_k); % Constraint values
    
    % Compute gradient (subgradient) of the objective function
    grad_f = nonsmooth_obj_subgradient(u_k);
    
%     % Compute f_bar(x_k) and g_bar(x_k)
%     f_bar = max(c);
%     
%     if f_bar == c(1) && f_bar == c(2)
%         % Priority based tie breaker: choose f1 if both are equal
%         g_bar = 0.4*x_k; % Gradient of f1(x) = 0.2x^2 - 3
%     elseif f_bar == c(1)
%         g_bar = 0.4*x_k; % Gradient of f1(x) = 0.2x^2 - 3
%     elseif f_bar == c(2)
%         g_bar = -2; % Gradient of f2(x) -2x + 1.5
%     else        
%         error('Unexpected case: f_bar does not match any constraint');
%     end
    
    % Define a small tolerance for floating-point comparison - more complex
    % functions
    tolerance = 1e-8;

    % Compute f_bar(x_k) and g_bar(x_k)
    f_bar = max(c); % Maximum of constraints
    
    check = 0;
    cnstr_flg = 0;
    for i = 1:length(c)
        if abs(f_bar - c(i)) < tolerance
            check = check + 1;
        end
    end
    
    if check > 1
        % Priority based tie breaker: choose f1 if more than one are equal
        g_bar = 1*ones(3*(N-1),1); % Gradient of f4(U) = U - Umax <= 0
    else
        for i = 1:length(c)
            if i < (N-1)*6 + 1 % f1(U) = -Fx0 - HU - JD + Xlb <= 0
                if abs(f_bar - c(i)) < tolerance % f1(U) = -Fx0 - HU - JD + Xlb <= 0
                    g_bar = -H*ones(3*(N-1),1); % Gradient of violated constraint
                    cnstr_flg = 1;
                    c_index = i;
                end
            elseif i < 2*(N-1)*6 + 1 % f2(U) = Fx0 + HU + JD - Xub <= 0
                if abs(f_bar - c(i)) < tolerance
                    g_bar = H*ones(3*(N-1),1); % Gradient of violated constraint
                    cnstr_flg = 1;
                    c_index = i;
                end
            elseif i < 2*(N-1)*6 + 2
                if abs(f_bar - c(i)) < tolerance
                    g_bar = -1*ones(3*(N-1),1); % Gradient of f3(U) = -U + Umin <= 0
                    c_index = i;
                end
            elseif i == 2*(N-1)*6 + 2
                if abs(f_bar - c(i)) < tolerance
                    g_bar = 1*ones(3*(N-1),1); % Gradient of f4(U) = U - Umax <= 0
                    c_index = i;
                end
            else        
                error('Unexpected case: f_bar does not match any constraint');
            end
        end
    end
    
    % Determine p_k
    if f_bar < norm(g_bar(c_index)) * step(k)
        p_k = grad_f; % Use gradient of objective
    else
        p_k = g_bar;
    end
    
    % Update rule with projection
    u_update = u_k;
    u_update(c_index) = u_k(c_index) - step(k) * p_k(c_index) / norm(p_k(c_index));
    u_next = project_with_fmincon_nonsmooth(u_update);
    
    % Check for convergence
    if norm(u_next - u_k) < tol
        fprintf('Converged at iteration %d\n', k);
        break;
    end
    
    % update x_k
    u_k = u_next;    
end

u_opt_sub = u_k;
f_opt_sub = fcn_nonsmooth_obj(u_k);


%% Propogate Satellite Position w/ control

% Discretization for Propogation
t0 = 0; % s
tf = 86400; % s (one day)
h_p = 500; % s
tspan_p = 0:h_p:tf;

X_SC = zeros(6,length(tspan_p)); % GEO Slot Center
X_SAT = zeros(6,length(tspan_p)); % Controlled Satellite
X_SAT_nom = zeros(6,length(tspan_p)); % Nominal Perturbed Satellite

X_SC_0(:,1) = [a_GEO;0;0;0;V_GEO;0];
X_SAT_0(:,1) = [a_GEO;0;0;0;V_GEO;0];
X_SC(:,1) = X_SC_0;
X_SAT(:,1) = X_SAT_0;
X_SAT_nom(:,1) = X_SAT_0;

uc = zeros(3,length(tspan_p));
thrust_count = 1;
u_active = u0(1:3);
for k = 1:length(tspan_p)-1
    if tspan_p(k) < h_c*thrust_count
        uc(:,k) = u_active;
    else
        uc(:,k) = u_opt_sub(thrust_count*3-2:thrust_count*3,1);
        u_active = uc(:,k);
        thrust_count = thrust_count+1;
    end
    X_SAT(:,k+1) = rk4(@(t,x) EOM_GEO_perturbing(t,x,uc(:,k)),h_p,tspan_p(k),X_SAT(:,k));
    X_SAT_nom(:,k+1) = rk4(@(t,x) EOM_GEO_perturbing(t,x,zeros(3,1)),h_p,tspan_p(k),X_SAT(:,k));
    X_SC(:,k+1) = rk4(@(t,x) EOM_GEO_slotcenter(t,x),h_p,tspan_p(k),X_SC(:,k));
end

% Using affine relationship to derive state and subsequent orbital parameters
X = zeros(6,length(tspan_p));
X(:,1) = zeros(6,1);
for k = 1:length(tspan_p)-1
    X(:,k+1) = GEO_rk4(tspan_p(k),X(:,k),uc(:,k),h_p);
end

% Conversion to orbital parameters
n = zeros(length(tspan_p),1);
omega = zeros(length(tspan_p),1);
e = zeros(length(tspan_p),1);
RA = zeros(length(tspan_p),1);
i = zeros(length(tspan_p),1);
for k = 1:length(tspan_p)
    n(k) = X(1,k)+nE;
    omega(k) = atan2(X(2,k),X(3,k));
    if omega(k) == 0
        e(k) = X(3,k)/cos(omega(k));
    elseif omega(k) == 1.0
        e(k) = X(2,k)/sin(omega(k));
    else
        e(k) = X(2,k)/sin(omega(k));
    end
    RA(k) = atan2(X(4,k),X(5,k));
    if RA(k) == 0
        i(k) = 2*asin(X(5,k)/cos(RA(k)));
    elseif RA(k) == 1.0
        i(k) = 2*asin(X(4,k)/sin(RA(k)));
    else
        i(k) = 2*asin(X(4,k)/sin(RA(k)));
    end
end

%% Output

% Display Propogation of Sat and SC:

%Plot of propogated motion about slot center
figure(1)
plot(X(2,:),X(3,:));
xlabel('E3');
ylabel('E2');
title('Relative Eccentricity Vector');
grid on

figure(2)
plot(X(4,:),X(5,:));
xlabel('E5');
ylabel('E4');
title('Relative Inclination Vector');
grid on

figure(3)
tiledlayout(3,1)
title('Relative Position to GEO Slot Center (ECI Frame)');
nexttile
plot(tspan_p/86400,X_SC(1,:)-X_SAT(1,:));
ylabel('$X_{SC} - X_{SAT}$','Interpreter','Latex');

nexttile
plot(tspan_p/86400,X_SC(2,:)-X_SAT(2,:));
ylabel('$Y_{SC} - Y_{SAT}$','Interpreter','Latex');

nexttile
plot(tspan_p/86400,X_SC(3,:)-X_SAT(3,:));
xlabel('Time [days]');
ylabel('$Z_{SC} - Z_{SAT}$','Interpreter','Latex');

figure(4)
tiledlayout(3,1)
title('Positions Over Time (ECI Frame)');
nexttile
plot(tspan_p/86400,X_SC(1,:));
hold on
plot(tspan_p/86400,X_SAT_nom(1,:));
plot(tspan_p/86400,X_SAT(1,:));
legend('$X_{SC}$','$X_{free}$','$X_{c}$', 'Interpreter', 'Latex');
ylabel('X');
nexttile
plot(tspan_p/86400,X_SC(2,:));
hold on
plot(tspan_p/86400,X_SAT_nom(2,:));
plot(tspan_p/86400,X_SAT(2,:));
legend('$Y_{SC}$','$Y_{free}$','$Y_{c}$', 'Interpreter', 'Latex');
ylabel('Y');
nexttile
plot(tspan_p/86400,X_SC(3,:));
hold on
plot(tspan_p/86400,X_SAT_nom(3,:));
plot(tspan_p/86400,X_SAT(3,:));
legend('$Z_{SC}$','$Z_{free}$','$Z_{c}$', 'Interpreter', 'Latex');
ylabel('Z');
xlabel('Time [days]');

% Display constraints and optimal solution
control_positions = -F*x0 - H*u_opt_sub - J*D;
thrust_magnitude = zeros(length(N),1);
thrust_magnitude(1) = norm(u0);
E = zeros(6,length(N));
E(:,1) = x0;
Xlower = zeros(6,length(N));
Xlower(:,1) = Xlb_set;
Xupper = zeros(6,length(N));
Xupper(:,1) = Xub_set;
for i = 1:N-1
    thrust_magnitude(i+1) = norm(u_opt_sub(3*i-2:3*i,1));
    E(:,i+1) = control_positions(6*i-5:6*i);
    Xlower(:,i+1) = Xlb(6*i-5:6*i);
    Xupper(:,i+1) = Xub(6*i-5:6*i);
end

figure(5)
tiledlayout(6,1)
title('Resultant Equinoctial Orbital Elements wrt Constraints');
nexttile
plot(tspan_c/86400,E(1,:), 'linewidth', 3, 'Color', 'red');
hold on
plot(tspan_c/86400,Xlower(1,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(1,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E1');
grid on

nexttile
plot(tspan_c/86400,E(2,:), 'linewidth', 3, 'Color', 'green');
hold on
plot(tspan_c/86400,Xlower(2,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(2,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E2');
grid on

nexttile
plot(tspan_c/86400,E(3,:), 'linewidth', 3, 'Color', 'blue');
hold on
plot(tspan_c/86400,Xlower(3,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(3,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E3');
grid on

nexttile
plot(tspan_c/86400,E(4,:), 'linewidth', 3, 'Color', 'yellow');
hold on
plot(tspan_c/86400,Xlower(4,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(4,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E4');
grid on

nexttile
plot(tspan_c/86400,E(5,:), 'linewidth', 3, 'Color', 'magenta');
hold on
plot(tspan_c/86400,Xlower(5,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(5,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E5');
grid on

nexttile
plot(tspan_c/86400,E(6,:), 'linewidth', 3, 'Color', 'cyan');
hold on
plot(tspan_c/86400,Xlower(6,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(6,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E6');
grid on
xlabel('Time [days]');

figure(6)
stairs(tspan_c/86400, thrust_magnitude, 'linewidth', 3, 'Color', 'green');
hold on
plot(tspan_c/86400, Umin*ones(length(tspan_c)), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400, Umax*ones(length(tspan_c)), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
xlabel('Time [days]');
ylabel('Thrust Magnitude');
title('Thrust Magnitude wrt Constraints');
grid on


%% Local Functions:

% Optimization Functions:

function f = fcn_nonsmooth_obj(u)
    % Objective function:
    f = norm(u,1);
end

function [c, ceq] = fcn_nonsmooth_cons(u)
    % Constraints:
    % f1(U) = -Fx0 - HU - JD + Xlb <= 0
    % f2(U) = Fx0 + HU + JD - Xub <= 0
    % f3(U) = -U + Umin <= 0
    % f4(U) = U - Umax <= 0
    global f1 f2 f3 f4
    c = [f1(u); f2(u); f3(u); f4(u)];
    ceq = [];
end

function g = nonsmooth_obj_subgradient(u)
    global N
    %Gradient of the objective function f(u)
    g = ones(3*(N-1),1);
end

function u_proj = project_with_fmincon_nonsmooth(u_update, Umax, Umin)
    % use fmincon to project u_update onto the feasible set Q
    options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');
    
    % Objective function: minimize ||u - u_update||
    objective = @(u) norm(u - u_update);
    
    % Define maximum and minimum constraint in A and b
    A = [u_update(1)/norm(u_update(1)),u_update(2)/norm(u_update(2)),u_update(3)/norm(u_update(3));
        u_update(1)/norm(u_update(1)),u_update(2)/norm(u_update(2)),u_update(3)/norm(u_update(3))];
    b = [Umax;-Umin];
    
    
    % Solve the projection problem with fmincon
    u_proj = fmincon(objective, u_update, A, b, [], [], [], [], [], options);
end

% Runge Kutta Propogator
function xout = rk4(fun,dt,tk,xk)
    f1 = fun(tk,xk);
    f2 = fun(tk+dt/2,xk+(dt/2)*f1);
    f3 = fun(tk+dt/2,xk+(dt/2)*f2);
    f4 = fun(tk+dt,xk+(dt)*f3);
    
    xout = xk + (dt/6)*(f1+2*f2+2*f3+f4);
end

% GEO Differential Function
function x_k = GEO_rk4(t,x,u,h)
    A = zeros(6,6);
    A(6,1) = 1;
    
    alpha_k1 = RA_GEO(t);
    alpha_k2 = RA_GEO(t+(h/2));
    alpha_k3 = RA_GEO(t+h);
    
    B_k1 = B_alpha(alpha_k1);
    B_k2 = B_alpha(alpha_k2);
    B_k3 = B_alpha(alpha_k3);
    
    ud_k1 = GEO_pert_accel(t);
    ud_k2 = GEO_pert_accel(t+(h/2));
    ud_k3 = GEO_pert_accel(t+h);
    
    At = (1/24*A^4*h^4)+(1/6*A^3*h^3)+(1/2*A^2*h^2)+(A*h)+eye(6);
    Bt = ((1/24*A^3*h^4)+(1/12*A^2*h^3)+(1/6*A*h^2)+(1/6*h*eye(6)))*B_k1...
    + ((1/12*A^2*h^3)+(1/3*A*h^2)+(2/3*h*eye(6)))*B_k2 + (1/6*h)*B_k3;
    d = ((1/24*A^3*h^4)+(1/12*A^2*h^3)+(1/6*A*h^2)+(1/6*h*eye(6)))*B_k1*ud_k1...
    + ((1/12*A^2*h^3)+(1/3*A*h^2)+(2/3*h*eye(6)))*B_k2*ud_k2 + (1/6*h)*B_k3*ud_k3;

    x_k = At*x + Bt*u + d;
end

% Right Ascension of the Geostationary slot center: [alpha_0 in radians]
function alpha = RA_GEO(t)
    global t0 alpha0 nE
    alpha = alpha0+nE*(t-t0);
end

% GEO slot center based upon alpha in ECI frame
function r_GSC = slot_center(alpha)
    global a_GEO

    r_GSC = a_GEO*[cos(alpha);sin(alpha);0];
end

% EOM for GEO slot center
function dx = EOM_GEO_slotcenter(t,x)
    global muE
    ddx = -muE/norm(x(1:3))^3*x(1:3);
    
    dx = [x(4);x(5);x(6);ddx(1);ddx(2);ddx(3)];
end

% Perturbing Acceleration Functions for affine dynamics
function ud = GEO_pert_accel(t)
    global J2 muE muS muM RE emiss area mass AU JD0

    JD = JD0 + (t/86400);
    TA = RA_GEO(t);
    r = slot_center(TA); %ECI frame

    % J2 perturbation:
    a_J2 = (-3/2)*J2*muE*(RE^2/norm(r)^5)*[norm(r);0;0]; % rtn frame
    
    % ECI to rtn transformation matrix (TA = alpha, RA = 0, i = 0)
    T_ECI_rtn = [cos(TA),sin(TA),0;...
        -sin(TA),cos(TA),0;...
        0,0,1];
    
    % 3rd Body Sun:
    rS = planetEphemeris(JD,'Earth','Sun'); % ICRF or ECI frame
    rS = rS';
    a_3b_S_ECI = muS*(((rS-r)/norm(rS-r)^3)-(rS/(norm(rS)^3)));
    a_3b_S_rtn = T_ECI_rtn*a_3b_S_ECI;
    % 3rd Body Moon:
    rM = planetEphemeris(JD,'Earth','Moon'); % ICRF or ECI frame
    rM = rM';
    a_3b_M_ECI = muM*(((rM-r)/norm(rM-r)^3)-(rM/(norm(rM)^3)));
    a_3b_M_rtn = T_ECI_rtn*a_3b_M_ECI;
    
    % Solar Radiation pressure
    P = 4.56*10^-9; % Nkm^-2
    CR = 1+emiss;
    rS_rtn = T_ECI_rtn*rS;
    a_SRP = (-P*CR*(area/mass)*norm(rS/(norm(rS)^3))*AU^2)*rS_rtn;

    ud = a_J2+a_3b_S_rtn+a_3b_M_rtn+a_SRP; %km/s^2 in RTN frame
end

% Differential equation of perturbed satellite
function dx = EOM_GEO_perturbing(t,x,uc)
    global J2 muE muS muM RE emiss area mass AU JD0

    JD = JD0 + (t/86400);
    TA = RA_GEO(t);
    r = slot_center(TA); %ECI frame
    
    % rtn to ECI transformation matrix (TA = alpha, RA = 0, i = 0)
    T_rtn_ECI = [cos(TA),-sin(TA),0;...
        sin(TA),cos(TA),0;...
        0,0,1];

    % J2 perturbation:
    a_J2_rtn = (-3/2)*J2*muE*(RE^2/norm(r)^5)*[norm(r);0;0]; % rtn frame
    a_J2_ECI = T_rtn_ECI*a_J2_rtn;
    
    % 3rd Body Sun:
    rS = planetEphemeris(JD,'Earth','Sun'); % ICRF or ECI frame
    rS = rS';
    a_3b_S_ECI = muS*(((rS-r)/norm(rS-r)^3)-(rS/(norm(rS)^3)));
    % 3rd Body Moon:
    rM = planetEphemeris(JD,'Earth','Moon'); % ICRF or ECI frame
    rM = rM';
    a_3b_M_ECI = muM*(((rM-r)/norm(rM-r)^3)-(rM/(norm(rM)^3)));
    
    % Solar Radiation pressure
    P = 4.56*10^-9; % Nkm^-2
    CR = 1+emiss;
    a_SRP = (-P*CR*(area/mass)*norm(rS/(norm(rS)^3))*AU^2)*rS;
    
    ddx_p = a_J2_ECI+a_3b_S_ECI+a_3b_M_ECI+a_SRP; %km/s^2 in ECI frame
    
    ddx_g = -muE/norm(x(1:3))^3*x(1:3);
    
    ddx = ddx_p+ddx_g+uc;
    
    dx = [x(4);x(5);x(6);ddx(1);ddx(2);ddx(3)];
end

% B Matrix, conversion matrix from radial, velocity, normal frame to equinoctial
% orbital elements: [alpha in radians]
function B = B_alpha(alpha)
    global V_GEO
    global a_GEO
    
    B = [0,-3/a_GEO,0;...
        -(1/V_GEO)*cos(alpha),(2/V_GEO)*sin(alpha),0;...
        (1/V_GEO)*sin(alpha),(2/V_GEO)*cos(alpha),0;...
        0,0,(1/(2*V_GEO))*sin(alpha);...
        0,0,(1/(2*V_GEO))*cos(alpha);...
        -2/V_GEO,0,0];
end
