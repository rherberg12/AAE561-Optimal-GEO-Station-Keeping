%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AAE 561 Convex Optimization
% Final Project: Optimal Station Keeping of a GEO Satellite
%
% Ryan Herberg December 7, 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear; close all;

%% Parameters
global nE a_GEO V_GEO J2 muE RE muS muM AU emiss area mass JD0...
    alpha0 t0 f1 f2 N

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
h_c = 500; % s
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

epsilonLB = -6E-4;
epsilonUB = 6E-4;
Xlb_set = [-100;-3E-2;-3E-2;-2.5E-2;-2.5E-2;epsilonLB];
Xub_set = -1.*Xlb_set;
Xlb = zeros(6*(N-1),1);
Xub = zeros(6*(N-1),1);
for i = 1:N-1
    Xlb(6*i-5:6*i,1) = Xlb_set;
    Xub(6*i-5:6*i,1) = Xub_set;
end
Umin = 0*10^-3;%0.5/h_c; % mN
Umax = 0.05*10^-3; % mN

x0 = zeros(6,1);

% Optimization Constraint Functions:
f1 = @(U) -F*x0 - H*U - J*D + Xlb;
f2 = @(U) F*x0 + H*U + J*D - Xub;

% Subgradient method parameters
Rglobal = 0.001; % Step size parameter
max_iter = 50000; % Maximum number of iterations
tol = 1e-9; % Convergence tolerance
u0 = zeros(3*(N-1),1); % initial point
u_k = zeros(3*(N-1),1); % Initialize uk
%step = @(k,R) R*(1-(k/(2*max_iter)));
step = @(f) f*0.1;

% Iteration
iter = 0;
c_index_old = 0;
step_increase_iter = max_iter*0.01;
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
    
    for i = 1:length(c)
        if i < (N-1)*6 + 1 % f1(U) = -Fx0 - HU - JD + Xlb <= 0
            if abs(f_bar - c(i)) < tolerance % f1(U) = -Fx0 - HU - JD + Xlb <= 0
                g_bar = -H*ones(3*(N-1),1); % Gradient of violated constraint
                c_index = i;
                driving_constraint = c_index;
            end
        elseif i < 2*(N-1)*6 + 1 % f2(U) = Fx0 + HU + JD - Xub <= 0
            if abs(f_bar - c(i)) < tolerance
                g_bar = H*ones(3*(N-1),1); % Gradient of violated constraint
                c_index = i - (6*(N-1));
                driving_constraint = i;
            end
        else
            error('Unexpected case: f_bar does not match any constraint');
        end
    end
    
    constr_flg = 0;
    % Determine p_k
    if f_bar < 0 % There is no natural improvement without thrust....
        if norm(u_k) == 0
            p_k = 0;
        else
            p_k = grad_f; % Use gradient of objective
        end
        constr_flg = 1;
    else
        p_k = g_bar;
    end
    
    % Define step size
    if c_index_old ~= driving_constraint
        Rlocal = Rglobal;
        search = 1; % Restart search iteration criteria
    elseif search < max_iter
        if search > step_increase_iter
            Rlocal = 2*Rlocal;
        end
        search = search+1;
    else
        fprintf('Max iterations exceeded');
        break;
    end
    
    % Update
    if constr_flg == 0
        x_step = zeros(6*(N-1),1);
        x_step(c_index) = step(f_bar)*p_k(c_index)/norm(p_k(c_index));
        u_step = H\x_step;
        u_update = u_k + u_step;
    else
        u_update = u_k - step(f_bar) * p_k;
    end
    
    % project each propulsive maneuver to the feasible set
    u_next = project_with_fmincon_nonsmooth(u_update,Umin,Umax);
    
    % Check for convergence
    conv = 0;
    for i = 1:length(u_k)
        if norm(u_next(i) - u_k(i)) > tol
            conv = 1;
        end
    end
    if conv ~= 1
        fprintf('Converged at iteration %d\n', k);
        break;
    end
    
    % update x_k
    u_k = u_next;
    
    % Search variables
    iter = iter+1;
    c_index_old = driving_constraint;
    disp(f_bar)
end

u_opt_sub = u_k;
f_opt_sub = fcn_nonsmooth_obj(u_k);


%% Propogate Satellite Position w/ control

% Discretization for Propogation
t0 = 0; % s
num_points = 1000;
h_p = tf/num_points; % s
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
ax = 14;
ti = 18;
Ti = 20;

%Plot of propogated motion about slot center
figure(1)
plot(X(2,:),X(3,:));
xlabel('E3','fontsize', ax);
ylabel('E2','fontsize', ax);
title('Relative Eccentricity Vector', 'fontsize', ti);
grid on

figure(2)
plot(X(4,:),X(5,:));
xlabel('E5','fontsize', ax);
ylabel('E4','fontsize', ax);
title('Relative Inclination Vector','fontsize', ti);
grid on

figure(3)
fig3 = tiledlayout(3,1);
title(fig3, 'Relative Position to GEO Slot Center (ECI Frame)','fontsize', Ti);
nexttile
plot(tspan_p/86400,X_SC(1,:)-X_SAT(1,:), 'Color', 'black');
ylabel('$X_{SC} - X_{SAT}$ [km]','Interpreter','Latex','fontsize',ax);
xlabel('Time [days]','fontsize', ax);
title('X Relative Position','fontsize', ti);
grid on

nexttile
plot(tspan_p/86400,X_SC(2,:)-X_SAT(2,:), 'Color', 'black');
ylabel('$Y_{SC} - Y_{SAT}$ [km]','Interpreter','Latex','fontsize', ax);
xlabel('Time [days]','fontsize', ax);
title('Y Relative Position','fontsize', ti);
grid on

nexttile
plot(tspan_p/86400,X_SC(3,:)-X_SAT(3,:), 'Color', 'black');
ylabel('$Z_{SC} - Z_{SAT}$ [km]','Interpreter','Latex');
xlabel('Time [days]','fontsize', ax);
title('Z Relative Position','fontsize', ti);
grid on

figure(4)
fig4 = tiledlayout(3,1);
title(fig4, 'Positions Over Time (ECI Frame)','fontsize', ti);
nexttile
plot(tspan_p/86400,X_SAT_nom(1,:)-X_SC(1,:));
hold on
grid on
plot(tspan_p/86400,X_SAT(1,:)-X_SC(1,:));
legend('$X_{free}$','$X_{c}$', 'Interpreter', 'Latex','fontsize', 11);
xlabel('Time [days]','fontsize', ax);
ylabel('X [km]','fontsize', ax);
nexttile
plot(tspan_p/86400,X_SAT_nom(2,:)-X_SC(2,:));
hold on
grid on
plot(tspan_p/86400,X_SAT(2,:)-X_SC(2,:));
legend('$Y_{free}$','$Y_{c}$', 'Interpreter', 'Latex');
xlabel('Time [days]','fontsize', ax);
ylabel('Y [km]','fontsize', ax);
nexttile
plot(tspan_p/86400,X_SAT_nom(3,:)-X_SC(3,:));
hold on
grid on
plot(tspan_p/86400,X_SAT(3,:)-X_SC(3,:));
legend('$Z_{free}$','$Z_{c}$', 'Interpreter', 'Latex');
ylabel('Z [km]','fontsize', ax);
xlabel('Time [days]','fontsize', ax);

% Display constraints and optimal solution
control_positions = -F*x0 - H*u_opt_sub - J*D;
thrust_magnitude = zeros(length(N),1);
thrust_magnitude(1) = norm(u0);
thrustX = zeros(length(N),1);
thrustX(1) = norm(u0(1));
thrustY = zeros(length(N),1);
thrustY(1) = norm(u0(2));
thrustZ = zeros(length(N),1);
thrustZ(1) = norm(u0(3));
E = zeros(6,length(N));
E(:,1) = x0;
Xlower = zeros(6,length(N));
Xlower(:,1) = Xlb_set;
Xupper = zeros(6,length(N));
Xupper(:,1) = Xub_set;
for i = 1:N-1
    thrust_magnitude(i+1) = norm(u_opt_sub(3*i-2:3*i,1));
    thrustX(i+1) = u_opt_sub(3*i-2,1);
    thrustY(i+1) = u_opt_sub(3*i-1,1);
    thrustZ(i+1) = u_opt_sub(3*i,1);
    E(:,i+1) = control_positions(6*i-5:6*i);
    Xlower(:,i+1) = Xlb(6*i-5:6*i);
    Xupper(:,i+1) = Xub(6*i-5:6*i);
end

figure(5)
fig5 = tiledlayout(3,1);
title(fig5,'Resultant Equinoctial Orbital Elements wrt Constraints','fontsize', Ti);
nexttile
plot(tspan_c/86400,E(1,:), 'linewidth', 3, 'Color', 'red');
hold on
plot(tspan_c/86400,Xlower(1,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(1,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E1','fontsize', ax);
grid on

nexttile
plot(tspan_c/86400,E(2,:), 'linewidth', 3, 'Color', 'green');
hold on
plot(tspan_c/86400,Xlower(2,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(2,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E2','fontsize', ax);
grid on

nexttile
plot(tspan_c/86400,E(3,:), 'linewidth', 3, 'Color', 'blue');
hold on
plot(tspan_c/86400,Xlower(3,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(3,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E3','fontsize', ax);
xlabel('Time [days]','fontsize', ax);
grid on

figure(6)
fig6 = tiledlayout(3,1);
title(fig6,'Resultant Equinoctial Orbital Elements wrt Constraints','fontsize', Ti);
nexttile
plot(tspan_c/86400,E(4,:), 'linewidth', 3, 'Color', 'yellow');
hold on
plot(tspan_c/86400,Xlower(4,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(4,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E4','fontsize', ax);
grid on

nexttile
plot(tspan_c/86400,E(5,:), 'linewidth', 3, 'Color', 'magenta');
hold on
plot(tspan_c/86400,Xlower(5,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(5,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E5','fontsize', ax);
grid on

nexttile
plot(tspan_c/86400,E(6,:), 'linewidth', 3, 'Color', 'cyan');
hold on
plot(tspan_c/86400,Xlower(6,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
plot(tspan_c/86400,Xupper(6,:), 'linewidth', 3, 'Linestyle', '--', 'Color', 'black');
ylabel('E6','fontsize', ax);
grid on
xlabel('Time [days]','fontsize', ax);

figure(7)
stairs(tspan_c/86400, thrust_magnitude*1000, 'linewidth', 3, 'Color', 'green');
hold on
xlabel('Time [days]','fontsize', ax);
ylabel('|U| [mN]','fontsize', ax);
title('Thrust Magnitude','fontsize', ti);
grid on


figure(8)
fig8 = tiledlayout(3,1);
title(fig8, 'Directional Thrust Values (RTN Frame)','fontsize', Ti);
nexttile
stairs(tspan_c/86400,thrustX*1000);
hold on
grid on
ylabel('$U_X$ [mN]','Interpreter','Latex','fontsize', ax);
nexttile
stairs(tspan_c/86400,thrustY*1000);
hold on
grid on
ylabel('$U_Y$ [mN]','Interpreter','Latex','fontsize', ax);
nexttile
stairs(tspan_c/86400,thrustZ*1000);
hold on
grid on
ylabel('$U_Z [mN]$','Interpreter','Latex','fontsize', ax);
xlabel('Time [days]','fontsize', ax);

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
    % f3(U) = -U + Umin <= 0 *Applied in projection function below
    % f4(U) = U - Umax <= 0 *Applied in projection function below
    global f1 f2
    c = [f1(u); f2(u)];
    ceq = [];
end

function g = nonsmooth_obj_subgradient(u)
    global N
    %Gradient of the objective function f(u)
    g = ones(3*(N-1),1);
end

function u_proj = project_with_fmincon_nonsmooth(u_update, Umin, Umax)
    % use fmincon to project u_update onto the feasible set Q
    options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');
    
    % Objective function: minimize ||u - u_update||
    objective = @(u) norm(u - u_update);
    
    % Define maximum and minimum constraint in A and b
    % f3(U) = -U + Umin <= 0
    % f4(U) = U - Umax <= 0
    A = zeros(length(u_update),length(u_update));
    b = zeros(length(u_update),1);
    Aeq = zeros(length(u_update),length(u_update));
    beq = zeros(length(u_update),1);
    
    for i = 1:length(u_update)
        if norm(u_update(i)) > Umin
            A(i,i) = u_update(i)/norm(u_update(i));
            b(i) = Umax;
        elseif norm(u_update(i)) > 0
            Aeq(i,i) = 1;
            beq(i) = sign(u_update(i))*Umin;
        end
    end
    
    % Solve the projection problem with fmincon
    u_proj = fmincon(objective, u_update, A, b, Aeq, beq, [], [], [], options);
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
