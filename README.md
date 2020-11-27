# control_toolbox

# LQR 
### Parameters  

Finite-Horizon Dynamics

x_{t+1} = A_tx_t + B_tu_t

t \\in \\{0,...,T\\}

Cost to minimize

J_{T}(u) = \\sum_{t=0}^{N-1}{x_t^TQ_tx_t + u_t^TR_tu_t + x_T^TQ_Tx_T}

Control Law

u_t^*(x) = -L_tx

L_t = -(R_t+B^TS_{t+1}B)^{-1}(B^TS_{t+1}A)x

Optimal Cost-to-Go Solution

S_t = A^TS_{t+1}A - (A^TS_{t+1}B)(R + B^TS_{t+1}B)^{-1}(B^TS_{t+1}A) + Q_t

S_T = Q_T
