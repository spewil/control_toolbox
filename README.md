# control_toolbox

# LQR 
### Parameters  

Finite-Horizon Dynamics

<img src="https://render.githubusercontent.com/render/math?math=x_{t+1} = A_tx_t + B_tu_t>

<img src="https://render.githubusercontent.com/render/math?math=t \\in \\{0,...,T\\}>

Cost to minimize

<img src="https://render.githubusercontent.com/render/math?math=J_{T}(u) = \\sum_{t=0}^{N-1}{x_t^TQ_tx_t + u_t^TR_tu_t + x_T^TQ_Tx_T}>

Control Law

<img src="https://render.githubusercontent.com/render/math?math=u_t^*(x) = -L_tx>

<img src="https://render.githubusercontent.com/render/math?math=L_t = -(R_t+B^TS_{t+1}B)^{-1}(B^TS_{t+1}A)x>

Optimal Cost-to-Go Solution

<img src="https://render.githubusercontent.com/render/math?math=S_t = A^TS_{t+1}A - (A^TS_{t+1}B)(R + B^TS_{t+1}B)^{-1}(B^TS_{t+1}A) + Q_t>

<img src="https://render.githubusercontent.com/render/math?math=S_T = Q_T>
