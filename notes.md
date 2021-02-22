TODO
- prove that K is invariant to goal
    - just feed back the task error to reach a set point!
        - the cost want's the state, whatever it is, to go to zero.
    - K, S, etc is translation invariant to the goal state
    - to build this translation into the state/dynamics, we augment the state/cost with the target
        - this just computes the error from that state internally
    - write about this:
        - L is linear in state, we can apply linear operations to it?
        - S is quadratic in state, but we can also linearly transform it?
        - value functions and controllers are additive... but still linear/quadratic
- gain intuition about matrix operations
    - control law computation
    - backwards recursion
- multiple targets?
    - this doesn't work by adding another target, doesn't converge
    - does this make sense? would need to decide between targets?
        - can't optimize for two errors simultaneously?
- make time, force, etc into reasonable units for interpretability

Goals
- finish cleaning this up
    - X infinite horizon lqr
    - "" "" with multiplicative noise
- model adaptation gradient
    - model corruption
    - how to compute gradient? 
    - how to check if A is diagonalizable?
- write up discussion of LQR
    - the need to compose/arbitrate LQR policies
        - goal uncertainty
        - multiple goals + perturbation
- analyze andy's data
    - draw specific plots i want
    - behavioral trajectories
    - EMG features -- low-dimensional trajectories
    - comparisons within sessions
    - comparisons across sessions