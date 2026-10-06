# Ride-service world model with all information supplied as input

## 1. Objective

Learn the conditional distribution of the evolution of a ride-service system under supplied platform actions:

$$
\boxed{
p_\Theta\!\left(
O_{t+1:t+H},E_{t+1:t+H}
\mid I_t,A_{t:t+H-1}
\right).
}
\tag{1}
$$

Here, $I_t$ contains all supplied information available at prediction time, $A$ denotes platform actions, $E$ denotes world events, and $O$ denotes observations. $\Theta$ collects the trainable parameters. The primary objective is predictive fidelity of the world dynamics. Fairness statistics are trajectory observables. Conditioning on a future action sequence denotes a controlled rollout: each transition uses only its current action and preceding context.

## 2. Time, entities, and world state

Let $t\in\{0,1,\ldots,T\}$ index intervals of duration $\Delta$. The timestamp at boundary $t$ is $\tau_t$. A decision occurs after observations at $\tau_t$ and before events in $(\tau_t,\tau_{t+1}]$.

Let

$$
G=(V,E_G),\qquad \mathcal Z=\{1,\ldots,Z\},
$$

be the directed road graph and its zone partition. Edge attributes include length and road restrictions; time-varying travel conditions belong to the dynamic state.

The entity sets are drivers $\mathcal D_t$, customers $\mathcal C_t$, and requests $\mathcal R_t$. A customer is a passenger; a driver is the service provider. One customer may generate multiple requests over time. The service considered here is non-pooled: a driver serves at most one active trip at a time.

Define the underlying state as

$$
S_t=(X_t,Z_t),
\qquad
X_t=(D_t,C_t,R_t,L_t,W_t,\Gamma_t),
\tag{2}
$$

where $Z_t$ contains unobserved behavioral and environmental factors. The structured components are:

| Component | Variables |
|---|---|
| $D_t$ | Driver identities, positions, online status, service status, assigned request, route progress, vehicle attributes, costs, ratings, and activity counters. |
| $C_t$ | Customer identities, available profile attributes, ratings, active request references, expenditure, and service-history counters. |
| $R_t$ | Request identities, customer references, creation times, origins, destinations, quoted fares, deadlines where recorded, offer references, and lifecycle status. |
| $L_t$ | Time-stamped payment, cost, working-time, rating, and service-event ledgers. |
| $W_t$ | Traffic, weather, events, zone queues, supply, demand history, and slower regional usage variables where observed. |
| $\Gamma_t$ | Current platform rules: pricing, commission, eligibility, service area, and matching-policy configuration. |

Driver position $\ell_{i,t}$ is a node or an edge with progress along that edge. Driver status belongs to

$$
\{\text{offline},\text{idle},\text{offered},\text{pickup},
\text{occupied},\text{repositioning}\}.
$$

Request status belongs to

$$
\{\text{pending},\text{offered},\text{accepted},\text{pickup},
\text{in-trip},\text{completed},\text{cancelled},\text{expired}\}.
$$

An observation $O_t$ contains the recorded subset of state variables and events available by $\tau_t$. Missingness and reporting times are retained explicitly.

## 3. Complete input specification

At every prediction call, the caller supplies

$$
\boxed{
I_t=(G,\mathcal Z,\tau_t,\Delta,P_t^D,P_t^C,
O_{0:t},A_{0:t-1},\Gamma_{0:t},K_t,M_{0:t}).
}
\tag{3}
$$

There is no persistent entity database queried by the model. All entity records and history used at this call are represented in $I_t$.

| Input | Formal contents |
|---|---|
| Road and zone information $(G,\mathcal Z)$ | Nodes, directed edges, lengths, restrictions, zone memberships, and recorded zone attributes. |
| Clock $(\tau_t,\Delta)$ | Current timestamp and simulation interval; calendar features are deterministic functions of $\tau_t$. |
| Driver records $P_t^D=\{p_{i,t}^D\}$ | ID, vehicle type and capacity, known operating-cost coefficients, tenure, rating sum/count, and other observed profile attributes as of $t$. |
| Customer records $P_t^C=\{p_{j,t}^C\}$ | ID, tenure, rating sum/count, and other observed customer attributes as of $t$. |
| Driver observations in $O_t$ | Location, online/idle/busy status, current offer/trip, route progress, known schedule, time since last trip, and observed earnings/time counters. |
| Customer/request observations in $O_t$ | Current requests, customer IDs, origins/destinations, elapsed wait, quotes, offer/acceptance/cancellation status, and recorded deadlines. |
| Environmental observations in $O_t$ | Current traffic and edge travel times, weather, demand/supply summaries, zone queues, and recorded regional context. |
| History $O_{0:t},A_{0:t-1}$ | Time-stamped observations, platform offers and dispatches, responses, movements, completed/failed trips, receipts, costs, working time, and rating events. |
| Rule history $\Gamma_{0:t}$ | The platform settings effective at each historical time and at the current boundary. |
| Known future context $K_t$ | Schedules and announcements already available at $t$, including their issue times and effective times. |
| Availability data $M_{0:t}$ | Per-field observation masks, event timestamps, receipt timestamps, and measured/estimated provenance. |

For an input field $v$ at time $k$, encode

$$
\widetilde v_k=(m_{v,k}v_k,m_{v,k},a_{v,k}),
\quad m_{v,k}\in\{0,1\},
\tag{4}
$$

where $a_{v,k}$ is age since the last available measurement. An unavailable numeric value uses a placeholder together with $m_{v,k}=0$.

For every input record $r$,

$$
t_{\mathrm{available}}(r)\le\tau_t.
\tag{5}
$$

Future realized demand, travel times, payments, ratings, and behavior are prediction targets. They enter $K_t$ only when they were actually known in advance, with that provenance retained.
