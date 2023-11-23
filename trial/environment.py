import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from scipy.integrate import solve_ivp


class ship_environment(py_environment.PyEnvironment):

    def __init__(self):

        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=-35*np.pi/180, maximum=35*np.pi/180, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=[-20, -np.pi, 0, -1],
                                                             maximum=[20, np.pi, 50, 1], name='observation')
        self.episode_ended = False
        self.counter = 0
        self.random_x = 0
        self.random_y = 0
        self.distance = 0
        self.obs_state = [0, 0, 0, 0, 0, 0, 0]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action):

        delta_c=action

        def KCS_ode(t, v):

            # Ship Geometry and Constants
            L = 230
            B = 32.2
            d_em = 10.8
            rho = 1025
            g = 9.80665

            Cb = 0.651
            Dsp = Cb * L * B * d_em
            Fn = 0.26
            U_des = Fn * np.sqrt(g * L)
            xG = -3.404
            kzzp = 0.25

            # Surge Hydrodynamic Derivatives in non-dimensional form
            X0 = -0.0167
            Xbb = -0.0549
            Xbr_minus_my = -0.1084
            Xrr = -0.0120
            Xbbbb = -0.0417

            # Sway Hydrodynamic Derivatives in non-dimensional form
            Yb = 0.2252
            Yr_minus_mx = 0.0398
            Ybbb = 1.7179
            Ybbr = -0.4832
            Ybrr = 0.8341
            Yrrr = -0.0050

            # Yaw Hydrodynamic Derivatives in non-dimensional form
            Nb = 0.1111
            Nr = -0.0465
            Nbbb = 0.1752
            Nbbr = -0.6168
            Nbrr = 0.0512
            Nrrr = -0.0387

            n_prop = 115.5 / 60
            Dp = 7.9
            wp = 1 - 0.645  # Effective Wake Fraction of the Propeller
            tp = 1 - 0.793  # Thrust Deduction Factor

            eps = 0.956
            eta = 0.7979
            kappa = 0.633
            xp_P = -0.4565  # Assuming propeller location is 10 m ahead of AP (Rudder Location)
            xp_R = -0.5

            mp = Dsp / (0.5 * (L ** 2) * d_em)
            xGp = xG / L
            # Added Mass and Mass Moment of Inertia (from MDLHydroD)
            mxp = 1790.85 / (0.5 * (L ** 2) * d_em)
            myp = 44324.18 / (0.5 * (L ** 2) * d_em)
            Jzzp = 140067300 / (0.5 * (L ** 4) * d_em)
            Izzp = mp * (kzzp ** 2) + mp * (xGp ** 2)

            a0 = 0.5228
            a1 = -0.4390
            a2 = -0.0609

            tR = 1 - 0.742
            aH = 0.361
            xp_H = -0.436

            A_R = L * d_em / 54.86
            Lamda = 2.164
            f_alp = 6.13 * Lamda / (2.25 + Lamda)

            # Nondimensional State Space Variables
            up = v[0]
            vp = v[1]
            rp = v[2]
            xp = v[3]
            yp = v[4]
            psi = v[5]
            delta = v[6]
            # n_prop = v[7]

            # Derived kinematic variables
            b = np.arctan2(-vp, up)  # Drift angle

            # Non-dimensional Surge Hull Hydrodynamic Force
            Xp_H = X0 * (up ** 2) \
                   + Xbb * (b ** 2) + Xbr_minus_my * b * rp \
                   + Xrr * (rp ** 2) + Xbbbb * (b ** 4)

            # Non-dimensional Sway Hull Hydrodynamic Force
            Yp_H = Yb * b + Yr_minus_mx * rp + Ybbb * (b ** 3) \
                   + Ybbr * (b ** 2) * rp + Ybrr * b * (rp ** 2) \
                   + Yrrr * (rp ** 3)

            # Non-dimensional Yaw Hull Hydrodynamic Moment
            Np_H = Nb * b + Nr * rp + Nbbb * (b ** 3) \
                   + Nbbr * (b ** 2) * rp + Nbrr * b * (rp ** 2) \
                   + Nrrr * (rp ** 3)

            # Propulsion Force Calculation

            # The value self propulsion RPM is taken from Yoshimura's SIMMAN study
            # Analysis of steady hydrodynamic force components and prediction of
            # manoeuvering ship motion with KVLCC1, KVLCC2 and KCS

            J = (up * U_des) * (1 - wp) / (n_prop * Dp)  # Advance Coefficient

            Kt = a0 + a1 * J + a2 * (J ** 2)  # Thrust Coefficient

            # Dimensional Propulsion Force
            X_P = (1 - tp) * rho * Kt * (Dp ** 4) * (n_prop ** 2)

            # Non-dimensional Propulsion Force
            Xp_P = X_P / (0.5 * rho * L * d_em * (U_des ** 2))

            # Rudder Force Calculation

            b_p = b - xp_P * rp

            if b_p > 0:
                gamma_R = 0.492
            else:
                gamma_R = 0.338

            lp_R = -0.755

            up_R = eps * (1 - wp) * up * np.sqrt(eta * (1 + kappa * \
                                                        (np.sqrt(1 + 8 * Kt / (np.pi * (J ** 2))) - 1)) ** 2 + (
                                                             1 - eta))

            vp_R = gamma_R * (vp + rp * lp_R)

            Up_R = np.sqrt(up_R ** 2 + vp_R ** 2)
            alpha_R = delta - np.arctan2(-vp_R, up_R)

            F_N = A_R / (L * d_em) * f_alp * (Up_R ** 2) * np.sin(alpha_R)

            Xp_R = - (1 - tR) * F_N * np.sin(delta)
            Yp_R = - (1 + aH) * F_N * np.cos(delta)
            Np_R = - (xp_R + aH * xp_H) * F_N * np.cos(delta)

            # Coriolis terms

            mp = Dsp / (0.5 * (L ** 2) * d_em)
            xGp = xG / L

            Xp_C = mp * vp * rp + mp * xGp * (rp ** 2)
            Yp_C = -mp * up * rp
            Np_C = -mp * xGp * up * rp

            # Net non-dimensional force and moment computation
            Xp = Xp_H + Xp_R + Xp_C + Xp_P
            Yp = Yp_H + Yp_R + Yp_C
            Np = Np_H + Np_R + Np_C

            # Net force vector computation in Abkowitz non-dimensionalization
            X = Xp
            Y = Yp
            N = Np

            # Added Mass and Mass Moment of Inertia (from MDLHydroD)
            mxp = 1790.85 / (0.5 * (L ** 2) * d_em)
            myp = 44324.18 / (0.5 * (L ** 2) * d_em)
            Jzzp = 140067300 / (0.5 * (L ** 4) * d_em)
            Izzp = mp * (kzzp ** 2) + mp * (xGp ** 2)

            Mmat = np.zeros((3, 3))

            Mmat[0, 0] = mp + mxp
            Mmat[1, 1] = mp + myp
            Mmat[2, 2] = Izzp + Jzzp
            Mmat[1, 2] = mp * xGp
            Mmat[2, 1] = mp * xGp

            Mmatinv = np.linalg.inv(Mmat)

            tau = np.array([X, Y, N])

            vel_der = Mmatinv @ tau

            # Derivative of state vector
            vd = np.zeros(7)

            vd[0:3] = vel_der
            vd[3] = up * np.cos(psi) - vp * np.sin(psi)
            vd[4] = up * np.sin(psi) + vp * np.cos(psi)
            vd[5] = rp

            # # Commanded Rudder Angle
            # delta_c = KCS_rudder_angle(t, v)

            T_rud = 0.1  # Corresponds to a time constant of 1 * L / U_des = 20 seconds
            deltad = (delta_c - delta) / T_rud

            deltad_max = 5 * np.pi / 180 * (L / U_des)  # Maximum rudder rate of 5 degrees per second

            # Rudder rate saturation
            if np.abs(deltad) > deltad_max:
                deltad = np.sign(deltad) * deltad_max

            vd[6] = deltad

            return vd

        tspan = (0, 0.3)
        yinit = self.obs_state

        sol = solve_ivp(KCS_ode, tspan, yinit, t_eval=tspan, dense_output=True)

        # sol.y outputs 0)surge vel 1) sway vel 2) yaw vel 3)psi 4) x_coord 5) y_coord

        self.obs_state[0] = sol.y[0][-1]  # surge vel
        self.obs_state[1] = sol.y[1][-1]  # sway vel
        self.obs_state[2] = sol.y[2][-1]  # yaw vel
        rad = sol.y[5][-1]  # psi
        rad = rad % (2 * np.pi)
        u = self.obs_state[0]
        v = self.obs_state[1]

        self.obs_state[5] = rad
        self.obs_state[3] = sol.y[3][-1]
        self.obs_state[4] = sol.y[4][-1]
        self.obs_state[6] = sol.y[6][-1]

        # REWARD FUNCTIONS

        x_init = 0
        y_init = 0
        x_goal = self.random_x
        y_goal = self.random_y
        x = self.obs_state[3]
        y = self.obs_state[4]
        psi = self.obs_state[5]
        r = self.obs_state[2]

        distance = ((x - x_goal) ** 2 + (y - y_goal) ** 2) ** 0.5

        # SIDE TRACK
        vec1 = np.array([x_goal - x_init, y_goal - y_init])
        vec2 = np.array([x_goal - x, y_goal - y])

        vec1 = vec1 / np.linalg.norm(vec1)
        cross_pro = np.cross(vec2, vec1)
        side_track_error = cross_pro

        # Termination
        x_dot = u * np.cos(psi) - v * np.sin(psi)
        y_dot = u * np.sin(psi) + v * np.cos(psi)
        vec3 = np.array([x_dot, y_dot])
        vec3 = vec3 / np.linalg.norm(vec3)
        vec2 = vec2 / np.linalg.norm(vec2)
        angle_btw23 = np.arccos(np.dot(vec2, vec3))
        angle_btw12 = np.arccos(np.dot(vec1, vec2))

        course_angle_err = np.arcsin(np.cross(vec2, vec3))

        R1 = 2 * np.exp(-0.08 * side_track_error ** 2) - 1
        # R1=2*np.exp (-0.25*abs(side_track_error))-1

        R2 = 1.3 * np.exp(-10 * (abs(course_angle_err))) - 0.3
        # R2= 2*np.exp (-5*(course_angle_err**2))-1

        R3 = -distance * 0.25

        reward = R1 + R2 + R3

        self.distance = distance

        observation = [side_track_error, course_angle_err, distance, r]

        # DESTINATION CHECK

        if abs(distance) <= 0.5:
            reward = 100
            self.episode_ended=True
            print("Destination reached")
            return ts.termination(np.array(observation, dtype=np.float32), reward)

        # # BOUNDARY CHECK

        if angle_btw12 > np.pi / 2 and angle_btw23 > np.pi / 2:
            self.episode_ended = True
            # print('terminated')
            return ts.termination(np.array(observation, dtype=np.float32), reward)
            # return ts.transition(np.array(observation, dtype=np.float32), reward, discount=0.95)
        else:
            return ts.transition(np.array(observation, dtype=np.float32), reward)

    def _reset(self):

        # print("Next episode")

        self.obs_state = [1, 0, 0, 0, 0, 0, 0]
        self.episode_ended = False

        radius=np.random.randint(8,28)
        # radius = 8 + (self.counter//500)
        random_theta = 2 * np.pi * np.random.rand()

        self.random_x = radius * np.cos(random_theta)
        self.random_y = radius * np.sin(random_theta)

        print("GOAL COORD", self.random_x, self.random_y)

        x_goal = self.random_x
        y_goal = self.random_y

        x_dot = 1
        y_dot = 0

        vec2 = np.array([x_goal, y_goal])
        vec3 = np.array([x_dot, y_dot])

        vec3 = vec3 / np.linalg.norm(vec3)
        vec2 = vec2 / np.linalg.norm(vec2)

        course_angle_err = np.arcsin(np.cross(vec2, vec3))

        # psip = np.arctan2(y_goal, x_goal)
        #
        # psi_ship=0

        # course_angle_err = psip - psi_ship
        # if course_angle_err>np.pi:
        #     course_angle_err=course_angle_err-2*np.pi
        # if course_angle_err<-np.pi:
        #     course_angle_err=course_angle_err+2*np.pi

        self.counter = self.counter + 1
        observation = np.array([0, course_angle_err, radius, 0], dtype=np.float32)

        return ts.restart(observation)


# ENDS HERE
