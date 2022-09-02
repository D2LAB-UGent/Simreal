"""classic Acrobot task"""
import numpy as np
from numpy import sin, cos, pi

from gym import core, spaces
from gym.utils import seeding

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class AcrobotEnv(core.Env):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    CHANGED TO CONTINUOUS ACTION SPACE between -max_torque and +max_torque
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    def __init__(self):

        self.dt = .025

        # init parameters, overwritten by domain randomization wrapper:
        # ALSO ADAPT IN STEP!

        #updated acrobot
        self.params = {'l1': 0.22, 'l2': 0.27, 'm1': 3.5, 'm2': 0.9}

        #old acrobot
        # self.params = {'l1': 0.153, 'l2': 0.175, 'm1': 0.135, 'm2': 0.12}

        self.LINK_LENGTH_1 = self.params['l1']  # [m]
        self.LINK_LENGTH_2 = self.params['l2']  # [m]
        self.LINK_MASS_1 = self.params['m1']  #: [kg] mass of link 1
        self.LINK_MASS_2 = self.params['m2']  #: [kg] mass of link 2

        #updated acrobot
        self.LINK_COM_POS_1 = -0.003  #: [m] position of the center of mass of link 1
        self.LINK_COM_POS_2 = 0.034  #: [m] position of the center of mass of link 2
        self.LINK_MOI_1 = 0.057  #: moments of inertia for link 1
        self.LINK_MOI_2 = 0.007  #: moments of inertia for link 2

        #old acrobot
        # self.LINK_COM_POS_1 = 0.0875  #: [m] position of the center of mass of link 1
        # self.LINK_COM_POS_2 = 0.93  #: [m] position of the center of mass of link 2
        # self.LINK_MOI_1 = 5.026*10**-4  #: moments of inertia for link 1
        # self.LINK_MOI_2 = 3.39*10**-4  #: moments of inertia for link 2

        self.MAX_VEL_1 = 20
        self.MAX_VEL_2 = 20

        self.max_torque = 4.0  # 1.4

        # friction for noisy env
        self.friction = 0

        #: use dynamics equations from the nips paper or the book
        self.action_arrow = None
        self.domain_fig = None
        self.actions_num = 3

        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        var = 0
        theta1_start = np.random.uniform(0 - var, 0 + var)
        theta2_start = np.random.uniform(-var, var)
        dtheta1_start = np.random.uniform(-4.5 * var, 4.5 * var)
        dtheta2_start = np.random.uniform(-5 * var, 5 * var)

        self.state = np.array([theta1_start, theta2_start, dtheta1_start, dtheta2_start])
        # self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        return self._get_ob()

    # def tip(self, theta1, theta2, l1, l2):
    #     x = l1*cos(theta1-pi/2.0) + l2*cos(theta2+theta1-pi/2.0)
    #     y = l1*sin(theta1-pi/2.0) + l2*sin(theta2+theta1-pi/2.0)
    #     return np.array([x, y])

    def step(self, a):

        # before step check possible randomized parameters
        self.LINK_LENGTH_1 = self.params['l1']  # [m]
        self.LINK_LENGTH_2 = self.params['l2']  # [m]
        self.LINK_MASS_1 = self.params['m1']  #: [kg] mass of link 1
        self.LINK_MASS_2 = self.params['m2']  #: [kg] mass of link 2

        #updated acrobot
        self.LINK_COM_POS_1 = -0.003  #: [m] position of the center of mass of link 1
        self.LINK_COM_POS_2 = 0.034  #: [m] position of the center of mass of link 2
        self.LINK_MOI_1 = 0.057  #: moments of inertia for link 1
        self.LINK_MOI_2 = 0.007  #: moments of inertia for link 2

        #old acrobot
        # self.LINK_COM_POS_1 = 0.0875  #: [m] position of the center of mass of link 1
        # self.LINK_COM_POS_2 = 0.93  #: [m] position of the center of mass of link 2
        # self.LINK_MOI_1 = 5.026*10**-4  #: moments of inertia for link 1
        # self.LINK_MOI_2 = 3.39*10**-4  #: moments of inertia for link 2

        s = self.state

        torque = np.clip(a, -self.max_torque, self.max_torque)[0]

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        #TASK 1
        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        return (self._get_ob(), reward, terminal, {})

        #TASK 2
        # theta1 = self.state[0]
        # theta2 = self.state[1]
        # l1 = self.LINK_LENGTH_1
        # l2 = self.LINK_LENGTH_2
        # theta1_des = pi
        # theta2_des = 0
        # reward = -np.linalg.norm(self.tip(theta1, theta2, l1, l2) - self.tip(theta1_des, theta2_des, l1, l2))
        # return (self._get_ob(), reward, False, {})

    def _get_ob(self):
        s = self.state
        return np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])

    def _terminal(self):
        s = self.state
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.)

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2) + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2

        #friction added !
        µ_1 = self.friction
        µ_2 = self.friction
        ddtheta2 = (a + d2 / d1 * phi1 + d2 / d1 * µ_1 * dtheta1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - µ_2 * dtheta2 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + µ_1 * dtheta1 + phi1) / d1

        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def render(self, mode='human'):
        from simreal.envs.classic_control import rendering

        s = self.state

        initial = {'l1': 0.22, 'l2': 0.27, 'm1': 3.5, 'm2': 0.9}  # updated acrobot
        # initial = {'l1': 0.153, 'l2': 0.175, 'm1': 0.135, 'm2': 0.12} #old acrobot

        scale_factor = (initial['l1'] + initial['l2'])/2
        bound = initial['l1'] + initial['l2'] + 0.2 * scale_factor  # 2.2 for default

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]- pi/2, s[0]+s[1]-pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        goal = (self.LINK_LENGTH_1+self.LINK_LENGTH_2)/2
        self.viewer.draw_line((-bound, goal), (bound, goal))
        #TASK 2 plot
        # tip = self.tip(s[0], s[1], self.LINK_LENGTH_1, self.LINK_LENGTH_2)
        # tip_des = self.tip(pi, 0, self.LINK_LENGTH_1, self.LINK_LENGTH_2)
        # self.viewer.draw_line((tip[0], tip[1]), (tip_des[0], tip_des[1]))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1*scale_factor, -.1*scale_factor
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1*scale_factor)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0


    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
