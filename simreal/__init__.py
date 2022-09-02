# include all gym dependencies
from gym.core import Env, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
from gym.spaces import Space

# point to the sim register
from simreal.envs import make, spec, register

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]


# Simreal
from simreal.rlbase.agent import BaseAgent
from simreal.rlbase.agent import RandomAgent

from simreal.rlbase.data import StepType
from simreal.rlbase.data import TimeStep
from simreal.rlbase.data import Trajectory

from simreal.rlbase.engine import BaseEngine

from simreal.rlbase.es import BaseES
from simreal.rlbase.es import CMAES
from simreal.rlbase.es import CEM

from simreal.rlbase.logger import Logger

from simreal.rlbase.runner import BaseRunner
from simreal.rlbase.runner import EpisodeRunner
from simreal.rlbase.runner import StepRunner

from simreal.sim2real.domain_randomization_wrapper import DomainRandomization

SEEDS = [1084389005, 1096831319, 1107694401, 1117177635, 1129282672,
         1140653297, 1150794856, 1160564931, 1172473350, 1182646876,
         1194869402, 1204124363, 1215002816, 1224696448, 1234004931,
         1244955702, 1253984800, 1264542370, 1275783021, 1286407000,
         1297948033, 1308368048, 1318561877, 1328120147, 1340427696,
         1350227676, 1359860441, 1369776250, 1381816286, 1392715469,
         1402458368, 1413180126, 1422627425, 1435082944, 1447324129,
         1456328541, 1465779442, 1475999157, 1486310528, 1494679302,
         1506172141, 1518028617, 1527780914, 1538526038, 1549325168,
         1560536970, 1571383129, 1581495367, 1592889501, 1602245422,
         1614303361, 1624632858, 1633118977, 1642094049, 1654421916,
         1665793034, 1677022745, 1686043992, 1697281408, 1706820597,
         1718965098, 1728564048, 1739519808, 1750007859, 1761378110,
         1771784242, 1781137871, 1791372216, 1800938711, 1810365863,
         1820490687, 1831406387, 1845967304, 1856705628, 1869071543,
         1879559156, 1890816383, 1900393289, 1909815351, 1918608275,
         1932280555, 1943384584, 1953589340, 1965014885, 1975516203,
         1987315053, 1998853165, 2010844016, 2022228780, 2031808701,
         2042192235, 2053396598, 2066616655, 2077143627, 2086966747,
         2097132773, 2108167629, 2118561231, 2128474801, 2138280732]