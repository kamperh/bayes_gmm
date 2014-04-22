"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014
"""

class NIW(object):
    """A normal-inverse-Wishart distribution."""
    def __init__(self, m_0, k_0, v_0, S_0):
        self.m_0 = m_0
        self.k_0 = k_0
        D = len(m_0)
        assert v_0 >= D, "v_0 must be larger or equal to dimension of data"
        self.v_0 = v_0
        self.S_0 = S_0
