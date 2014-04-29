# -*- coding: utf-8 -*-

"""
TODO: Periodic boundary conditions, in time and space, etc.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from .exceptions import PycelleException

try:
    from _caalgo import eca_cyevolve
except ImportError:
    eca_cyevolve = None

try:
    from _lightcones import lightcone_counts
except ImportError:
    lightcone_counts = None

__all__ = [
    'ECA',
]
if lightcone_counts:
    __all__.append('lightcone_counts')

class ECA(object):
    def __init__(self, rule, shape, ic=None):
        """Initialize the ECA.

        Parameters
        ----------
        rule : int
            The ECA rule to initialize.
        shape : 2-tuple
            The shape of the spacetime array.
        ic : array | str | None
            The initial condition of the ECA. If `None`, then 'single' is used.

        Examples
        --------
        >>> x = ECA(54, (64,64))
        >>> y = ECA(54, (32,32), 'random')

        """
        if rule < 0 or rule >= 256:
            raise Exception('Rule must be between 0 and 255, inclusive.')
        else:
            self.rule = rule

            # Privately, store a list of the binary bits
            self._lookup = map(int, '{0:08b}'.format(rule))

        # spacetime array
        self.ic = None
        self.t = 0
        self._sta = np.zeros(shape, dtype=np.uint8, order='C')
        self.initialize(ic, clear=False)

        if eca_cyevolve is not None:
            self._cythonized = True
        else:
            self._cythonized = False

        # Drawing defaults
        self._update_extent = True


    def __repr__(self):
        return 'ECA({0})'.format(self.rule)


    def draw(self, ax=None):
        """Draw the current spacetime array.

        Parameters
        ----------
        ax : Matplotlib Axes | None
            The axis to receive the plot.

        """
        if ax is None:
            ax = plt.gca()

        t, sta = self.t, self._sta

        # We do not call get_spacetime() in order to avoid a copy when possible.
        div, mod = divmod(t, sta.shape[0])
        if not div:
            # Then we have not "rolled" over. Show it all.
            arr = sta
        else:
            # Roll so that the current row becomes the last row.
            # This forces a copy!
            arr = np.roll(sta, sta.shape[0] - mod - 1, axis=0)

        if self._update_extent:
            if not div:
                extent = [0, sta.shape[1] - 1, sta.shape[0], 0]
            else:
                extent = [0, sta.shape[1] - 1, t, t - sta.shape[0]]
        else:
            extent = None

        ax.matshow(arr, cmap=plt.cm.gray_r, extent=extent)
        ax.set_title('Rule {0}'.format(self.rule))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$', rotation='horizontal')
        ax.xaxis.set_ticks_position('bottom')
        plt.draw()


    def eval(self, parents):
        """Returns the output of the ECA given the parents.

        Parameters
        ----------
        parents : NumPy array
            An array consisting of 3 elements, which specify the parents.

        Returns
        -------
        val : int
            The value of the cell immediately below the parents.

        """
        iparents = int(''.join(map(str, parents)), 2)
        return self.eval_int(iparents)


    def eval_int(self, parents):
        """Returns the output of the ECA given the parents as an integer.

        Parameters
        ----------
        parents : int
            An integer between 0 and 7, inclusive, that represents the
            the 3 parent cells.  Recall that in Wolfram's notation,
            111 (integer 7) corresponds is the first lookup in the rule.

        Returns
        -------
        val : int
            The value of the cell immediately below the parents.

        """
        if parents < 0 or parents > 7:
            raise Exception('Invalid parents: {0}'.format(parents))
        else:
            # Flip the index so that 7 is index 0.
            idx = -(parents - 7)
            return self._lookup[idx]


    def evolve(self, t=None, draw=True, **kwargs):
        """Evolves the cellular automaton from the inital row by t time steps.

        Parameters
        ----------
        t : int | None
            The number of times to evolve the cellular automaton. If `None`,
            then the cellular automaton is evolved to the last row in the
            spacetime array. If the cellular automaton is already on the last
            row, then it is evolved again until it returns to the last row.
        draw : bool
            If `True`, then draw the ECA after evolving it.

        Returns
        -------
        current_t : int
            The current time of the ECA.

        """
        if 'ic' in kwargs:
            self.initialize( kwargs.pop('ic') )

        if t is None:
            # If we are not the last row, evolve to the last row.
            nRows = self._sta.shape[0]
            mod = self.t % nRows
            if mod == nRows - 1:
                t = nRows
            else:
                t = nRows - 1 - mod

        if self._cythonized:
            self._evolve_cython(t)
        else:
            self._evolve_python(t)

        self.t += t

        if draw:
            self.draw(**kwargs)

        return self.t

    def _evolve_python(self, iterations):
        sta = self._sta
        nRows = sta.shape[0]
        for i in range(self.t, self.t + iterations):
            row = sta[i % nRows]
            # This makes 3 copies of the data. Oh well.
            windows = np.vstack([np.roll(row,1), row, np.roll(row,-1)])
            windows = windows.transpose()
            for j,parents in enumerate(windows):
                sta[(i+1) % nRows,j] = self.eval(parents) # slow


    def _evolve_cython(self, iterations):
        eca_cyevolve(self._lookup, self._sta, iterations, self.t)


    def get_spacetime(self):
        """Returns a copy of the spacetime array."""
        div, mod = divmod(self.t, self._sta.shape[0])
        if not div:
            # Then we have not "rolled" over. Make a copy.
            arr = self._sta.copy()
        else:
            # Roll so that the current row becomes the last row.
            # This forces a copy!
            arr = np.roll(self._sta, self._sta.shape[0] - mod - 1, axis=0)
        return arr


    def get_state(self, t=None):
        """Returns a view of the current state of the cellular automaton.

        Parameters
        ----------
        t : int | None
            If `t` is an integer, then the state of the cellular automaton
            at time `t` is returned. If `t` is `None`, then the current
            state of the cellular automaton is returned.

        Returns
        -------
        state : 1d array

        Raises
        ------
        CMPyException
            Raised for `t` greater than the current evolution of the automaton.

        """
        if t is None:
            t = self.t
        elif t > self.t:
            msg = 't must be less than or equal to {0}'.format(self.t)
            raise CMPyException(msg)

        return self._sta[t % self._sta.shape[0]]


    def get_tikzrule(self, boxes=True, numbers=True, rule=True, stand_alone=False):
        """Returns TiKz code for displaying the ECA rule.

        Parameters
        ----------
        boxes : bool
            If `True`, each mapping is enclosed in a box.
        numbers : bool
            If `True`, the output of each mapping is displayed as 0 or 1.
        rule : bool
            If `True`, the name of the rule is included in the TikZ code.
        stand_alone : bool
            If `True`, a standalone TeX document is returned.

        Returns
        -------
        tikz : str
            The TikZ code which displays the rule.

        """
        return get_tikzrule(self, boxes, numbers, rule, stand_alone)


    def initialize(self, ic=None, clear=True):
        """Initialize the ECA's spacetime array.

        A new spacetime array is allocated only if necessary.
        The first row of the array is populated with the initial condition.

        Parameters
        ----------
        ic : str | array
            A specification of how to initialize the spacetime array.
            If some other initialization is desired, one can explicitly
            set the first row of the spacetime array via self._sta[0].
            If `None`, then 'single' is used.

            Valid options:

                'random'
                    The spacetime array is initialized randomly.

                'single'
                    The spacetime array has a single, centered black cell.
                    Cell (0, floor(width/2)) is filled black.

        clear : bool
            If `True`, then the spacetime array is cleared before the first
            row is initialized.

        """
        if ic is None:
            ic = 'single'

        # Reset the array
        self.t = 0
        if clear:
            self._sta[1:] *= 0

        # Set the initial condition
        nRows, nCols = self._sta.shape
        if ic == 'random':
            self.ic = np.random.randint(0, 2, size=nCols).astype(np.uint8)
            self._sta[0] = self.ic.copy()
        elif ic == 'single':
            self.ic = np.zeros(nCols, dtype=np.uint8)
            self.ic[ int(np.floor(nCols/2)) ] = 1
            self._sta[0] = self.ic.copy()
        else:
            try:
                self.ic = np.array(list(ic))
            except:
                raise PycelleException('Invalid `ic` specificiation.')
            else:
                # An explicit copy is required to avoid a view of the row
                self._sta[0] = self.ic.copy()

    def reset(self):
        """Resets the ECA back to its initial condition."""
        self.t = 0
        self._sta *= 0
        self._sta[0] = self.ic.copy()

def get_tikzrule(eca, boxes=True, numbers=True, rule=True, stand_alone=False):
    """Returns TiKz code for displaying the rule."""
    # Something like:
    #   http://mathworld.wolfram.com/ElementaryCellularAutomaton.html
    #   always include blocks
    #   option: containing boxes True|False
    #   option: numbers below    True|False
    #   option: rule in decimal  True|False
    full_tex = r"""
    \documentclass{{article}}
    \usepackage{{tikz}}

    \begin{{document}}

    {tikz_code}

    \end{{document}}"""

    tikz_code=r"""
    \begin{{tikzpicture}}[node distance=15pt]
    \tikzstyle{{b}}=[draw=black,fill=black]
    \tikzstyle{{w}}=[draw=black,fill=white]
    \tikzstyle{{mstyle}}=[draw={borders},column sep=1pt,row sep=1pt]

    %Layout the blocks containing the rule
    \matrix [mstyle]{{

    \node[b]  {{}}; & \node[b] {{}}; & \node[b] {{}};\\
                               & \node[{bw_boxes[0]}] (o1) {{}}; & \\
    }};

    \matrix [mstyle] at (30pt,0){{

    \node[b]  {{}}; & \node[b] {{}}; & \node[w] {{}};\\
                               & \node[{bw_boxes[1]}] (o2) {{}}; & \\
    }};

    \matrix [mstyle] at (60pt,0){{

    \node[b]  {{}}; & \node[w] {{}}; & \node[b] {{}};\\
                               & \node[{bw_boxes[2]}] (o3) {{}}; & \\
    }};

    \matrix [mstyle] at (90pt,0){{

    \node[b]  {{}}; & \node[w] {{}}; & \node[w] {{}};\\
                               & \node[{bw_boxes[3]}] (o4) {{}}; & \\
    }};

    \matrix [mstyle] at (120pt,0){{

    \node[w]  {{}}; & \node[b] {{}}; & \node[b] {{}};\\
                               & \node[{bw_boxes[4]}] (o5) {{}}; & \\
    }};

    \matrix [mstyle] at (150pt,0){{

    \node[w]  {{}}; & \node[b] {{}}; & \node[w] {{}};\\
                               & \node[{bw_boxes[5]}] (o6) {{}}; & \\
    }};

    \matrix [mstyle] at (180pt,0){{

    \node[w]  {{}}; & \node[w] {{}}; & \node[b] {{}};\\
                               & \node[{bw_boxes[6]}] (o7) {{}}; & \\
    }};

    \matrix [mstyle] at (210pt,0){{
    \node[w]  {{}}; & \node[w] {{}}; & \node[w] {{}};\\
                               & \node[{bw_boxes[7]}] (o8) {{}}; & \\
    }};

    {title}

    {bit_labels}

    \end{{tikzpicture}}
    """

    title_tex=r"\node at (105pt,20pt) {{{title}}};"

    bit_label_tex=r"""
    %Numbers under the blocks
    \node [below of=o1]{{{0}}};
    \node [below of=o2]{{{1}}};
    \node [below of=o3]{{{2}}};
    \node [below of=o4]{{{3}}};
    \node [below of=o5]{{{4}}};
    \node [below of=o6]{{{5}}};
    \node [below of=o7]{{{6}}};
    \node [below of=o8]{{{7}}};
    """

    bit_string = '{0:08b}'.format(eca.rule)
    colors = ['w','b']
    bw_boxes = [colors[int(i)] for i in bit_string]

    if rule:
        title = title_tex.format(title='Rule %i' %(eca.rule))
    else:
        title = ''

    if numbers:
        bit_labels = bit_label_tex.format(*bit_string)
    else:
        bit_labels = ''

    if boxes:
        borders = 'black'
    else:
        borders = 'white'

    options = {'title':title,'borders':borders,'bit_labels':bit_labels,
                'bw_boxes':bw_boxes}

    tikz_code = tikz_code.format(**options)

    if stand_alone:
        return full_tex.format(tikz_code=tikz_code)
    else:
        return tikz_code

def show_rule(eca, boxes=True, numbers=True, rule=True):
    """Show the rule."""
    # matplotlib
    pass

def show_lightcone(eca, cell):
    """Show the light cone for the specified cell.

    The darker the cell, the more times it has been counted.

    Parameters
    ----------
    cell : 2-tuple
        A 2-tuple (t, s) where t is the row index and s is the column
        index of the cell.

    """
    # matplotlib
    pass

def show_twolightcones(eca, cell1, cell2, color1=None, color2=None, color3=None):
    """Shows two light cones.

    The intent is to highlight the intersection of two light cones.

    Parameters
    ----------
    cell1, cell2 : 2-tuple
        A 2-tuple (t, s) where t is the row index and s is the column
        index of the cell.

    color1 : Matplotlib color
        The color to use for cell1's light cone.
    color2 : Matplotlib color
        The color to use for cell2's light cone.
    color3 : Matplotlib color
        The color to use for the intersection of the light cones of
        cell1 and cell2.

    """
    # matplotlib
    pass

