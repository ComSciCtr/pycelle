from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

try:
    from _caalgo import eca_cyevolve, lightcone_counts
except ImportError:
    eca_cyevolve = None

class ECA(object):
    def __init__(self, rule):
        """Initialize the ECA.
        
        Parameters
        ----------
        rule : int
            The ECA rule to initialize.
            
        """
        if rule < 0 or rule >= 256:
            raise Exception('Rule must be between 0 and 255, inclusive.')
        else:
            self.rule = rule
            
            # Privately, store a list of the binary bits
            self._lookup = map(int, '{0:08b}'.format(rule))

        # spacetime array
        self.sta = None
        
        if eca_cyevolve is not None:
            self._cythonized = True
        else:
            self._cythonized = False


    def __repr__(self):
        return 'ECA({0})'.format(self.rule)

            
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

            
    def initialize(self, t=None, width=None, ic='random', clear=False):
        """Initialize the ECA's spacetime array.
        
        A new spacetime array is allocated only if necessary.
        
        Parameters
        ----------
        t : int | None
            The maximum number of evolutions for the ECA. The number of rows in
            the spacetime array will be t+1.  If the ECA has been previously
            initialized, then `t` can be `None` and the previous value will
            be used.
            
        width : int | str
            The width of the spacetime array. Recall that the radius of a cell
            at time t >= 0 goes as r(t) = 1 + 2t.  Note that if the width is 
            too small and time is too large, then the light cone of a cell will 
            intersect with itself. In addition to specifying an integer, a 
            number of peset options are available, each of which depends on the
            value of t. If `width` is `None`, then 'square' is used. 
            
            Valid options:
            
                'square'
                    The spacetime array is a square. So the width is set equal
                    to the nearest odd integer greater than t+1.
                    
                'double'
                    The width is set equal to the nearest odd integer greater
                    than t/2.  So the spacetime array is twice as tall as it
                    is wide.
                    
                'no_intersections'
                    The width is choosen so that the light cone of any cell
                    does not intersect with itself. Explicitly, the past light
                    cone will self-intersection whenever: t >= ceil(width/2).
                    So, we set the width to be width = 1 + 2t. Effectively,
                    this means the spacetime array will be twice as wide as it
                    is tall.
                    
                The focus on the 'nearest odd integer' is so that an initial
                condition can easily begin with a single black node.
                
        ic : str
            A specification of how to initialize the spacetime array.
            If some other initialization is desired, one can explicitly
            set the first row of the spacetime array via self.sta[0].
            
            Valid options:
                
                'random'
                    The spacetime array is initialized randomly.
                
                'single'
                    The spacetime array has a single, centered black cell.
                    Cell (0, floor(width/2)) is filled black.
                    
        clear : bool
            If `True`, then the rest of the spacetime array is cleared.
            Usually it is not necessary to do this since the values will
            be overridden during evolution.
                    
        """
        if t is None:
            if self.sta is None:
                msg = 'Spacetime array is not initialized. Specify t.'
                raise Exception(msg)
            else:
                # Subtract one, since t represents the number of evolutions.
                t = self.sta.shape[0] - 1
        t = int(t)
                
        if width is None:
            width = 'square'
        
        try:
            width = int(width)
        except ValueError:  
            # Then it is some preset option.
            if width == 'square':
                width = t+1
            elif width == 'double':
                width = int(t/2)
            elif width == 'no_intersections':
                width = 1 + 2*t # guaranteed to be odd
            else:
                raise Exception('Invalid `width` specification.')
                
            # make it odd
            if width % 2 == 0:
                width += 1
            
        shape = (t+1, width)
        if self.sta is None or self.sta.shape != shape:
            # allocate a new array
            if clear:
                self.sta = np.zeros(shape, dtype=np.int32, order='C')
            else:
                self.sta = np.empty(shape, dtype=np.int32, order='C')
        elif clear:
            # otherwise, clear it if desired (keeping the original array)
            self.sta *= 0
            
        if ic == 'random':
            self.sta[0] = np.random.randint(0, 2, size=self.sta.shape[1])
        elif ic == 'single':
            self.sta[0] *= 0
            self.sta[0][int(np.floor(width/2))] = 1
        else:
            raise Exception('Invalid `ic` specificiation.')
            

    def evolve(self, t=None, draw=True):
        """Evolves the cellular automaton from the inital row by t time steps.
        
        Parameters
        ----------
        t : int | None
            If `None`, then the cellular automaton is evolved to fill the
            allocated spacetime array.
            
        draw : bool
            If `True`, then draw the ECA after evolving it.
            
        """
        self._verify_initialized()  
        if t is None:
            t = self.sta.shape[0] - 1  
            
        if self._cythonized:
            self._evolve_cython(t)
        else:        
            self._evolve_python(t)

        if draw:
            draw_spacetime(self)


    def _verify_initialized(self):
        """Internal function to make sure the spacetime array is initialized."""
        if self.sta is None:
            raise Exception('Spacetime array is not initialized.')
        

    def _evolve_python(self, iterations):
        for i,row in enumerate(self.sta[:-1]):
            windows = np.vstack([np.roll(row,1), row, np.roll(row,-1)])
            windows = windows.transpose()
            for j,parents in enumerate(windows):
                self.sta[i+1,j] = self.eval(parents)

    def _evolve_cython(self, iterations):
        eca_cyevolve(self._lookup, self.sta, iterations)

    
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

    if rule==True:
        title = title_tex.format(title='Rule %i' %(eca.rule))
    else:
        title = ''
    
    if numbers==True:
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
        return tikz_code
    else:
        return full_tex.format(tikz_code=tikz_code) 
    
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
    eca._verify_initialized()
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
    eca._verify_initialized()        
    # matplotlib
    pass

    
def draw_spacetime(eca, twindow=None, xwindow=None, ax=None):
    """Show the existing spacetime array.
    
    Parameters
    ----------
    eca : ECA
        The evolved ECA to display.
    twindow : 2-tuple
        The start and end time indexes to display.
    xwindow : 2-tuple
        The start and end space indexes to display.
    ax : Matplotlib Axes | None
        The axis to receive the plot.
        
    """
    eca._verify_initialized()
    if twindow is None and xwindow is None:
        arr = eca.sta
    else:
        if twindow is None:
            twindow = (0, eca.sta.shape[0])
        if xwindow is None:
            xwindow = (0, eca.sta.shape[1])
        arr = eca.sta[twindow[0]:twindow[1], xwindow[0]:xwindow[1]]
        
    if ax is None:
        ax = plt.gca()
        
    ax.matshow(arr, cmap=plt.cm.gray_r)
    ax.set_title('Rule {0}'.format(eca.rule))
    plt.draw()

