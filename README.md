###############################################################################
##### Filename: README.md for the code for paper 'https://arxiv.org/pdf/2411.12014'
##### Title: ON-THE-GO PATH PLANNING AND REPAIR IN STATIC AND DYNAMIC SCENARIOS
##### Author: Daniel Ajeleye
##### Code date: Nov 2023
###############################################################################

#------------------------------------------------------------------------------
##### How to run the code:
#------------------------------------------------------------------------------

To run this code, one has to provide the required parameters as inputs
to the involved functions, then one has to call:

    python3 planner_2d.py 

or 

    python3 planner_3d.py

in its directory.

Alternatively, press "run" in your IDE of choice (as long is it has a python 
extension) for this file. 

#------------------------------------------------------------------------------

***DEPENDENCIES:

Make sure that you have the "numpy", "matplotlib", "random", and "math" libraries 
installed, since this code will not run without these libraries. If you 
do not have those installed, you can get them with these commands in terminal:

    pip install numpy
    pip install math
    pip install matplotlib
    pip install random

#------------------------------------------------------------------------------

The scenario that runs is the one for Case 4, which allows the obstacles to follow 
a dynamics. Other cases could be uncomment as well to implement under the section:

#-----------------------------------------------------------------------------#
# Run the planner for the scenarios
#-----------------------------------------------------------------------------#

of the scripts. The code progressively
pop up the screenshots of the motion planned so far based on the provided number of steps. 
You may need to cancel(save) a provided figure in order to see the next.


