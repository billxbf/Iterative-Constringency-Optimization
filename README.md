# Iterative Constringency Optimization
ICO is a preprocessing framework to enhance model performance on agent interactive data. Agent interactive data involves two agent data sets and one interaction data set, where the agent data contains private features only related with the entity itself, and the interaction data contains interactive features of two agents. The main idea behind ICO is to fully exploit respective features of two agents before modeling by clustering two agents into groups, hoping to find potential relationships among groups of agents. To determine a proper clustering solution, we optimize Constringency, a quantity describing the overall strength of correlation between groups of two agents.

## Paperwork
[Iterative Constringency Optimization](https://github.com/billxbf/Iterative-Constringency-Optimization/blob/master/ICO_Paper.pdf)


### Prerequisites

Following packages are required to run ICO.
* Numpy
* Pandas
* Scikit-learn
* Seaborn
* imblearn
* scipy
* lightgbm

### Running Test
API and documentation are not available for now. Core functions can be found in HM.py. Some test examples mentioned in the paper can be found [here](). 
