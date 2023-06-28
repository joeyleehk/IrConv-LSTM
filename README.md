# Improving short-term bike sharing demand forecast through an irregular convolutional neural network
An irregular convolution Long Short-Term Memory network to predict shared bicycle usage in urban areas during the next one hour.
 
_Li X, Xu Y, Zhang X, et al. Improving short-term bike sharing demand forecast through an irregular convolutional neural network[J]. Transportation Research Part C: Emerging Technologies, 2023, 147: 103984._
# Abstract
In recent years, many deep learning algorithms have been introduced to improve bicycle usage forecast. A typical
practice is to integrate convolutional (CNN) and recurrent neural network (RNN) to capture spatial-temporal dependency in historical travel demand. 
For typical CNN, the convolution operation is conducted through a kernel that moves across a “matrix-format” city to extract features over spatially
adjacent urban areas. This practice assumes that areas close to each other could provide useful
information that improves prediction accuracy. However, bicycle usage in neighboring areas might
not always be similar, given spatial variations in built environment characteristics and travel behavior that affect cycling activities. 
Yet, areas that are far apart can be relatively more similar in temporal
usage patterns. 
 
To utilize the hidden linkage among these distant urban areas, the study proposes
an **Irregular Convolutional Long Short-Term Memory model (IrConv+LSTM)** to improve short-term
bike sharing demand forecast. The model modifies traditional CNN with irregular convolutional
architecture to extract dependency among “semantic neighbors”. 
 
### **This study has been published on [Transportation Research Part C: Emerging Technologies](https://doi.org/10.1016/j.trc.2022.103984).**
 
# Model Architecture
 <div align=center><img src="https://github.com/joeyleehk/IrConv-LSTM/blob/master/architecture.jpg" width="600" height="580" alt="Model Architecture"/></div>
  
IrConv+LSTM contains three modules with the same structure. Each module adopts three layers of irregular
convolutional architecture to capture the characteristics of bicycle demand among urban areas based on the semantic neighbors. The
vector sequence formed by flattening the output of the irregular convolution is used as the input to
the LSTM model to extract the temporal information in the sequence. The outputs of three hybrid
modules are fed into a feature fusion layer. The output of the feature fusion layer is activated by a
non-linear function generating the predicted value. 
 
The semantic neighbors used in irregular convolution are selected based on the similarity of bicycle usage patterns. Unlike spatial neighbors in regular convolution,  the semantic neighbor can be located in any urban area. For specific definitions of semantic neighbors, please refer to Section 4.2 in the [paper](https://arxiv.org/abs/2202.04376). 
<div align=center><img src="https://github.com/joeyleehk/IrConv-LSTM/blob/master/neighbors.jpg" width="655" height="348" alt="Semantic Neighbors"/></div>
 
# Datasets
 
We select five bike-sharing datasets to evaluate the robustness and reliability of our proposed model, 
including one dockless bike-sharing system in _Singapore_, and four station-based systems in _Chicago, Washington, D.C., London, and New York_. 
The data for station-based systems are open for download, such as [DivvyBike](https://ride.divvybikes.com/system-data) in Chicago, [CapitalBike](https://ride.capitalbikeshare.com/system-data) in D.C., and [CitiBike](https://ride.citibikenyc.com/system-data) in New York. 
 
# Overall Accuracy
<div align=center><img src="https://github.com/joeyleehk/IrConv-LSTM/blob/master/overall accuracy.png" width="600" height="348" alt="Overall Accuracy"/></div>

According the overall accuracy on all datasets, the proposed model, IrConv+LSTM, generally performs better than a collection of benchmark models, especially for CNN+LSTM model leveraging spatial dependency from spatially adjacent areas. Moreover, the semantic and spatial neighbors are very different from each other. For more detials, please refer to _Section 5.4_ in the [paper](https://arxiv.org/abs/2202.04376). 

# Running the model
### Required Packages
Pytorch, numpy, pandas, math, and tensorboardX.
### File description
This repository provided a New York dataset used in the study, including the [bicycle usage data](./NYC/nyc_raw_data.npy) and [a look-up table](./NYC/DTW_Similarity_Table.csv.npy) that queries the semantic neighbors corresponding to all  predicted urban areas (measured by Dynamic Time Warping).  
 
Here is the description of each project file:
 
**model/irregular_convolution_LSTM.py**: The implementation of the proposed deep learning architecture. 
 
**training_model.py**: Train the irregular convolution LSTM model on the New York dataset.
 
**evaluation_model.py**: Evaluate the model and generate the prediction results during the validation period. 
 
**accuracy_assessment.py**: Calculate the three indicators (MAPE, MAE, and RMSE) to evaluate the model performance. 
 
**data.py**: Load historical bicycle usage data.
 
**NYC/DTW_Similarity_Table.csv**: A look-up table to query the semantic neighbor IDs (selected based on DTW distance) of each central cell involved in irregular convolution.
 
**NYC/nyc_raw_data.npy**: CitiBike in New York dataset which has been aggregated into regular grid map with 1km spatial resolution. 
 
Please feel free to share feedback, discuss topics or ask questions. (Email: lxyjoeylee@gmail.com)
 
# Citation
Li X, Xu Y, Zhang X, et al. Improving short-term bike sharing demand forecast through an irregular convolutional neural network[J]. Transportation Research Part C: Emerging Technologies, 2023, 147: 103984.


