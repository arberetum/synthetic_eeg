# A Comparative Study of Machine Learning Models in Producing Synthetic iEEG Data for Epilepsy Analysis

This project investigates the generation of synthetic intracranial EEG (iEEG) data for epilepsy analysis using two advanced machine learning models. The goal is to address data scarcity challenges in EEG research and enhance the performance of classification models through the incorporation of synthetic data.

## Motivation
High-quality EEG data are often limited due to privacy and collection constraints, which poses significant challenges for research in seizure prediction and epilepsy analysis. This study focuses on generating synthetic iEEG data that mimics both interictal (non-seizure) and preictal (pre-seizure) states. Two complementary generative models are employed:
<ul>
  <li>Wasserstein Generative Adversarial Network (WGAN): Uses an adversarial framework with a critic network to generate realistic 10-second EEG segments.</li> 
  <li>Denoising Diffusion Probabilistic Model (DDPM): Leverages a diffusion process to iteratively refine noisy inputs toward producing high-fidelity synthetic EEG signals.</li>
</ul>


## Methodology
The project utilizes the [2014 American Epilepsy Society Seizure Prediction Challenge dataset](https://kaggle.com/competitions/seizure-prediction) (focused on canine EEG data) to train the models.  
<p align="center">
  <img width="396" alt="image" src="https://github.com/user-attachments/assets/ead3e98b-39cc-4c63-9fe6-92419091bd5b" />
</p>


Key aspects of the approach include:
- **Model Training:**  
  Both the **WGAN** and **DDPM** models were trained separately to generate synthetic data for interictal and preictal conditions. The **WGAN** employs the earth-mover distance as a loss metric to ensure stability, while the **DDPM** iteratively refines random noise through a learned diffusion process. As shown in Figure 1, the WGAN’s training and validation loss curves demonstrate convergence over 200 epochs. Similarly, Figure 2 illustrates the DDPM's loss curves, highlighting the stability of the diffusion process during training.
<p align="center">
  <img width="829" alt="image" src="https://github.com/user-attachments/assets/4c91a484-2c37-4d27-a476-c51d174283e0" />
  <img width="829" alt="image" src="https://github.com/user-attachments/assets/c0b03603-150e-46b2-938c-8feb3503f33a" />

</p>

- **Evaluation:**  
  The quality of the synthetic data was assessed through:
  - **Qualitative Analysis:** Visual comparison of time-domain signal characteristics and frequency-domain spectrograms.
  - **Quantitative Analysis:** Evaluation of classification performance by training models (e.g., Naïve Bayes, Ridge, and AdaBoost) on varying mixtures of real and synthetic data, with performance measured via metrics such as macro F1-score and AUC.


## Results and Discussion
### **Signal Quality:**  
  Both synthetic data generation methods captured key characteristics of real iEEG signals. However, subtle differences remain in spectral properties, particularly in low-frequency ranges for WGAN and higher variance noise levels for DDPM.  
    Figure 3 provides a visual comparison of the spectrograms from real and synthetic iEEG data. Notice how the synthetic signals, particularly from the DDPM, exhibit higher variance in the higher frequency bands compared to the real signals.
    <p align="center">
      <img width="818" alt="image" src="https://github.com/user-attachments/assets/15a11632-5afd-4a0a-952c-24b72d4c9a0f" />
    </p>

### **Classifier Performance:**  
  Experiments showed that classification performance is generally maintained when a balanced mix of real and synthetic data is used. Performance tends to decline when synthetic data exceeds about 60% of the training set, indicating that while the synthetic signals are informative, they may not fully replicate all nuanced features of real EEG recordings.  
  Table 3 summarizes the classifier performance across different metrics. Additionally, Figure 4 plots how the macro F1-score and AUC change as the proportion of synthetic data increases, underscoring the delicate balance required for optimal performance.
  <p align="center">
    <img width="797" alt="image" src="https://github.com/user-attachments/assets/cb2c491d-07f0-4d09-bdb1-56e701c80ec3" />
    <img width="791" alt="image" src="https://github.com/user-attachments/assets/b5e1ebf2-6b9f-4ef3-afed-bf55707f1bd8" />
    </p>

### **Insights:**  
  The findings underscore the potential of synthetic data to supplement limited real datasets. However, further refinement of the generation techniques is necessary to capture the detailed spectral and temporal features critical for robust seizure prediction.


## Conclusion
This study demonstrates that synthetic iEEG data generated via WGAN and DDPM approaches can effectively complement real datasets in epilepsy research. The models exhibit promising performance, yet the observed discrepancies in frequency characteristics highlight the need for continued advancements. Future work should focus on improving the fidelity of synthetic data to better serve diagnostic and predictive applications in neurology.

## Contributions
The authors contributed equally. [S. Kaushik](https://github.com/shoibolina) explored preprocessing methods and developed the DDPM model. [A. Maloney-Bertelli](https://github.com/arberetum) coded the classification models for evaluation as well as the WGAN.

---
<i>For a detailed account of the methodology, experiments, and analyses, please refer to the accompanying project report.</i>

---
### References
- Carrle, F. P., Hollenbenders, Y., & Reichenbach, A. (2023). Generation of synthetic EEG data for training algorithms supporting the diagnosis of major depressive disorder. Frontiers in Neuroscience.
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein Generative Adversarial Networks.
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
- Rasheed, K., et al. (2021). A generative model to synthesize EEG data for epileptic seizure prediction. IEEE Transactions on Neural Systems and Rehabilitation Engineering.
- Najafi, T., et al. (2022). A classification model of EEG signals based on RNN-LSTM for diagnosing focal and generalized epilepsy. Sensors.
- bbrinkm, Will Cukierski. (2014). American Epilepsy Society Seizure Prediction Challenge. Kaggle.
