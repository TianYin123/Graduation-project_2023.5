# Graduation-project_2023.5
## Abstract
Autism Spectrum Disorder (ASD) is a developmental disorder that affects an individual's social interaction, language communication skills, and behavioral patterns. It typically appears in early childhood and lasts a lifetime. Since the exact cause of autism is unclear, diagnosis mainly relies on the subjective assessment of clinical professionals, which is complex and requires high professional standards. Therefore, an automated diagnostic method is urgently needed. In recent years, functional magnetic resonance imaging (fMRI), with its non-invasive, safe, and high-resolution advantages, has become a research focus and provides a new perspective for ASD diagnosis. This study uses a brain functional connectivity analysis method of resting-state functional magnetic resonance imaging to propose a multimodal feature fusion diagnostic strategy for ASD. First, a convolutional neural network (CNN) based on spatial attention and channel attention mechanisms is designed to extract functional connectivity features of Eickhoff-Zilles (EZ), Harvard-Oxford (HO), and Automated Anatomical Labeling atlas (AAL) time series in the ABIDE database. Then, a discriminative module based on the self-attention mechanism is designed to fuse and diagnose the three modal features. Through 5-fold cross-validation, the algorithm achieves an average accuracy of 70.95%, sensitivity of 74.42%, and specificity of 69.15%, thereby realizing the automatic, efficient, and accurate diagnosis of autism.

## Files explanation
- "graduation-paper-without page number.doc" is my graduation paper chinese version. If you want english version, let me know through 877265440@qq.com or comments.
- "presentation_PPT.pptx" is the presentation ppt of my final presentation chinese version. If you want english version, let me know through 877265440@qq.com or comments.
- "BestStructure-3attention" is the final version of the best outcome in accuracy. The outline of the structure has been explained in the abstract.
  - local means run in my computer.
  - colab mean run on the google colabs.
- "OtherAttempts"
  - Before I found the best strategy of building the model, I made a lot attempts.
  - "2DImg mix 3DImg-History" folder means I used to mix 3D feature of FMRI with 2D feature of FMRI to get a better outcome. I almost successed, but because the deadline was coming soon, I have no time to improve the overfit of self-attention in 3D CNN model. If you are interested in the further study, let me know through 877265440@qq.com or comments
  - "Only2D-History" folder records how I successfully choose the right model and data to get the best outcome.

 **I use abide-master to download the data. You can find it in the official site of abide dataset.**  
 **The .pth file of my model is too big(55.3mb per .pth file) so I can not upload on github. But if you want it, let me know**  
 **If you have other questiones, let me know through 877265440@qq.com or comments.**  
 
