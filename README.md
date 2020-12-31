# Flower_Super_Resolution_Deep_Learning
Test on super resolution using perceptual loss

## Model

Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 256, 256, 3) 0                                            
__________________________________________________________________________________________________
g_e1 (Conv2D)                   (None, 256, 256, 64) 9472        input_2[0][0]                    
__________________________________________________________________________________________________
g_e1_bn (InstanceNormalization) (None, 256, 256, 64) 2           g_e1[0][0]                       
__________________________________________________________________________________________________
activation (Activation)         (None, 256, 256, 64) 0           g_e1_bn[0][0]                    
__________________________________________________________________________________________________
g_e2 (Conv2D)                   (None, 128, 128, 128 73856       activation[0][0]                 
__________________________________________________________________________________________________
g_e2_bn (InstanceNormalization) (None, 128, 128, 128 2           g_e2[0][0]                       
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 128, 128, 128 0           g_e2_bn[0][0]                    
__________________________________________________________________________________________________
g_e3 (Conv2D)                   (None, 64, 64, 256)  295168      activation_1[0][0]               
__________________________________________________________________________________________________
g_e3_bn (InstanceNormalization) (None, 64, 64, 256)  2           g_e3[0][0]                       
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 64, 64, 256)  0           g_e3_bn[0][0]                    
__________________________________________________________________________________________________
g_r1 (Conv2D)                   (None, 64, 64, 256)  590080      activation_2[0][0]               
__________________________________________________________________________________________________
g_r1_bn (InstanceNormalization) (None, 64, 64, 256)  2           g_r1[0][0]                       
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 64, 64, 256)  0           g_r1_bn[0][0]                    
__________________________________________________________________________________________________
g_r1_2 (Conv2D)                 (None, 64, 64, 256)  590080      activation_3[0][0]               
__________________________________________________________________________________________________
g_r1_bn2 (InstanceNormalization (None, 64, 64, 256)  2           g_r1_2[0][0]                     
__________________________________________________________________________________________________
add (Add)                       (None, 64, 64, 256)  0           g_r1_bn2[0][0]                   
                                                                 activation_2[0][0]               
__________________________________________________________________________________________________
g_r2 (Conv2D)                   (None, 64, 64, 256)  590080      add[0][0]                        
__________________________________________________________________________________________________
g_r2_bn (InstanceNormalization) (None, 64, 64, 256)  2           g_r2[0][0]                       
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 64, 64, 256)  0           g_r2_bn[0][0]                    
__________________________________________________________________________________________________
g_r2_2 (Conv2D)                 (None, 64, 64, 256)  590080      activation_4[0][0]               
__________________________________________________________________________________________________
g_r2_bn2 (InstanceNormalization (None, 64, 64, 256)  2           g_r2_2[0][0]                     
__________________________________________________________________________________________________
add_1 (Add)                     (None, 64, 64, 256)  0           g_r2_bn2[0][0]                   
                                                                 add[0][0]                        
__________________________________________________________________________________________________
g_r3 (Conv2D)                   (None, 64, 64, 256)  590080      add_1[0][0]                      
__________________________________________________________________________________________________
g_r3_bn (InstanceNormalization) (None, 64, 64, 256)  2           g_r3[0][0]                       
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 64, 64, 256)  0           g_r3_bn[0][0]                    
__________________________________________________________________________________________________
g_r3_2 (Conv2D)                 (None, 64, 64, 256)  590080      activation_5[0][0]               
__________________________________________________________________________________________________
g_r3_bn2 (InstanceNormalization (None, 64, 64, 256)  2           g_r3_2[0][0]                     
__________________________________________________________________________________________________
add_2 (Add)                     (None, 64, 64, 256)  0           g_r3_bn2[0][0]                   
                                                                 add_1[0][0]                      
__________________________________________________________________________________________________
g_r4 (Conv2D)                   (None, 64, 64, 256)  590080      add_2[0][0]                      
__________________________________________________________________________________________________
g_r4_bn (InstanceNormalization) (None, 64, 64, 256)  2           g_r4[0][0]                       
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 64, 64, 256)  0           g_r4_bn[0][0]                    
__________________________________________________________________________________________________
g_r4_2 (Conv2D)                 (None, 64, 64, 256)  590080      activation_6[0][0]               
__________________________________________________________________________________________________
g_r4_bn2 (InstanceNormalization (None, 64, 64, 256)  2           g_r4_2[0][0]                     
__________________________________________________________________________________________________
add_3 (Add)                     (None, 64, 64, 256)  0           g_r4_bn2[0][0]                   
                                                                 add_2[0][0]                      
__________________________________________________________________________________________________
g_r5 (Conv2D)                   (None, 64, 64, 256)  590080      add_3[0][0]                      
__________________________________________________________________________________________________
g_r5_bn (InstanceNormalization) (None, 64, 64, 256)  2           g_r5[0][0]                       
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 64, 64, 256)  0           g_r5_bn[0][0]                    
__________________________________________________________________________________________________
g_r5_2 (Conv2D)                 (None, 64, 64, 256)  590080      activation_7[0][0]               
__________________________________________________________________________________________________
g_r5_bn2 (InstanceNormalization (None, 64, 64, 256)  2           g_r5_2[0][0]                     
__________________________________________________________________________________________________
add_4 (Add)                     (None, 64, 64, 256)  0           g_r5_bn2[0][0]                   
                                                                 add_3[0][0]                      
__________________________________________________________________________________________________
g_r6 (Conv2D)                   (None, 64, 64, 256)  590080      add_4[0][0]                      
__________________________________________________________________________________________________
g_r6_bn (InstanceNormalization) (None, 64, 64, 256)  2           g_r6[0][0]                       
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 64, 64, 256)  0           g_r6_bn[0][0]                    
__________________________________________________________________________________________________
g_r6_2 (Conv2D)                 (None, 64, 64, 256)  590080      activation_8[0][0]               
__________________________________________________________________________________________________
g_r6_bn2 (InstanceNormalization (None, 64, 64, 256)  2           g_r6_2[0][0]                     
__________________________________________________________________________________________________
add_5 (Add)                     (None, 64, 64, 256)  0           g_r6_bn2[0][0]                   
                                                                 add_4[0][0]                      
__________________________________________________________________________________________________
g_r7 (Conv2D)                   (None, 64, 64, 256)  590080      add_5[0][0]                      
__________________________________________________________________________________________________
g_r7_bn (InstanceNormalization) (None, 64, 64, 256)  2           g_r7[0][0]                       
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 64, 64, 256)  0           g_r7_bn[0][0]                    
__________________________________________________________________________________________________
g_r7_2 (Conv2D)                 (None, 64, 64, 256)  590080      activation_9[0][0]               
__________________________________________________________________________________________________
g_r7_bn2 (InstanceNormalization (None, 64, 64, 256)  2           g_r7_2[0][0]                     
__________________________________________________________________________________________________
add_6 (Add)                     (None, 64, 64, 256)  0           g_r7_bn2[0][0]                   
                                                                 add_5[0][0]                      
__________________________________________________________________________________________________
g_r8 (Conv2D)                   (None, 64, 64, 256)  590080      add_6[0][0]                      
__________________________________________________________________________________________________
g_r8_bn (InstanceNormalization) (None, 64, 64, 256)  2           g_r8[0][0]                       
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 64, 64, 256)  0           g_r8_bn[0][0]                    
__________________________________________________________________________________________________
g_r8_2 (Conv2D)                 (None, 64, 64, 256)  590080      activation_10[0][0]              
__________________________________________________________________________________________________
g_r8_bn2 (InstanceNormalization (None, 64, 64, 256)  2           g_r8_2[0][0]                     
__________________________________________________________________________________________________
add_7 (Add)                     (None, 64, 64, 256)  0           g_r8_bn2[0][0]                   
                                                                 add_6[0][0]                      
__________________________________________________________________________________________________
g_r9 (Conv2D)                   (None, 64, 64, 256)  590080      add_7[0][0]                      
__________________________________________________________________________________________________
g_r9_bn (InstanceNormalization) (None, 64, 64, 256)  2           g_r9[0][0]                       
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 64, 64, 256)  0           g_r9_bn[0][0]                    
__________________________________________________________________________________________________
g_r9_2 (Conv2D)                 (None, 64, 64, 256)  590080      activation_11[0][0]              
__________________________________________________________________________________________________
g_r9_bn2 (InstanceNormalization (None, 64, 64, 256)  2           g_r9_2[0][0]                     
__________________________________________________________________________________________________
add_8 (Add)                     (None, 64, 64, 256)  0           g_r9_bn2[0][0]                   
                                                                 add_7[0][0]                      
__________________________________________________________________________________________________
g_d1_dc (Conv2DTranspose)       (None, 128, 128, 128 295040      add_8[0][0]                      
__________________________________________________________________________________________________
g_d1_dc_bn (InstanceNormalizati (None, 128, 128, 128 2           g_d1_dc[0][0]                    
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 128, 128, 128 0           g_d1_dc_bn[0][0]                 
__________________________________________________________________________________________________
g_d2_dc (Conv2DTranspose)       (None, 256, 256, 64) 73792       activation_12[0][0]              
__________________________________________________________________________________________________
g_d2_dc_bn (InstanceNormalizati (None, 256, 256, 64) 2           g_d2_dc[0][0]                    
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 256, 256, 64) 0           g_d2_dc_bn[0][0]                 
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 256, 256, 3)  9411        activation_13[0][0]              
==================================================================================================
Total params: 11,378,225
Trainable params: 11,378,225
Non-trainable params: 0
__________________________________________________________________________________________________


## Results

Last epochs inferences. First image is the original, second the low-res input and last the prediction.

![](./results/epoch_5800.jpg)
![](./results/epoch_5900.jpg)
