# pytorch_DSOD_RFBs_variants

Thanks to [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [uoip/SSD-variants](https://github.com/uoip/SSD-variants).

`RFBs`（Revese feature blocks） refers to those similar module designs, like FPN, RON, DSSD etc.<br>
Motivation of this program is that I was confused by the “Effectiveness” of those various operations. Many unanswered questions make the process of designing a net like a mystery.<br>

Which is the best way for upsampling, nearest? bilinear? Deconv?<br>
Do we need to re-extract features inside RFBs？<br>
Dose it really contribute that using compliated Inception blocks between feature map outputs and final predict layers？<br> 

Thus, I did some experiments to explore the above questions.<br>

I use [DSOD_smallest](https://github.com/szq0214/DSOD) as a benckmark and modify its input into 320 and a new list of size of feature maps [40, 20, 10, 5, 3, 1]. And train with Adam and 16 batch size. The lr uses 2 options 1e-3 and 1e-4. Mostly, I use 1e-3 for fine-tune about 40-50 epoches and than, change to 1e-4 for unforzen model. 

Trained on VOC 07+12. Test on VOC 07.<br>
 &emsp;&emsp;&emsp;&emsp; Model &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; mAP<br>
DSOD 64/16 on Paper &emsp;&emsp;&emsp;&emsp;          73.6 <br>
My implement &emsp;&emsp;&emsp;&emsp;               72.7<br>
DSOD64/16 + FPN(Nearest) &emsp;&emsp;&emsp;&emsp;             72.4<br>
DSOD64/16 + FPN_Bilinear(with BN ReLu)   &emsp;&emsp;  73.8<br>
DSOD64/16 + FPN_Nearest(with BN ReLu)   &emsp;&emsp;  73.9<br>
DSOD64/16 + DSSD  &emsp;&emsp;&emsp;&emsp;   74.5<br>
DSOD64/16 + DSSD_s(remove 3x3 conv and BN)  &emsp;&emsp; 74.3<br>
DSOD64/16 + DSSD_s_SE [*SE block](https://arxiv.org/abs/1709.01507)  &emsp;&emsp; 75.2<br>

Then, I tried different predict layers to instead origin 3x3 Conv for cls and loc predict layer.<br>
1x1 will significantly down performance. Double 3x3 and other various Inception like blocks for enriching receptive fields contribute nothing.
