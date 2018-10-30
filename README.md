# pytorch_DSOD_RFBs_variants

Thanks to [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [uoip/SSD-variants](https://github.com/uoip/SSD-variants).

`RFBs`（Revese feature blocks） refers to those similar module designs, like FPN, RON, DSSD etc.<br>
Motivation of this program is that I was confused by the “Effectiveness” of those various operations. Many unanswered questions make the process of designing a net like a mystery.<br>

Which is the best way for upsampling, nearest? bilinear? Deconv?<br>
Do we need to re-extract features inside RFBs？<br>
Dose it really contribute that using compliated Inception blocks between feature map outputs and final predict layers？<br> 

Thus, I did some experiments to explore the above questions.<br>

I use
