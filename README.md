# pytorch_DSOD_RFBs_variants

RFBs（Revese feature blocks） refers to those similar designs, like FPN, RON, DSSD etc.<br>
Motivation is that I was confused by the “Effectiveness” of those various operations. <br>
Many unanswered questions make the process of designing a net like a mystery.<br>

Which is the best way for upsampling, nearest? bilinear? Deconv?<br>
Do we need to re-extract features inside RFBs？<br>
Dose it really contribute that using compliated Inception blocks between feature map outputs and final predict layers？ 

