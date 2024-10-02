The implementation is based on the theory of "3D Bounding Box Estimation Using Deep Learning and Geometry" by Arsalan Mousavian, Dragomir Anguelov, John Flynn, Jana Kosecka(https://arxiv.org/abs/1612.00496).<br>
Code modifies the repository "3D_detection" by Cersar(https://github.com/cersar/3D_detection?tab=readme-ov-file).<br>
Thanks lots for their contribution.<br>

<img src="https://github.com/Phonlin/Mo-Co-V2V/blob/main/demo.gif" width="100%" height="100%">

<h2>ABSTRACT</h2>
As autonomous driving technology advances, traditional single-vehicle perception systems may become a bottleneck due to insufficient information or misjudgments in complex road conditions or occlusions. This study aims to predict the 3D positions and orientations of other vehicles using a single-camera vision approach and share this information with other vehicles. By leveraging cameras on other vehicles for cooperative perception, we achieve a broader field of view and better recognition capabilities, maximizing driving safety. The implementation is based on the theory of "3D Bounding Box Estimation Using Deep Learning and Geometry" and modifies the repository "3D_detection" shared by Cersar on GitHub. Our proposed method uses deep learning and 2D geometry to predict 3D bounding boxes and angles from a single camera. When getting the position and angle of the vehicle, we convert it into a bird's eye view (BEV), and all the information generated in this process will be shared with other vehicles. When the camera on any vehicle is occluded, the vehicle’s BEV will still display the correct location and direction information of other vehicles. By using this method, it was found that the angle can be effectively restored when the target vehicle is occluded, with a recognition accuracy as high as 98.89% for the vehicle's side and a minimum of 95.28% for the front and rear. 

<h2>Useage</h2>
For video 3D bboxex compute, use:
<pre><code>python3 detection_video.py
</code></pre>
For single camera streaming, use:<br>
<pre><code>python3 detection_stream.py
</code></pre>
For two camera streaming, use:<br>
<pre><code>python3 detection_stream_for2.py
</code></pre>
