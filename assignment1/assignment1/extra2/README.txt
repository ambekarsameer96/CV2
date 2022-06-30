//////////// DATA/data FILES INCLUDED ///////////////////////

- ##########.jpeg file       : RGB images recorded
- ##########.pcd file        : point clouds recorded
- ##########_camera.xml file : camera parameters
- ##########_depth.png  file : depth images recorded
- ##########_normal.pcd file : normals extracted
- ##########_mask.jpeg  file : object masks


///////////////////////////////////////////////////

Note: The provided point clouds does not only consist of person (it also contains background).
Please apply a distance threshold to remove background (e.g. remove points further than 2meters to the camera).

In addition to the dataset, there are also 2 sets of dummy data which can be used to test your ICP algorithm.
These are:
wave_{source|target}.npy and bunny_{source|target}.npy

///////////////////////////////////////////////////
A little bit of example code is provided in SupplementalCode/, feel free to change as you see fit.

///////////////////////////////////////////////////
For a number of papers relevant to the assignment, a copy of the PDF is provided in Reading/
The pdfs are named such that the number at the start of the name of the file corresponds to the reference in the assignment.
