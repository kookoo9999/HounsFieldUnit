# HounsFieldUnit
This module is scripted-modules of 3D Slicer.
If you use this module, you can extract HounsField Unit values (HU) from segmented(.nrrd) and volume(volume node).

Use "git clone" for download.
Execute 3D Slicer 4.13(stable released version) and open toolbar Edit - Application Settings.
Click ">>" button positioned "Additional module paths : " pannel and click "Add".
Import the directory you did git clone and push "OK" button.
Restart 3D Slicer and you can check this module at Exxamples - HounsFieldUnit.

For extract HU values , this module need segment(.nrrd) and its volumenode.

If you use volume for .stl , change format by using "Segmentation" module in 3D Slicer from .stl to .nrrd.
And edit vtkMRMLScalarVolumeNode's name to Segment , vtkMRMLSegmentationNode to Segment_1_1 in Data module.

After push "Get HU Values" button, you can see labelVolume segmented in Volume Display and csv file of result coordinates in c:/ .
