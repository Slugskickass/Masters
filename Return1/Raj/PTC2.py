#@ DatasetIOService ds
#@ UIService ui

import ij
import ij.IJ as IJ
from math import sqrt
import os
from ij.io import DirectoryChooser
from ij.gui import Plot
from java.awt import Color
#import ij.ImagePlus.getNSlices as getNSlices
 
def computeMean(pixels):
  return sum(pixels) / float(len(pixels))
 
def computeStdDev(pixels, mean):
  s = 0
  for i in range(len(pixels)):
    s += pow(pixels[i] - mean, 2)
  return sqrt(s / float(len(pixels) -1))

def ImagesMean(img,z):
  means = []
  for i in range(1, z+1):
	ip = img.getStack().getProcessor(i).convertToFloat()
	mean = computeMean(ip.getPixels())
	means.append(mean)
  return means

def ImagesStd(img, mean, z):
  stds = []
  for i in range(1, z+1):
  	ip = img.getStack().getProcessor(i).convertToFloat()
  	std = computeStdDev(ip.getPixels(), mean[i-1])
  	stds.append(std)
  return stds

def group_stds(liststds):
	std = 0
	for i in range(len(liststds)):
		std = std + (float(liststds[i])**2)
	std = std/len(liststds)
	return sqrt(std)

def get_file_list(dir, filetype = '.tif'):
    # dir = '/ugproj/Raj/Flash4/'
    file_list = []
    for root, directories, filenames in os.walk(dir):
    	for filename in filenames:
	        if not filename.endswith('.tif'):
	        	continue
	        path = os.path.join(root,filename)
		file_list.append(path)
	return file_list

def main(filelist):
	filemeans = []
	filestds = []
	
	for file in filelist:
	# load the dataset
		dataset = ds.open(file)
		
		z = dataset.dimension(2)
		
	# Load the dataset as an ImagepPlus/ImageStack
		dataset = IJ.openImage(file)
		
	# Means per frame and then mean of mean
		listmeans = ImagesMean(dataset, z)
		stackmeans = computeMean(listmeans)
		filemeans.append(stackmeans)
		
	# Stds per frame.
		liststds = ImagesStd(dataset,listmeans, z)
		grouped = group_stds(liststds)
		#std the std.
		filestds.append(grouped)
	
	return filemeans, filestds



# MAIN CODE
srcDir = DirectoryChooser("Choose").getDirectory()
filelist = get_file_list(srcDir, '.tif')
means, stds = main(filelist)


# PLOTTING
plot = Plot("PTC", "Mean", "Std")
plot.setLimits(0.00, 200.0, 0.00, 100.0)
plot.setColor(Color.BLUE)
plot.addPoints(means, stds, Plot.CROSS)
plot.show()
print means, stds

